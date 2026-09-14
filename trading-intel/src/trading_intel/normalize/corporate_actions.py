"""企業行動引擎：未調整原始價 + 獨立調整因子表。

SPEC 3.2 第 1 點禁止直接儲存調整後價格，理由是「新的企業行動會讓歷史調整價
全部改變，破壞可重現性」。

具體的失敗長這樣：今天存下 2330 的調整後價，明天它除息，昨天存的每一筆歷史
調整價就都變了。用那份資料跑的回測，昨天和今天會得到不同答案，而且沒有任何
地方會報錯。

本模組的做法是把「事實」與「詮釋」分開：
- 原始價（``RawPrice``）是事實，落地後不再變動；
- 調整因子（``CorporateAction``）是詮釋，每一筆都帶自己的 ``ingest_time``；
- 調整後價格是查詢時才算出來的，且算出什麼取決於 ``asof``。

因此同一段歷史在除息公告發布前後查詢，會得到不同的結果——這正是
SPEC 第 11 節 Phase 1 的驗收條件之一。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from trading_intel.core.clock import ensure_utc
from trading_intel.core.errors import LookaheadError
from trading_intel.core.ids import EntityId


class ActionType(StrEnum):
    """企業行動類別。值等於名稱，方便入庫與日誌。"""

    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    CAPITAL_REDUCTION = "CAPITAL_REDUCTION"
    MERGER = "MERGER"


class ActionSource(StrEnum):
    """因子的來源，決定它可以被多信任。"""

    #: 來自交易所公告，是事實。
    OFFICIAL = "OFFICIAL"
    #: 由價格跳動反推，是推測。台股有漲跌幅限制，單日跳動超過限制必為企業行動，
    #: 但「是哪一種、確切比例多少」是猜的。
    INFERRED = "INFERRED"
    #: 由既有已調整價序列繼承而來，無法還原成因。
    LEGACY = "LEGACY"


class CorporateAction(BaseModel):
    """一筆企業行動，以及它對歷史價格的乘性調整因子。

    ``factor`` 的定義：除權息日之前的原始價要乘以 ``factor`` 才能與之後的價格
    在同一個尺度上比較。現金股利 5 元、除息前收盤 100 元，則 factor = 0.95。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: EntityId
    action_type: ActionType
    #: 除權除息交易日。這一天（含）之後的價格不需要調整。
    ex_date: date
    factor: Decimal = Field(gt=0)
    source: ActionSource
    #: 對這筆因子的信心。OFFICIAL 為 1.0，INFERRED 依偵測強度給值。
    confidence: float = Field(ge=0, le=1)
    #: 我們何時得知這筆企業行動。point-in-time 查詢據此過濾。
    ingest_time: AwareDatetime
    detail: str = ""

    @model_validator(mode="after")
    def _validate_source_confidence(self) -> Self:
        if self.source is ActionSource.OFFICIAL and self.confidence != 1.0:
            msg = "官方公告的 confidence 必須為 1.0"
            raise ValueError(msg)
        if self.source is not ActionSource.OFFICIAL and self.confidence >= 1.0:
            msg = "非官方來源的 confidence 不得為 1.0，那會讓推測看起來像事實"
            raise ValueError(msg)
        return self


class RawPrice(BaseModel):
    """未調整的原始收盤價。落地之後就是事實，不再變動。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: EntityId
    trade_date: date
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    #: 這筆價格何時進入系統。回測以此過濾，杜絕前視。
    ingest_time: AwareDatetime


class AdjustedPrice(BaseModel):
    """查詢時算出來的調整後價格，帶著它是怎麼算出來的。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: EntityId
    trade_date: date
    raw_close: Decimal
    adjusted_close: Decimal
    cumulative_factor: Decimal
    #: 參與這次調整的企業行動數量。0 代表沒有任何調整。
    applied_actions: int
    #: 參與調整的因子裡最低的信心值。1.0 代表全部來自官方公告。
    min_confidence: float


class CorporateActionStore:
    """企業行動的 point-in-time 儲存體。

    刻意只提供帶 ``asof`` 的查詢介面——沒有「取全部因子」的方法，
    因為那正是前視偏誤最容易溜進來的地方（CLAUDE.md 第 2 條）。
    """

    def __init__(self, actions: Iterable[CorporateAction] = ()) -> None:
        self._actions: list[CorporateAction] = []
        for action in actions:
            self.add(action)

    def add(self, action: CorporateAction) -> None:
        self._actions.append(action)
        self._actions.sort(key=lambda item: (item.entity_id, item.ex_date, item.ingest_time))

    def __len__(self) -> int:
        return len(self._actions)

    def actions_for(
        self,
        entity_id: EntityId,
        *,
        asof: datetime,
    ) -> list[CorporateAction]:
        """取 ``asof`` 當下已知的、屬於該標的的所有企業行動。

        注意過濾的是 ``ingest_time`` 而非 ``ex_date``：一個 3 月除息、
        4 月才公告的行動，在 3 月的回測裡是不存在的。
        """
        boundary = ensure_utc(asof)
        return [
            action
            for action in self._actions
            if action.entity_id == entity_id and action.ingest_time <= boundary
        ]

    def cumulative_factor(
        self,
        entity_id: EntityId,
        trade_date: date,
        *,
        asof: datetime,
    ) -> tuple[Decimal, int, float]:
        """回傳 ``trade_date`` 當天價格應乘上的累積因子。

        只有除權息日**晚於** ``trade_date`` 的行動會影響它——因為調整的意義是
        「把過去的價格拉到現在的尺度」。
        """
        applicable = [
            action
            for action in self.actions_for(entity_id, asof=asof)
            if action.ex_date > trade_date
        ]
        factor = Decimal("1")
        for action in applicable:
            factor *= action.factor
        confidence = min((action.confidence for action in applicable), default=1.0)
        return factor, len(applicable), confidence


def get_adjusted_prices(
    prices: Sequence[RawPrice],
    store: CorporateActionStore,
    *,
    entity_id: EntityId,
    start: date,
    end: date,
    asof: datetime,
) -> list[AdjustedPrice]:
    """SPEC 第 11 節 Phase 1 指定的介面：以 ``asof`` 為界的調整後價格。

    三道過濾缺一不可：
    1. ``ingest_time <= asof`` — 還沒送達我們手上的價格不存在；
    2. ``start <= trade_date <= end`` — 呼叫端要的區間；
    3. 因子只取 ``asof`` 當下已知的 — 未來才公告的除息不影響過去的判斷。
    """
    boundary = ensure_utc(asof)
    selected = [
        price
        for price in prices
        if price.entity_id == entity_id
        and price.ingest_time <= boundary
        and start <= price.trade_date <= end
    ]
    # 同一天可能有多筆（更正），取 asof 之前最後送達的那一筆。
    latest_by_date: dict[date, RawPrice] = {}
    for price in sorted(selected, key=lambda item: item.ingest_time):
        latest_by_date[price.trade_date] = price

    results: list[AdjustedPrice] = []
    for trade_date in sorted(latest_by_date):
        price = latest_by_date[trade_date]
        factor, applied, confidence = store.cumulative_factor(entity_id, trade_date, asof=boundary)
        results.append(
            AdjustedPrice(
                entity_id=entity_id,
                trade_date=trade_date,
                raw_close=price.close,
                adjusted_close=price.close * factor,
                cumulative_factor=factor,
                applied_actions=applied,
                min_confidence=confidence,
            )
        )
    return results


def assert_no_lookahead(prices: Sequence[RawPrice], asof: datetime) -> None:
    """確認一批價格中沒有任何一筆晚於 ``asof`` 才送達。

    給資料管線在交界處自我檢查用；正常路徑上 ``get_adjusted_prices`` 已經濾掉了。
    """
    boundary = ensure_utc(asof)
    offenders = [price for price in prices if price.ingest_time > boundary]
    if offenders:
        raise LookaheadError(
            "資料集含有 asof 之後才送達的價格",
            asof=boundary.isoformat(),
            offender_count=len(offenders),
            first_offender=offenders[0].ingest_time.isoformat(),
        )
