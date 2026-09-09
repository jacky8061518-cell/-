"""可交易宇宙：每支標的每一天是否可交易的完整紀錄。

SPEC 3.2 第 2 點要求無存活者偏誤。存活者偏誤是回測最常見也最致命的錯誤：
用「今天還在市的股票」回測十年，等於事先知道哪些公司不會倒。這種回測的績效
永遠漂亮，而且錯得毫無徵兆。

本模組的做法是把宇宙記成**區間**而非快照：每個標的的每一段成員資格都有
起訖日與狀態，因此任何一天的宇宙都能重建，包括那天還在、後來下市的標的。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from trading_intel.core.clock import ensure_utc
from trading_intel.core.enums import Market, TradingState
from trading_intel.core.ids import EntityId


class MembershipSpan(BaseModel):
    """一段成員資格：某標的在某段期間內的交易狀態。

    ``valid_to`` 為 None 代表至今仍然有效。狀態改變（例如被列為處置股）
    會把前一段收尾、開一段新的，而不是修改舊紀錄——舊紀錄是歷史事實。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: EntityId
    market: Market
    valid_from: date
    valid_to: date | None = None
    state: TradingState = TradingState.TRADABLE
    #: 這段紀錄何時進入系統。point-in-time 查詢據此過濾。
    ingest_time: AwareDatetime
    reason: str = ""

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.valid_to is not None and self.valid_to < self.valid_from:
            msg = "valid_to 不得早於 valid_from"
            raise ValueError(msg)
        return self

    def covers(self, day: date) -> bool:
        if day < self.valid_from:
            return False
        return self.valid_to is None or day <= self.valid_to


class UniverseSnapshot(BaseModel):
    """某一天的宇宙全貌，含不可交易的標的與其原因。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_date: date
    asof: AwareDatetime
    #: 標的 → 當日狀態。含 NO_TRADE / HALTED / RESTRICTED，不是只有可買的。
    states: dict[EntityId, TradingState]

    @property
    def tradable(self) -> frozenset[EntityId]:
        """當日真正可以下單的標的。回測選股只能從這裡選。"""
        return frozenset(
            entity_id for entity_id, state in self.states.items() if state is TradingState.TRADABLE
        )

    def __len__(self) -> int:
        return len(self.states)


class UniverseStore:
    """成員資格的 point-in-time 儲存體。"""

    def __init__(self, spans: Iterable[MembershipSpan] = ()) -> None:
        self._spans: list[MembershipSpan] = []
        for span in spans:
            self.add(span)

    def add(self, span: MembershipSpan) -> None:
        self._spans.append(span)
        self._spans.sort(key=lambda item: (item.entity_id, item.valid_from, item.ingest_time))

    def __len__(self) -> int:
        return len(self._spans)

    def snapshot(self, trade_date: date, *, asof: datetime) -> UniverseSnapshot:
        """重建 ``trade_date`` 當日、以 ``asof`` 為認知界線的宇宙。

        SPEC 第 11 節 Phase 1 驗收條件：「給定任一歷史日期，
        可完整重建當日的可交易宇宙」。

        同一標的同一天若有多段紀錄（例如先標為可交易、盤中被打入處置），
        取 ``asof`` 之前最後送達的那一段。
        """
        boundary = ensure_utc(asof)
        latest: dict[EntityId, MembershipSpan] = {}
        for span in sorted(self._spans, key=lambda item: item.ingest_time):
            if span.ingest_time > boundary:
                continue
            if not span.covers(trade_date):
                continue
            latest[span.entity_id] = span
        return UniverseSnapshot(
            trade_date=trade_date,
            asof=boundary,
            states={entity_id: span.state for entity_id, span in latest.items()},
        )

    def tradable_on(self, trade_date: date, *, asof: datetime) -> frozenset[EntityId]:
        """當日可交易標的的捷徑。"""
        return self.snapshot(trade_date, asof=asof).tradable

    def mark(
        self,
        entity_id: EntityId,
        *,
        market: Market,
        state: TradingState,
        valid_from: date,
        ingest_time: datetime,
        valid_to: date | None = None,
        reason: str = "",
    ) -> MembershipSpan:
        """新增一段狀態紀錄。舊紀錄保留不動，因為那是歷史事實。"""
        span = MembershipSpan(
            entity_id=entity_id,
            market=market,
            valid_from=valid_from,
            valid_to=valid_to,
            state=state,
            ingest_time=ensure_utc(ingest_time),
            reason=reason,
        )
        self.add(span)
        return span
