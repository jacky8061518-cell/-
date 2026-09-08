"""凍結的資料契約。

全部設 ``frozen=True`` 與 ``extra="forbid"``。
frozen 是為了可重現性：下游階段不得偷改上游的事實，重放才會重現原本的結果。
extra="forbid" 是因為欄位名稱打錯應該大聲報錯，而不是安靜地被丟掉。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Self, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from trading_intel.core.clock import MAX_CLOCK_SKEW, UTC
from trading_intel.core.enums import (
    DecisionLevel,
    Direction,
    DocType,
    Horizon,
    Market,
    QualityCheck,
    Severity,
    TradingState,
)
from trading_intel.core.errors import TemporalIntegrityError
from trading_intel.core.ids import EntityId, EvidenceId, SignalId

FROZEN_CONFIG = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class TemporalModel(BaseModel):
    """所有「在某時點發生、於稍後被觀察到」的資料的基底。

    ``event_time`` 是事實在世界上成立的時間；``ingest_time`` 是我們得知的時間。
    兩者都保留，回測才問得出「在時間 T 我們知道什麼」，而不是「現在什麼是真的」。
    這是 SPEC 1 的雙時間戳鐵律在型別層的落實。
    """

    model_config = FROZEN_CONFIG

    event_time: AwareDatetime
    ingest_time: AwareDatetime

    @model_validator(mode="after")
    def _validate_time_order(self) -> Self:
        for name, value in (("event_time", self.event_time), ("ingest_time", self.ingest_time)):
            if value.utcoffset() != UTC.utcoffset(None):
                raise TemporalIntegrityError(
                    "時間戳必須以 UTC 表示",
                    field=name,
                    value=value.isoformat(),
                    model=type(self).__name__,
                )
        if self.ingest_time < self.event_time - MAX_CLOCK_SKEW:
            raise TemporalIntegrityError(
                "ingest_time 早於 event_time 且超出允許的時鐘偏移",
                event_time=self.event_time.isoformat(),
                ingest_time=self.ingest_time.isoformat(),
                max_skew_seconds=MAX_CLOCK_SKEW.total_seconds(),
                model=type(self).__name__,
            )
        return self


class Instrument(BaseModel):
    """參考資料，不是事件：它有存續期間，沒有單一時間戳。"""

    model_config = FROZEN_CONFIG

    entity_id: EntityId
    market: Market
    local_symbol: str
    name: str
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    listed_date: date
    delisted_date: date | None = None

    @model_validator(mode="after")
    def _validate_lifetime(self) -> Self:
        if self.delisted_date is not None and self.delisted_date < self.listed_date:
            msg = "delisted_date 不得早於 listed_date"
            raise ValueError(msg)
        return self

    def is_active_on(self, d: date) -> bool:
        """存活者偏誤的防線：宇宙的建構必須經過這個判斷（SPEC 3.2）。"""
        if d < self.listed_date:
            return False
        return self.delisted_date is None or d <= self.delisted_date


class Bar(TemporalModel):
    """一根 OHLCV bar，以未調整原始價儲存，調整因子獨立存放。

    SPEC 3.2 明令禁止直接存調整後價格：新的企業行動會讓歷史調整價全部改變。
    """

    entity_id: EntityId
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)
    adj_factor: Decimal = Decimal("1")
    state: TradingState = TradingState.TRADABLE

    @model_validator(mode="after")
    def _validate_ohlc(self) -> Self:
        if self.low > min(self.open, self.close) or max(self.open, self.close) > self.high:
            msg = (
                f"OHLC 順序違規：low={self.low} open={self.open} "
                f"close={self.close} high={self.high}"
            )
            raise ValueError(msg)
        return self


class Document(TemporalModel):
    """一份文本證據，含來源出處與提示注入偵測旗標（SPEC 4.3）。"""

    doc_id: EvidenceId
    doc_type: DocType
    source: str
    source_credibility: float = Field(ge=0, le=1)
    title: str
    body: str
    entity_ids: tuple[EntityId, ...] = ()
    suspicious: bool = False


class FeatureVector(TemporalModel):
    """特徵值與各自的產生者版本，讓漂移可以被追溯。"""

    entity_id: EntityId
    asof: AwareDatetime
    values: Mapping[str, float]
    feature_versions: Mapping[str, str]

    @model_validator(mode="after")
    def _validate_versions(self) -> Self:
        missing = sorted(set(self.values) - set(self.feature_versions))
        if missing:
            msg = f"下列特徵缺少 feature_versions：{', '.join(missing)}"
            raise ValueError(msg)
        return self


class Signal(TemporalModel):
    """帶明確保存期限與明確失效條件的方向性觀點。"""

    signal_id: SignalId
    entity_id: EntityId
    direction: Direction
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    half_life_days: float = Field(gt=0)
    # 說不出什麼情況代表自己錯了的判斷，不算訊號。用 min_length 把這件事
    # 變成型別系統的責任，而不是紀律問題。
    invalidation_condition: str = Field(min_length=10)
    horizon: Horizon
    evidence_ids: tuple[EvidenceId, ...]
    model_version: str
    decision_level: DecisionLevel


class OrderIntent(TemporalModel):
    """我們想要持有的部位，在風控表態之前。"""

    entity_id: EntityId
    direction: Direction
    target_weight: float = Field(ge=-1, le=1)
    limit_price: Decimal | None = None
    source_signal_ids: tuple[SignalId, ...]
    decision_level: DecisionLevel


class RiskVerdict(TemporalModel):
    """風控對一筆意圖的裁決。否決時必須指名違反了什麼。"""

    intent_hash: str
    approved: bool
    breached_limits: tuple[str, ...] = ()
    adjusted_weight: float | None = None
    note: str = ""

    @model_validator(mode="after")
    def _validate_verdict(self) -> Self:
        if not self.approved and not self.breached_limits:
            msg = "遭否決的意圖必須至少列出一項違反的限額"
            raise ValueError(msg)
        return self


class DataQualityAlert(TemporalModel):
    """一次失敗的品質檢查，以及它強制設定的交易狀態（SPEC 3.3）。"""

    entity_id: EntityId | None
    check: QualityCheck
    severity: Severity
    detail: str
    resulting_state: TradingState


#: SPEC 第 11 節 Phase 0 以 MarketEvent 與 NewsEvent 稱呼下列兩個型別。
#: 這裡保留較精確的實作名稱，並提供 SPEC 用語作為別名，兩者指向同一個類別。
MarketEvent = Bar
NewsEvent = Document

T = TypeVar("T", bound=BaseModel)


class AgentOutput[T: BaseModel](BaseModel):
    """每個 LLM agent 都必須滿足的契約（SPEC 4.1）。Phase 3 才使用，契約現在就定死。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    payload: T | None
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[EvidenceId, ...]
    reasoning_digest: str = Field(max_length=200)
    abstain: bool = False
    model_name: str
    prompt_hash: str

    @model_validator(mode="after")
    def _validate_abstain(self) -> Self:
        if self.abstain and self.payload is not None:
            msg = "棄權的 agent 不得回傳 payload"
            raise ValueError(msg)
        if not self.abstain and self.payload is None:
            msg = "未棄權的 agent 必須回傳 payload"
            raise ValueError(msg)
        return self
