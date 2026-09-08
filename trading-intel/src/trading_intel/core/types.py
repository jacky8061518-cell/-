"""Frozen data contracts.

Everything here is ``frozen=True`` and ``extra="forbid"``. Frozen because a
downstream stage must not be able to quietly mutate an upstream fact — that is
what makes a replay reproduce the original result. ``extra="forbid"`` because a
typo in a field name should be a loud error, not a silently dropped value.
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
    """Base for anything that happened at a point in time and was later observed.

    ``event_time`` is when the fact became true in the world; ``ingest_time`` is
    when we learned about it. Keeping both is what lets a backtest ask "what did
    we know at time T" instead of "what is true now".
    """

    model_config = FROZEN_CONFIG

    event_time: AwareDatetime
    ingest_time: AwareDatetime

    @model_validator(mode="after")
    def _validate_time_order(self) -> Self:
        for name, value in (("event_time", self.event_time), ("ingest_time", self.ingest_time)):
            if value.utcoffset() != UTC.utcoffset(None):
                raise TemporalIntegrityError(
                    "timestamps must be expressed in UTC",
                    field=name,
                    value=value.isoformat(),
                    model=type(self).__name__,
                )
        if self.ingest_time < self.event_time - MAX_CLOCK_SKEW:
            raise TemporalIntegrityError(
                "ingest_time precedes event_time by more than the allowed clock skew",
                event_time=self.event_time.isoformat(),
                ingest_time=self.ingest_time.isoformat(),
                max_skew_seconds=MAX_CLOCK_SKEW.total_seconds(),
                model=type(self).__name__,
            )
        return self


class Instrument(BaseModel):
    """Reference data, not an event: it has a lifetime, not a timestamp."""

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
            msg = "delisted_date cannot precede listed_date"
            raise ValueError(msg)
        return self

    def is_active_on(self, d: date) -> bool:
        """Survivorship-bias guard: universes must be built through this."""
        if d < self.listed_date:
            return False
        return self.delisted_date is None or d <= self.delisted_date


class Bar(TemporalModel):
    """One OHLCV bar, stored unadjusted with an independent adjustment factor."""

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
                f"OHLC ordering violated: low={self.low} open={self.open} "
                f"close={self.close} high={self.high}"
            )
            raise ValueError(msg)
        return self


class Document(TemporalModel):
    """A piece of text evidence, with provenance and a prompt-injection flag."""

    doc_id: EvidenceId
    doc_type: DocType
    source: str
    source_credibility: float = Field(ge=0, le=1)
    title: str
    body: str
    entity_ids: tuple[EntityId, ...] = ()
    suspicious: bool = False


class FeatureVector(TemporalModel):
    """Features plus the version of each producer, so a drift can be traced."""

    entity_id: EntityId
    asof: AwareDatetime
    values: Mapping[str, float]
    feature_versions: Mapping[str, str]

    @model_validator(mode="after")
    def _validate_versions(self) -> Self:
        missing = sorted(set(self.values) - set(self.feature_versions))
        if missing:
            msg = f"missing feature_versions for: {', '.join(missing)}"
            raise ValueError(msg)
        return self


class Signal(TemporalModel):
    """A directional view with an explicit expiry and an explicit kill condition."""

    signal_id: SignalId
    entity_id: EntityId
    direction: Direction
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    half_life_days: float = Field(gt=0)
    # A signal you cannot say would be wrong is not a signal. min_length makes
    # that a type-system responsibility rather than a matter of discipline.
    invalidation_condition: str = Field(min_length=10)
    horizon: Horizon
    evidence_ids: tuple[EvidenceId, ...]
    model_version: str
    decision_level: DecisionLevel


class OrderIntent(TemporalModel):
    """What we would like to hold, before risk has had its say."""

    entity_id: EntityId
    direction: Direction
    target_weight: float = Field(ge=-1, le=1)
    limit_price: Decimal | None = None
    source_signal_ids: tuple[SignalId, ...]
    decision_level: DecisionLevel


class RiskVerdict(TemporalModel):
    """Risk's answer to an intent. A rejection must name what it breached."""

    intent_hash: str
    approved: bool
    breached_limits: tuple[str, ...] = ()
    adjusted_weight: float | None = None
    note: str = ""

    @model_validator(mode="after")
    def _validate_verdict(self) -> Self:
        if not self.approved and not self.breached_limits:
            msg = "a rejected intent must list at least one breached limit"
            raise ValueError(msg)
        return self


class DataQualityAlert(TemporalModel):
    """A failed quality check and the trading state it forces."""

    entity_id: EntityId | None
    check: QualityCheck
    severity: Severity
    detail: str
    resulting_state: TradingState


T = TypeVar("T", bound=BaseModel)


class AgentOutput[T: BaseModel](BaseModel):
    """The contract every LLM agent must satisfy. Used from Phase 3; fixed now."""

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
            msg = "an abstaining agent must not return a payload"
            raise ValueError(msg)
        if not self.abstain and self.payload is None:
            msg = "a non-abstaining agent must return a payload"
            raise ValueError(msg)
        return self
