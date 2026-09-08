from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

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
from trading_intel.core.ids import EntityId, EvidenceId, SignalId, make_entity_id
from trading_intel.core.types import (
    AgentOutput,
    Bar,
    DataQualityAlert,
    Document,
    FeatureVector,
    Instrument,
    OrderIntent,
    RiskVerdict,
    Signal,
    TemporalModel,
)

EVENT_TIME = datetime(2020, 3, 19, 12, 0, tzinfo=UTC)
INGEST_TIME = EVENT_TIME + timedelta(minutes=1)
ENTITY = make_entity_id(Market.TW, "2330")


def bar(**overrides: object) -> Bar:
    kwargs: dict[str, object] = {
        "event_time": EVENT_TIME,
        "ingest_time": INGEST_TIME,
        "entity_id": ENTITY,
        "open": Decimal("100"),
        "high": Decimal("110"),
        "low": Decimal("90"),
        "close": Decimal("105"),
        "volume": 1000,
    }
    kwargs.update(overrides)
    return Bar(**kwargs)


def signal(**overrides: object) -> Signal:
    kwargs: dict[str, object] = {
        "event_time": EVENT_TIME,
        "ingest_time": INGEST_TIME,
        "signal_id": SignalId("abc123"),
        "entity_id": ENTITY,
        "direction": Direction.LONG,
        "score": 0.4,
        "confidence": 0.7,
        "half_life_days": 5.0,
        "invalidation_condition": "close below the 20-day moving average",
        "horizon": Horizon.DAYS,
        "evidence_ids": (EvidenceId("deadbeef"),),
        "model_version": "v1.0.0",
        "decision_level": DecisionLevel.CONFIRM,
    }
    kwargs.update(overrides)
    return Signal(**kwargs)


# --- TemporalModel ---------------------------------------------------------


class _Sample(TemporalModel):
    value: int


def test_temporal_model_accepts_ingest_after_event() -> None:
    model = _Sample(event_time=EVENT_TIME, ingest_time=INGEST_TIME, value=1)
    assert model.ingest_time > model.event_time


def test_temporal_model_allows_small_clock_skew() -> None:
    model = _Sample(
        event_time=EVENT_TIME,
        ingest_time=EVENT_TIME - MAX_CLOCK_SKEW,
        value=1,
    )
    assert model.value == 1


def test_temporal_model_rejects_ingest_far_before_event() -> None:
    # A dedicated exception, not a generic ValidationError: this is a data
    # pipeline bug, and callers need to be able to catch exactly it.
    with pytest.raises(TemporalIntegrityError) as excinfo:
        _Sample(
            event_time=EVENT_TIME,
            ingest_time=EVENT_TIME - MAX_CLOCK_SKEW - timedelta(seconds=1),
            value=1,
        )
    assert "clock skew" in str(excinfo.value)


def test_temporal_integrity_error_carries_structured_context() -> None:
    with pytest.raises(TemporalIntegrityError) as excinfo:
        _Sample(event_time=EVENT_TIME, ingest_time=EVENT_TIME - timedelta(days=1), value=1)
    context = excinfo.value.context
    assert context["model"] == "_Sample"
    assert context["event_time"] == EVENT_TIME.isoformat()


def test_temporal_model_rejects_non_utc_timestamps() -> None:
    taipei = EVENT_TIME.astimezone(ZoneInfo("Asia/Taipei"))
    with pytest.raises(TemporalIntegrityError, match="UTC"):
        _Sample(event_time=taipei, ingest_time=INGEST_TIME, value=1)


def test_temporal_model_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        _Sample(event_time=datetime(2020, 3, 19, 12, 0), ingest_time=INGEST_TIME, value=1)  # noqa: DTZ001


def test_temporal_model_is_frozen() -> None:
    model = _Sample(event_time=EVENT_TIME, ingest_time=INGEST_TIME, value=1)
    with pytest.raises(ValidationError):
        model.value = 2


def test_temporal_model_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _Sample(event_time=EVENT_TIME, ingest_time=INGEST_TIME, value=1, typo=3)


# --- Instrument ------------------------------------------------------------


def test_instrument_is_active_between_listing_and_delisting() -> None:
    inst = Instrument(
        entity_id=ENTITY,
        market=Market.TW,
        local_symbol="2330",
        name="TSMC",
        currency="TWD",
        listed_date=date(1994, 9, 5),
        delisted_date=None,
    )
    assert inst.is_active_on(date(2020, 3, 19))
    assert not inst.is_active_on(date(1994, 9, 4))


def test_instrument_respects_delisting() -> None:
    inst = Instrument(
        entity_id=ENTITY,
        market=Market.TW,
        local_symbol="0000",
        name="Gone",
        currency="TWD",
        listed_date=date(2000, 1, 1),
        delisted_date=date(2010, 6, 30),
    )
    assert inst.is_active_on(date(2010, 6, 30))
    assert not inst.is_active_on(date(2010, 7, 1))


def test_instrument_rejects_a_delisting_before_listing() -> None:
    with pytest.raises(ValidationError, match="delisted_date"):
        Instrument(
            entity_id=ENTITY,
            market=Market.TW,
            local_symbol="0000",
            name="Impossible",
            currency="TWD",
            listed_date=date(2010, 1, 1),
            delisted_date=date(2000, 1, 1),
        )


def test_instrument_rejects_a_bad_currency_code() -> None:
    with pytest.raises(ValidationError):
        Instrument(
            entity_id=ENTITY,
            market=Market.TW,
            local_symbol="2330",
            name="TSMC",
            currency="twd",
            listed_date=date(1994, 9, 5),
        )


# --- Bar -------------------------------------------------------------------


def test_valid_bar() -> None:
    b = bar()
    assert b.state is TradingState.TRADABLE
    assert b.adj_factor == Decimal("1")


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"low": Decimal("101")}, "low above open"),
        ({"low": Decimal("106")}, "low above close"),
        ({"high": Decimal("104")}, "high below close"),
        ({"open": Decimal("120")}, "open above high"),
    ],
)
def test_bar_rejects_broken_ohlc(overrides: dict[str, Decimal], why: str) -> None:
    with pytest.raises(ValidationError, match="OHLC ordering violated"):
        bar(**overrides)
    assert why


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        bar(volume=-1)


def test_bar_keeps_decimal_precision() -> None:
    b = bar(close=Decimal("105.123456789"))
    assert b.close == Decimal("105.123456789")
    assert isinstance(b.close, Decimal)


# --- Document / FeatureVector ---------------------------------------------


def test_document_defaults() -> None:
    doc = Document(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        doc_id=EvidenceId("abc"),
        doc_type=DocType.NEWS,
        source="cna",
        source_credibility=0.8,
        title="t",
        body="b",
    )
    assert doc.entity_ids == ()
    assert doc.suspicious is False


def test_document_rejects_out_of_range_credibility() -> None:
    with pytest.raises(ValidationError):
        Document(
            event_time=EVENT_TIME,
            ingest_time=INGEST_TIME,
            doc_id=EvidenceId("abc"),
            doc_type=DocType.NEWS,
            source="cna",
            source_credibility=1.5,
            title="t",
            body="b",
        )


def test_feature_vector_requires_a_version_per_feature() -> None:
    with pytest.raises(ValidationError, match="missing feature_versions"):
        FeatureVector(
            event_time=EVENT_TIME,
            ingest_time=INGEST_TIME,
            entity_id=ENTITY,
            asof=EVENT_TIME,
            values={"mom_20d": 0.3, "rsi_14": 55.0},
            feature_versions={"mom_20d": "v1"},
        )


def test_feature_vector_accepts_matching_versions() -> None:
    fv = FeatureVector(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        entity_id=ENTITY,
        asof=EVENT_TIME,
        values={"mom_20d": 0.3},
        feature_versions={"mom_20d": "v1"},
    )
    assert fv.values["mom_20d"] == 0.3


# --- Signal ----------------------------------------------------------------


def test_valid_signal() -> None:
    assert signal().direction is Direction.LONG


@pytest.mark.parametrize("condition", ["", "   ", "too short", "9 chars!!"])
def test_signal_rejects_a_missing_or_short_invalidation_condition(condition: str) -> None:
    with pytest.raises(ValidationError, match=r"invalidation_condition|at least 10"):
        signal(invalidation_condition=condition)


def test_signal_accepts_a_real_invalidation_condition() -> None:
    s = signal(invalidation_condition="foundry utilisation falls below 80%")
    assert len(s.invalidation_condition) >= 10


@pytest.mark.parametrize("score", [-1.5, 1.5])
def test_signal_rejects_out_of_range_score(score: float) -> None:
    with pytest.raises(ValidationError):
        signal(score=score)


def test_signal_rejects_a_non_positive_half_life() -> None:
    with pytest.raises(ValidationError):
        signal(half_life_days=0.0)


def test_signal_cannot_be_mutated() -> None:
    s = signal()
    with pytest.raises(ValidationError):
        s.score = 0.9


# --- OrderIntent / RiskVerdict --------------------------------------------


def test_order_intent_weight_bounds() -> None:
    intent = OrderIntent(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        entity_id=ENTITY,
        direction=Direction.LONG,
        target_weight=0.03,
        source_signal_ids=(SignalId("abc123"),),
        decision_level=DecisionLevel.CONFIRM,
    )
    assert intent.limit_price is None
    with pytest.raises(ValidationError):
        OrderIntent(
            event_time=EVENT_TIME,
            ingest_time=INGEST_TIME,
            entity_id=ENTITY,
            direction=Direction.LONG,
            target_weight=1.5,
            source_signal_ids=(),
            decision_level=DecisionLevel.CONFIRM,
        )


def test_risk_verdict_rejection_must_name_a_breached_limit() -> None:
    with pytest.raises(ValidationError, match="breached limit"):
        RiskVerdict(
            event_time=EVENT_TIME,
            ingest_time=INGEST_TIME,
            intent_hash="h",
            approved=False,
        )


def test_risk_verdict_rejection_with_a_reason_is_accepted() -> None:
    verdict = RiskVerdict(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        intent_hash="h",
        approved=False,
        breached_limits=("max_position_weight",),
    )
    assert verdict.breached_limits == ("max_position_weight",)


def test_risk_verdict_approval_needs_no_breaches() -> None:
    verdict = RiskVerdict(
        event_time=EVENT_TIME, ingest_time=INGEST_TIME, intent_hash="h", approved=True
    )
    assert verdict.breached_limits == ()


# --- DataQualityAlert ------------------------------------------------------


def test_data_quality_alert_allows_a_market_wide_alert() -> None:
    alert = DataQualityAlert(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        entity_id=None,
        check=QualityCheck.FRESHNESS,
        severity=Severity.P0,
        detail="feed stalled for 30 minutes",
        resulting_state=TradingState.NO_TRADE,
    )
    assert alert.entity_id is None
    assert alert.resulting_state is TradingState.NO_TRADE


# --- AgentOutput -----------------------------------------------------------


class _Payload(BaseModel):
    verdict: str


def agent_output(**overrides: object) -> AgentOutput[_Payload]:
    kwargs: dict[str, object] = {
        "payload": _Payload(verdict="bullish"),
        "confidence": 0.6,
        "evidence_ids": (EvidenceId("deadbeef"),),
        "reasoning_digest": "margin expansion confirmed by two independent sources",
        "abstain": False,
        "model_name": "test-model",
        "prompt_hash": "ph",
    }
    kwargs.update(overrides)
    return AgentOutput[_Payload](**kwargs)


def test_agent_output_with_payload_is_accepted() -> None:
    assert agent_output().payload is not None


def test_abstaining_agent_must_not_return_a_payload() -> None:
    with pytest.raises(ValidationError, match="must not return a payload"):
        agent_output(abstain=True)


def test_abstaining_agent_with_no_payload_is_accepted() -> None:
    out = agent_output(abstain=True, payload=None)
    assert out.payload is None


def test_non_abstaining_agent_must_return_a_payload() -> None:
    with pytest.raises(ValidationError, match="must return a payload"):
        agent_output(payload=None)


def test_reasoning_digest_is_length_capped() -> None:
    with pytest.raises(ValidationError):
        agent_output(reasoning_digest="x" * 201)


def test_agent_output_is_frozen() -> None:
    out = agent_output()
    with pytest.raises(ValidationError):
        out.confidence = 0.1  # type: ignore[misc]


def test_entity_id_newtype_is_a_plain_string_at_runtime() -> None:
    assert isinstance(EntityId("TW:2330"), str)
