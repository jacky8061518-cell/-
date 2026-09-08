"""事件信封與內容雜湊。

idempotency_key 由內容推導，重放同一個事實時下游可據此去重。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from trading_intel.core.clock import UTC, SimulatedClock, use_clock
from trading_intel.core.enums import Market
from trading_intel.core.events import EventEnvelope, payload_hash, wrap
from trading_intel.core.ids import CorrelationId, make_entity_id
from trading_intel.core.types import Bar

EVENT_TIME = datetime(2020, 3, 19, 12, 0, tzinfo=UTC)
INGEST_TIME = EVENT_TIME + timedelta(minutes=1)
ENTITY = make_entity_id(Market.TW, "2330")


def make_bar(close: str = "105") -> Bar:
    return Bar(
        event_time=EVENT_TIME,
        ingest_time=INGEST_TIME,
        entity_id=ENTITY,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal(close),
        volume=1000,
    )


class _A(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    alpha: int
    beta: str


class _B(BaseModel):
    """欄位與 ``_A`` 相同，但宣告順序相反。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    beta: str
    alpha: int


def test_payload_hash_ignores_field_declaration_order() -> None:
    assert payload_hash(_A(alpha=1, beta="x")) == payload_hash(_B(beta="x", alpha=1))


def test_payload_hash_ignores_keyword_argument_order() -> None:
    assert payload_hash(_A(alpha=1, beta="x")) == payload_hash(_A(beta="x", alpha=1))


def test_payload_hash_changes_when_content_changes() -> None:
    assert payload_hash(_A(alpha=1, beta="x")) != payload_hash(_A(alpha=2, beta="x"))


def test_payload_hash_is_stable_across_calls() -> None:
    bar = make_bar()
    assert payload_hash(bar) == payload_hash(bar)


def test_wrap_populates_the_envelope() -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        envelope = wrap(make_bar(), event_type="market.bar")
    assert envelope.event_type == "market.bar"
    assert envelope.schema_version == 1
    assert envelope.produced_at == EVENT_TIME
    assert envelope.idempotency_key == payload_hash(make_bar())
    assert envelope.payload == make_bar()


def test_wrap_uses_the_active_clock() -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        assert wrap(make_bar(), event_type="market.bar").produced_at.year == 2020


def test_identical_payloads_share_an_idempotency_key() -> None:
    first = wrap(make_bar(), event_type="market.bar")
    second = wrap(make_bar(), event_type="market.bar")
    assert first.idempotency_key == second.idempotency_key
    assert first.event_id == second.event_id
    # ……但追蹤用的 id 不同，因為那是每次投遞各自產生的。
    assert first.correlation_id != second.correlation_id


def test_different_payloads_get_different_keys() -> None:
    assert (
        wrap(make_bar("105"), event_type="market.bar").idempotency_key
        != wrap(make_bar("106"), event_type="market.bar").idempotency_key
    )


def test_event_id_separates_event_types() -> None:
    assert (
        wrap(make_bar(), event_type="market.bar").event_id
        != wrap(make_bar(), event_type="market.bar.corrected").event_id
    )


def test_event_id_separates_schema_versions() -> None:
    assert (
        wrap(make_bar(), event_type="market.bar", schema_version=1).event_id
        != wrap(make_bar(), event_type="market.bar", schema_version=2).event_id
    )


def test_supplied_correlation_id_is_kept() -> None:
    cid = CorrelationId("abc123")
    assert wrap(make_bar(), event_type="market.bar", correlation_id=cid).correlation_id == cid


def test_envelope_is_frozen() -> None:
    envelope = wrap(make_bar(), event_type="market.bar")
    with pytest.raises(ValidationError):
        envelope.event_type = "other"  # type: ignore[misc]


def test_envelope_round_trips() -> None:
    envelope = wrap(make_bar(), event_type="market.bar")
    restored = EventEnvelope[Bar].model_validate_json(envelope.model_dump_json())
    assert restored == envelope


def test_envelope_rejects_a_zero_schema_version() -> None:
    with pytest.raises(ValidationError):
        wrap(make_bar(), event_type="market.bar", schema_version=0)


def test_envelope_rejects_a_blank_event_type() -> None:
    with pytest.raises(ValidationError):
        wrap(make_bar(), event_type="")
