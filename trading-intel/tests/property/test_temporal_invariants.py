"""必須對每一個值都成立的不變條件，不只是我們想得到的那幾個。"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from trading_intel.core.clock import MAX_CLOCK_SKEW, UTC
from trading_intel.core.enums import DocType, Market, TradingState
from trading_intel.core.events import payload_hash
from trading_intel.core.ids import EvidenceId, make_entity_id, make_evidence_id
from trading_intel.core.types import Bar, Document

ENTITY = make_entity_id(Market.TW, "2330")

# hypothesis 接受 naive 的上下界，時區由它自己附加。
aware_datetimes = st.datetimes(
    min_value=datetime(1990, 1, 1),  # noqa: DTZ001
    max_value=datetime(2100, 1, 1),  # noqa: DTZ001
    timezones=st.just(UTC),
)

#: 價格在建構上就是精確的：兩位小數，全程不碰 float。
prices = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("1000000"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def bars(draw: st.DrawFn) -> Bar:
    event_time = draw(aware_datetimes)
    lag = draw(st.timedeltas(min_value=-MAX_CLOCK_SKEW, max_value=timedelta(days=3)))
    a = draw(prices)
    b = draw(prices)
    open_, close = (a, b)
    low = draw(prices.filter(lambda p: p <= min(open_, close)))
    high = draw(prices.filter(lambda p: p >= max(open_, close)))
    return Bar(
        event_time=event_time,
        ingest_time=event_time + lag,
        entity_id=ENTITY,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=draw(st.integers(min_value=0, max_value=10**12)),
        adj_factor=draw(prices),
        state=draw(st.sampled_from(TradingState)),
    )


@st.composite
def documents(draw: st.DrawFn) -> Document:
    event_time = draw(aware_datetimes)
    lag = draw(st.timedeltas(min_value=-MAX_CLOCK_SKEW, max_value=timedelta(days=3)))
    body = draw(st.text(max_size=200))
    return Document(
        event_time=event_time,
        ingest_time=event_time + lag,
        doc_id=make_evidence_id("test", body.encode("utf-8")),
        doc_type=draw(st.sampled_from(DocType)),
        source=draw(st.sampled_from(["twse", "mops", "cna", "reuters"])),
        source_credibility=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        title=draw(st.text(max_size=50)),
        body=body,
        entity_ids=(ENTITY,),
        suspicious=draw(st.booleans()),
    )


@settings(max_examples=200)
@given(bar=bars())
def test_bar_json_round_trip_is_exact(bar: Bar) -> None:
    assert Bar.model_validate_json(bar.model_dump_json()) == bar


@settings(max_examples=200)
@given(bar=bars())
def test_bar_round_trip_preserves_decimal_precision(bar: Bar) -> None:
    """價格採用 ``Decimal`` 的理由：走一趟 float 會掉分位。"""
    restored = Bar.model_validate_json(bar.model_dump_json())
    for field in ("open", "high", "low", "close", "adj_factor"):
        original = getattr(bar, field)
        recovered = getattr(restored, field)
        assert isinstance(recovered, Decimal)
        assert recovered == original
        assert recovered.as_tuple() == original.as_tuple()


@settings(max_examples=100)
@given(doc=documents())
def test_document_json_round_trip_is_exact(doc: Document) -> None:
    assert Document.model_validate_json(doc.model_dump_json()) == doc


@settings(max_examples=200)
@given(bar=bars())
def test_temporal_ordering_holds_for_every_constructed_model(bar: Bar) -> None:
    assert bar.ingest_time >= bar.event_time - MAX_CLOCK_SKEW
    assert bar.event_time.utcoffset() == timedelta(0)
    assert bar.ingest_time.utcoffset() == timedelta(0)


@settings(max_examples=200)
@given(bar=bars())
def test_ohlc_invariant_holds_for_every_constructed_bar(bar: Bar) -> None:
    assert bar.low <= min(bar.open, bar.close)
    assert max(bar.open, bar.close) <= bar.high


@settings(max_examples=200)
@given(bar=bars())
def test_equal_payloads_hash_equal(bar: Bar) -> None:
    twin = Bar.model_validate_json(bar.model_dump_json())
    assert twin == bar
    assert payload_hash(twin) == payload_hash(bar)


@settings(max_examples=200)
@given(bar=bars(), delta=st.integers(min_value=1, max_value=10**6))
def test_changing_any_field_changes_the_hash(bar: Bar, delta: int) -> None:
    mutated = bar.model_copy(update={"volume": bar.volume + delta})
    assert payload_hash(mutated) != payload_hash(bar)


@settings(max_examples=100)
@given(
    source=st.text(min_size=1, max_size=20),
    payload=st.binary(max_size=200),
)
def test_evidence_id_is_a_pure_function(source: str, payload: bytes) -> None:
    first = make_evidence_id(source, payload)
    assert first == make_evidence_id(source, payload)
    assert isinstance(first, str)
    assert len(first) == 16


@settings(max_examples=100)
@given(doc=documents())
def test_document_hash_is_order_independent(doc: Document) -> None:
    # 以打亂順序的 kwargs 重建，雜湊值不得改變。
    fields = dict(reversed(list(doc.model_dump().items())))
    rebuilt = Document(**fields)
    assert payload_hash(rebuilt) == payload_hash(doc)


def test_evidence_id_newtype_round_trips() -> None:
    assert EvidenceId("abc") == "abc"
