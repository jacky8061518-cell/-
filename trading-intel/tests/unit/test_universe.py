"""可交易宇宙：任一歷史日期都能完整重建，含當時還在、後來下市的標的。"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market, TradingState
from trading_intel.core.ids import make_entity_id
from trading_intel.normalize.universe import MembershipSpan, UniverseStore

TSMC = make_entity_id(Market.TW, "2330")
DELISTED = make_entity_id(Market.TW, "2489")
PUNISHED = make_entity_id(Market.TW, "1234")

T0 = datetime(2014, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def store() -> UniverseStore:
    return UniverseStore(
        [
            MembershipSpan(
                entity_id=TSMC,
                market=Market.TW,
                valid_from=date(1994, 9, 5),
                state=TradingState.TRADABLE,
                ingest_time=T0,
            ),
            # 2015 年下市的公司。它在 2014 年的宇宙裡必須存在。
            MembershipSpan(
                entity_id=DELISTED,
                market=Market.TW,
                valid_from=date(2000, 1, 1),
                valid_to=date(2015, 6, 30),
                state=TradingState.TRADABLE,
                ingest_time=T0,
                reason="下市",
            ),
        ]
    )


def test_delisted_company_exists_in_a_past_universe() -> None:
    """存活者偏誤的直接檢驗：2014 年的宇宙必須含 2015 年才下市的公司。"""
    snapshot = store().snapshot(date(2014, 6, 30), asof=NOW)
    assert DELISTED in snapshot.tradable
    assert TSMC in snapshot.tradable
    assert len(snapshot) == 2


def test_delisted_company_is_absent_after_its_delisting() -> None:
    snapshot = store().snapshot(date(2016, 6, 30), asof=NOW)
    assert DELISTED not in snapshot.states
    assert TSMC in snapshot.tradable


def test_last_day_of_listing_is_inclusive() -> None:
    assert DELISTED in store().snapshot(date(2015, 6, 30), asof=NOW).tradable
    assert DELISTED not in store().snapshot(date(2015, 7, 1), asof=NOW).states


def test_before_listing_the_entity_is_absent() -> None:
    assert TSMC not in store().snapshot(date(1994, 9, 4), asof=NOW).states
    assert TSMC in store().snapshot(date(1994, 9, 5), asof=NOW).tradable


def test_restricted_entity_is_in_the_universe_but_not_tradable() -> None:
    """處置股仍在宇宙內，但不可交易。兩者必須分開表達。"""
    universe = store()
    universe.mark(
        PUNISHED,
        market=Market.TW,
        state=TradingState.RESTRICTED,
        valid_from=date(2014, 5, 1),
        valid_to=date(2014, 5, 12),
        ingest_time=T0,
        reason="處置股",
    )
    snapshot = universe.snapshot(date(2014, 5, 5), asof=NOW)
    assert PUNISHED in snapshot.states
    assert snapshot.states[PUNISHED] is TradingState.RESTRICTED
    assert PUNISHED not in snapshot.tradable


def test_halted_entity_is_excluded_from_tradable() -> None:
    universe = store()
    universe.mark(
        TSMC,
        market=Market.TW,
        state=TradingState.HALTED,
        valid_from=date(2014, 7, 1),
        valid_to=date(2014, 7, 2),
        ingest_time=T0,
    )
    snapshot = universe.snapshot(date(2014, 7, 1), asof=NOW)
    assert snapshot.states[TSMC] is TradingState.HALTED
    assert TSMC not in snapshot.tradable


def test_latest_span_by_ingest_time_wins() -> None:
    """同日多筆紀錄取最後送達的那一筆。"""
    universe = store()
    universe.mark(
        TSMC,
        market=Market.TW,
        state=TradingState.NO_TRADE,
        valid_from=date(2014, 7, 1),
        valid_to=date(2014, 7, 1),
        ingest_time=datetime(2014, 7, 1, 12, tzinfo=UTC),
        reason="資料品質閘門",
    )
    snapshot = universe.snapshot(date(2014, 7, 1), asof=NOW)
    assert snapshot.states[TSMC] is TradingState.NO_TRADE


def test_asof_hides_spans_learned_later() -> None:
    """CLAUDE.md 第 2 條：以 asof 過濾 ingest_time。"""
    universe = store()
    universe.mark(
        PUNISHED,
        market=Market.TW,
        state=TradingState.TRADABLE,
        valid_from=date(2014, 1, 1),
        ingest_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    early = universe.snapshot(date(2014, 6, 30), asof=datetime(2015, 1, 1, tzinfo=UTC))
    late = universe.snapshot(date(2014, 6, 30), asof=NOW)
    assert PUNISHED not in early.states
    assert PUNISHED in late.states


def test_a_late_correction_does_not_rewrite_the_past_view() -> None:
    """在舊 asof 下，後來才知道的處置紀錄不存在。"""
    universe = store()
    universe.mark(
        TSMC,
        market=Market.TW,
        state=TradingState.RESTRICTED,
        valid_from=date(2014, 6, 1),
        valid_to=date(2014, 6, 30),
        ingest_time=datetime(2014, 8, 1, tzinfo=UTC),
    )
    at_the_time = universe.snapshot(date(2014, 6, 15), asof=datetime(2014, 6, 16, tzinfo=UTC))
    in_hindsight = universe.snapshot(date(2014, 6, 15), asof=NOW)
    assert at_the_time.states[TSMC] is TradingState.TRADABLE
    assert in_hindsight.states[TSMC] is TradingState.RESTRICTED


def test_tradable_on_is_a_shortcut() -> None:
    assert store().tradable_on(date(2014, 6, 30), asof=NOW) == frozenset({TSMC, DELISTED})


def test_reversed_span_is_rejected() -> None:
    with pytest.raises(ValidationError, match="valid_to 不得早於 valid_from"):
        MembershipSpan(
            entity_id=TSMC,
            market=Market.TW,
            valid_from=date(2014, 6, 30),
            valid_to=date(2014, 1, 1),
            ingest_time=T0,
        )


def test_single_day_span_is_allowed() -> None:
    span = MembershipSpan(
        entity_id=TSMC,
        market=Market.TW,
        valid_from=date(2014, 6, 30),
        valid_to=date(2014, 6, 30),
        ingest_time=T0,
    )
    assert span.covers(date(2014, 6, 30))
    assert not span.covers(date(2014, 7, 1))


def test_empty_store_yields_an_empty_snapshot() -> None:
    snapshot = UniverseStore().snapshot(date(2014, 6, 30), asof=NOW)
    assert len(snapshot) == 0
    assert snapshot.tradable == frozenset()


def test_snapshot_is_frozen() -> None:
    snapshot = store().snapshot(date(2014, 6, 30), asof=NOW)
    with pytest.raises(ValidationError):
        snapshot.trade_date = date(2015, 1, 1)  # type: ignore[misc]
