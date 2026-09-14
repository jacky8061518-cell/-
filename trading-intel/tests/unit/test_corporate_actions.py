"""企業行動引擎：未調整原始價 + 獨立因子表，查詢結果隨 asof 改變。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.core.errors import LookaheadError, NaiveDatetimeError
from trading_intel.core.ids import make_entity_id
from trading_intel.normalize.corporate_actions import (
    ActionSource,
    ActionType,
    CorporateAction,
    CorporateActionStore,
    RawPrice,
    assert_no_lookahead,
    get_adjusted_prices,
)

TSMC = make_entity_id(Market.TW, "2330")
UMC = make_entity_id(Market.TW, "2303")

#: 價格在 3/1 就送達；除息公告 3/20 才送達。這個時間差是本模組的全部重點。
PRICE_INGEST = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
DIVIDEND_INGEST = datetime(2024, 3, 20, 0, 0, tzinfo=UTC)
EX_DATE = date(2024, 3, 15)


def price(day: int, close: str, *, entity=TSMC, ingest=PRICE_INGEST) -> RawPrice:  # type: ignore[no-untyped-def]
    return RawPrice(
        entity_id=entity,
        trade_date=date(2024, 3, day),
        close=Decimal(close),
        volume=1000,
        ingest_time=ingest,
    )


PRICES = [price(11, "100"), price(12, "102"), price(13, "101"), price(14, "100")]

DIVIDEND = CorporateAction(
    entity_id=TSMC,
    action_type=ActionType.CASH_DIVIDEND,
    ex_date=EX_DATE,
    factor=Decimal("0.95"),
    source=ActionSource.OFFICIAL,
    confidence=1.0,
    ingest_time=DIVIDEND_INGEST,
    detail="現金股利 5 元",
)


# --- 驗收條件：asof 在企業行動前後，同一段歷史結果不同 ----------------------


def test_asof_before_the_announcement_sees_unadjusted_prices() -> None:
    store = CorporateActionStore([DIVIDEND])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 14, tzinfo=UTC),
    )
    assert len(result) == 4
    assert all(item.applied_actions == 0 for item in result)
    assert result[0].adjusted_close == Decimal("100")
    assert result[0].cumulative_factor == Decimal("1")


def test_asof_after_the_announcement_sees_adjusted_prices() -> None:
    store = CorporateActionStore([DIVIDEND])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    assert all(item.applied_actions == 1 for item in result)
    assert result[0].adjusted_close == Decimal("95.00")
    assert result[0].cumulative_factor == Decimal("0.95")


def test_the_two_queries_differ() -> None:
    """SPEC 第 11 節 Phase 1 驗收：asof 前後查同一段歷史，結果應不同。"""
    store = CorporateActionStore([DIVIDEND])
    before = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 14, tzinfo=UTC),
    )
    after = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    assert [item.adjusted_close for item in before] != [item.adjusted_close for item in after]
    # 但原始價完全相同——事實沒變，變的只是我們對它的詮釋。
    assert [item.raw_close for item in before] == [item.raw_close for item in after]


def test_raw_prices_never_change() -> None:
    store = CorporateActionStore([DIVIDEND])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    assert [item.raw_close for item in result] == [
        Decimal("100"),
        Decimal("102"),
        Decimal("101"),
        Decimal("100"),
    ]


# --- 因子只作用於除權息日之前的價格 ----------------------------------------


def test_prices_after_the_ex_date_are_not_adjusted() -> None:
    store = CorporateActionStore([DIVIDEND])
    later = [price(18, "96"), price(19, "97")]
    result = get_adjusted_prices(
        [*PRICES, *later],
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 19),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    by_date = {item.trade_date: item for item in result}
    assert by_date[date(2024, 3, 14)].applied_actions == 1
    assert by_date[date(2024, 3, 18)].applied_actions == 0
    assert by_date[date(2024, 3, 18)].adjusted_close == Decimal("96")


def test_multiple_actions_compound() -> None:
    split = CorporateAction(
        entity_id=TSMC,
        action_type=ActionType.SPLIT,
        ex_date=date(2024, 3, 13),
        factor=Decimal("0.5"),
        source=ActionSource.OFFICIAL,
        confidence=1.0,
        ingest_time=DIVIDEND_INGEST,
    )
    store = CorporateActionStore([DIVIDEND, split])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    by_date = {item.trade_date: item for item in result}
    # 3/12 之前同時受分割與除息影響：0.5 * 0.95。
    assert by_date[date(2024, 3, 12)].cumulative_factor == Decimal("0.475")
    assert by_date[date(2024, 3, 12)].applied_actions == 2
    # 3/13（分割除權日當天）只受除息影響。
    assert by_date[date(2024, 3, 13)].cumulative_factor == Decimal("0.95")


def test_actions_of_other_entities_are_ignored() -> None:
    other = CorporateAction(
        entity_id=UMC,
        action_type=ActionType.CASH_DIVIDEND,
        ex_date=EX_DATE,
        factor=Decimal("0.5"),
        source=ActionSource.OFFICIAL,
        confidence=1.0,
        ingest_time=DIVIDEND_INGEST,
    )
    store = CorporateActionStore([DIVIDEND, other])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    assert result[0].cumulative_factor == Decimal("0.95")


# --- asof 過濾價格本身 -----------------------------------------------------


def test_prices_ingested_after_asof_are_invisible() -> None:
    late = price(15, "99", ingest=datetime(2024, 4, 1, tzinfo=UTC))
    result = get_adjusted_prices(
        [*PRICES, late],
        store := CorporateActionStore(),
        entity_id=TSMC,
        start=date(2024, 3, 11),
        end=date(2024, 3, 20),
        asof=datetime(2024, 3, 20, tzinfo=UTC),
    )
    assert len(store) == 0
    assert date(2024, 3, 15) not in {item.trade_date for item in result}


def test_a_correction_supersedes_the_original() -> None:
    """同一天兩筆價格，取 asof 之前最後送達的那一筆。"""
    corrected = price(14, "104", ingest=datetime(2024, 3, 16, tzinfo=UTC))
    result = get_adjusted_prices(
        [*PRICES, corrected],
        CorporateActionStore(),
        entity_id=TSMC,
        start=date(2024, 3, 14),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 18, tzinfo=UTC),
    )
    assert result[0].raw_close == Decimal("104")


def test_before_the_correction_the_original_is_returned() -> None:
    corrected = price(14, "104", ingest=datetime(2024, 3, 16, tzinfo=UTC))
    result = get_adjusted_prices(
        [*PRICES, corrected],
        CorporateActionStore(),
        entity_id=TSMC,
        start=date(2024, 3, 14),
        end=date(2024, 3, 14),
        asof=datetime(2024, 3, 15, tzinfo=UTC),
    )
    assert result[0].raw_close == Decimal("100")


# --- 來源與信心 ------------------------------------------------------------


def test_official_source_must_have_full_confidence() -> None:
    with pytest.raises(ValidationError, match=r"confidence 必須為 1\.0"):
        CorporateAction(
            entity_id=TSMC,
            action_type=ActionType.SPLIT,
            ex_date=EX_DATE,
            factor=Decimal("0.5"),
            source=ActionSource.OFFICIAL,
            confidence=0.9,
            ingest_time=DIVIDEND_INGEST,
        )


def test_inferred_source_must_not_claim_full_confidence() -> None:
    """推測不得偽裝成事實。"""
    with pytest.raises(ValidationError, match="讓推測看起來像事實"):
        CorporateAction(
            entity_id=TSMC,
            action_type=ActionType.SPLIT,
            ex_date=EX_DATE,
            factor=Decimal("0.5"),
            source=ActionSource.INFERRED,
            confidence=1.0,
            ingest_time=DIVIDEND_INGEST,
        )


def test_min_confidence_propagates_to_the_result() -> None:
    inferred = CorporateAction(
        entity_id=TSMC,
        action_type=ActionType.SPLIT,
        ex_date=date(2024, 3, 13),
        factor=Decimal("0.5"),
        source=ActionSource.INFERRED,
        confidence=0.6,
        ingest_time=DIVIDEND_INGEST,
    )
    store = CorporateActionStore([DIVIDEND, inferred])
    result = get_adjusted_prices(
        PRICES,
        store,
        entity_id=TSMC,
        start=date(2024, 3, 12),
        end=date(2024, 3, 12),
        asof=datetime(2024, 3, 25, tzinfo=UTC),
    )
    assert result[0].min_confidence == 0.6


def test_zero_or_negative_factor_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CorporateAction(
            entity_id=TSMC,
            action_type=ActionType.SPLIT,
            ex_date=EX_DATE,
            factor=Decimal("0"),
            source=ActionSource.OFFICIAL,
            confidence=1.0,
            ingest_time=DIVIDEND_INGEST,
        )


def test_naive_ingest_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RawPrice(
            entity_id=TSMC,
            trade_date=date(2024, 3, 11),
            close=Decimal("100"),
            volume=0,
            ingest_time=datetime(2024, 3, 1),  # noqa: DTZ001
        )


# --- 前視偵測 --------------------------------------------------------------


def test_assert_no_lookahead_passes_on_clean_data() -> None:
    assert_no_lookahead(PRICES, datetime(2024, 3, 5, tzinfo=UTC))


def test_assert_no_lookahead_detects_future_data() -> None:
    late = price(15, "99", ingest=datetime(2024, 4, 1, tzinfo=UTC))
    with pytest.raises(LookaheadError) as excinfo:
        assert_no_lookahead([*PRICES, late], datetime(2024, 3, 20, tzinfo=UTC))
    assert excinfo.value.context["offender_count"] == 1


def test_assert_no_lookahead_rejects_naive_asof() -> None:
    with pytest.raises(NaiveDatetimeError):
        assert_no_lookahead(PRICES, datetime(2024, 3, 20))  # noqa: DTZ001


def test_store_actions_for_filters_by_ingest_time() -> None:
    store = CorporateActionStore([DIVIDEND])
    assert store.actions_for(TSMC, asof=datetime(2024, 3, 19, tzinfo=UTC)) == []
    assert len(store.actions_for(TSMC, asof=datetime(2024, 3, 21, tzinfo=UTC))) == 1


def test_empty_window_returns_nothing() -> None:
    result = get_adjusted_prices(
        PRICES,
        CorporateActionStore(),
        entity_id=TSMC,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        asof=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert result == []
