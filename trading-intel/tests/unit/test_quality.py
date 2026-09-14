"""資料品質閘門：五種髒資料都要被攔下並產生正確告警。

SPEC 第 11 節 Phase 1 驗收條件：「刻意注入五種髒資料（缺 bar、負量、
high<low、重複、過期），五種都被閘門攔下且產生正確告警」。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market, QualityCheck, Severity, TradingState
from trading_intel.core.ids import make_entity_id
from trading_intel.core.settings import QualityLimits
from trading_intel.normalize.corporate_actions import RawPrice
from trading_intel.normalize.quality import (
    check_completeness,
    check_consistency,
    check_drift,
    check_freshness,
    check_raw_sanity,
    check_sanity,
    population_stability_index,
    resulting_state,
)

TSMC = make_entity_id(Market.TW, "2330")
ASOF = datetime(2024, 3, 20, 0, 0, tzinfo=UTC)
INGEST = datetime(2024, 3, 19, 12, 0, tzinfo=UTC)

LIMITS = QualityLimits(
    max_staleness_minutes=1440,
    tw_daily_return_limit=0.105,
    us_daily_return_limit=0.50,
    max_missing_session_ratio=0.02,
    psi_warn=0.25,
    psi_disable=0.5,
    cross_source_tolerance=0.001,
)

SESSIONS = [date(2024, 3, day) for day in (11, 12, 13, 14, 15, 18, 19)]


def price(day: int, close: str, volume: int = 1000, ingest: datetime = INGEST) -> RawPrice:
    return RawPrice(
        entity_id=TSMC,
        trade_date=date(2024, 3, day),
        close=Decimal(close),
        volume=volume,
        ingest_time=ingest,
    )


CLEAN = [
    price(11, "100"),
    price(12, "101"),
    price(13, "102"),
    price(14, "101"),
    price(15, "103"),
    price(18, "104"),
    price(19, "105"),
]


def test_clean_data_passes_every_gate() -> None:
    """乾淨資料不得產生任何告警——否則閘門就是雜訊產生器。"""
    alerts = [
        *check_freshness(CLEAN, entity_id=TSMC, asof=ASOF, limits=LIMITS),
        *check_completeness(CLEAN, SESSIONS, entity_id=TSMC, asof=ASOF, limits=LIMITS),
        *check_sanity(CLEAN, entity_id=TSMC, market=Market.TW, asof=ASOF, limits=LIMITS),
    ]
    assert alerts == []
    assert resulting_state(alerts) is TradingState.TRADABLE


# --- 髒資料一：缺 bar ------------------------------------------------------


def test_missing_bar_is_caught() -> None:
    dirty = [item for item in CLEAN if item.trade_date != date(2024, 3, 13)]
    alerts = check_completeness(dirty, SESSIONS, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert len(alerts) == 1
    assert alerts[0].check is QualityCheck.COMPLETENESS
    assert alerts[0].severity is Severity.P0
    assert alerts[0].resulting_state is TradingState.NO_TRADE
    assert "2024-03-13" in alerts[0].detail


def test_missing_bars_below_the_tolerance_do_not_alert() -> None:
    many_sessions = [date(2024, 3, 1) + timedelta(days=index) for index in range(100)]
    prices = [
        RawPrice(
            entity_id=TSMC,
            trade_date=day,
            close=Decimal("100"),
            volume=1,
            ingest_time=INGEST,
        )
        for day in many_sessions[1:]
    ]
    alerts = check_completeness(prices, many_sessions, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert alerts == []


# --- 髒資料二：負成交量 ----------------------------------------------------


def test_negative_volume_is_caught() -> None:
    """負量在原始值層攔下。RawPrice 的 ge=0 讓它根本建構不起來，
    所以閘門必須在型別驗證之前就檢查。"""
    rows = [
        (date(2024, 3, 13), Decimal("102"), 1000),
        (date(2024, 3, 14), Decimal("101"), -500),
    ]
    alerts = check_raw_sanity(rows, entity_id=TSMC, asof=ASOF)
    assert len(alerts) == 1
    assert alerts[0].check is QualityCheck.SANITY
    assert "成交量為負數" in alerts[0].detail
    assert resulting_state(alerts) is TradingState.NO_TRADE


def test_zero_volume_with_price_change_is_caught() -> None:
    dirty = [*CLEAN[:3], price(14, "101", volume=0), *CLEAN[4:]]
    alerts = check_completeness(dirty, SESSIONS, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert any("成交量為零但價格" in alert.detail for alert in alerts)


# --- 髒資料三：high < low --------------------------------------------------


def test_high_below_low_is_caught() -> None:
    bars = [
        (date(2024, 3, 11), Decimal("100"), Decimal("99"), Decimal("101"), Decimal("100")),
    ]
    alerts = check_consistency(bars, entity_id=TSMC, asof=ASOF)
    assert len(alerts) == 1
    assert alerts[0].check is QualityCheck.CONSISTENCY
    assert "high 99 < low 101" in alerts[0].detail


def test_close_outside_high_low_is_caught() -> None:
    bars = [
        (date(2024, 3, 11), Decimal("100"), Decimal("105"), Decimal("99"), Decimal("110")),
    ]
    alerts = check_consistency(bars, entity_id=TSMC, asof=ASOF)
    assert "close 110 落在" in alerts[0].detail


def test_open_outside_high_low_is_caught() -> None:
    bars = [
        (date(2024, 3, 11), Decimal("90"), Decimal("105"), Decimal("99"), Decimal("100")),
    ]
    alerts = check_consistency(bars, entity_id=TSMC, asof=ASOF)
    assert "open 90 落在" in alerts[0].detail


def test_consistent_bar_passes() -> None:
    bars = [
        (date(2024, 3, 11), Decimal("100"), Decimal("105"), Decimal("99"), Decimal("103")),
    ]
    assert check_consistency(bars, entity_id=TSMC, asof=ASOF) == []


# --- 髒資料四：重複／異常跳動（未登錄的企業行動）---------------------------


def test_impossible_daily_move_is_caught() -> None:
    """台股有漲跌幅限制，單日 -50% 必是未登錄的企業行動而非真實報酬。"""
    dirty = [*CLEAN[:3], price(14, "50"), *CLEAN[4:]]
    alerts = check_sanity(dirty, entity_id=TSMC, market=Market.TW, asof=ASOF, limits=LIMITS)
    assert any("超過 TW 市場的" in alert.detail for alert in alerts)
    assert any("未登錄的企業行動" in alert.detail for alert in alerts)


def test_a_move_within_the_limit_passes() -> None:
    dirty = [price(13, "102"), price(14, "92")]  # -9.8%，在 10.5% 內
    alerts = check_sanity(dirty, entity_id=TSMC, market=Market.TW, asof=ASOF, limits=LIMITS)
    assert alerts == []


def test_us_market_uses_a_looser_limit() -> None:
    dirty = [price(13, "100"), price(14, "80")]  # -20%
    tw_alerts = check_sanity(dirty, entity_id=TSMC, market=Market.TW, asof=ASOF, limits=LIMITS)
    us_alerts = check_sanity(dirty, entity_id=TSMC, market=Market.US, asof=ASOF, limits=LIMITS)
    assert tw_alerts != []
    assert us_alerts == []


def test_duplicate_rows_do_not_double_count() -> None:
    """同一天重複送達應收斂為一筆，不得被誤判為零報酬序列。"""
    duplicated = [*CLEAN, price(19, "105")]
    alerts = check_completeness(duplicated, SESSIONS, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert alerts == []


# --- 髒資料五：過期 --------------------------------------------------------


def test_stale_data_is_caught() -> None:
    stale = [price(11, "100", ingest=datetime(2024, 3, 1, tzinfo=UTC))]
    alerts = check_freshness(stale, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert len(alerts) == 1
    assert alerts[0].check is QualityCheck.FRESHNESS
    assert alerts[0].severity is Severity.P0
    assert "已過期" in alerts[0].detail


def test_no_visible_data_is_caught() -> None:
    future = [price(11, "100", ingest=datetime(2024, 4, 1, tzinfo=UTC))]
    alerts = check_freshness(future, entity_id=TSMC, asof=ASOF, limits=LIMITS)
    assert "沒有任何可見的價格資料" in alerts[0].detail


def test_fresh_data_passes() -> None:
    assert check_freshness(CLEAN, entity_id=TSMC, asof=ASOF, limits=LIMITS) == []


# --- 分布漂移 --------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions() -> None:
    values = [float(index) for index in range(100)]
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_divergence() -> None:
    baseline = [float(index) for index in range(100)]
    mild = [float(index) + 5 for index in range(100)]
    severe = [float(index) + 500 for index in range(100)]
    assert population_stability_index(baseline, mild) < population_stability_index(baseline, severe)


def test_drift_beyond_the_disable_threshold_stops_trading() -> None:
    baseline = [float(index) for index in range(100)]
    shifted = [float(index) + 500 for index in range(100)]
    alerts = check_drift(
        baseline, shifted, entity_id=TSMC, feature_name="mom_20d", asof=ASOF, limits=LIMITS
    )
    assert alerts[0].severity is Severity.P0
    assert alerts[0].resulting_state is TradingState.NO_TRADE
    assert "停用門檻" in alerts[0].detail


def test_no_drift_yields_no_alert() -> None:
    values = [float(index) for index in range(100)]
    assert (
        check_drift(values, values, entity_id=TSMC, feature_name="f", asof=ASOF, limits=LIMITS)
        == []
    )


def test_psi_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="都不得為空"):
        population_stability_index([], [1.0])


def test_psi_rejects_too_few_buckets() -> None:
    with pytest.raises(ValueError, match="至少為 2"):
        population_stability_index([1.0], [1.0], buckets=1)


# --- 彙整 ------------------------------------------------------------------


def test_any_blocking_alert_stops_trading() -> None:
    """不做平均、不做投票：一個要求停止就停止。"""
    alerts = check_raw_sanity([(date(2024, 3, 14), Decimal("101"), -1)], entity_id=TSMC, asof=ASOF)
    assert resulting_state(alerts) is TradingState.NO_TRADE


def test_no_alerts_means_tradable() -> None:
    assert resulting_state([]) is TradingState.TRADABLE


def test_non_positive_close_is_caught() -> None:
    alerts = check_raw_sanity([(date(2024, 3, 11), Decimal("0"), 1)], entity_id=TSMC, asof=ASOF)
    assert any("收盤價非正數" in alert.detail for alert in alerts)


def test_clean_raw_rows_pass() -> None:
    rows = [(date(2024, 3, 11), Decimal("100"), 1000)]
    assert check_raw_sanity(rows, entity_id=TSMC, asof=ASOF) == []
