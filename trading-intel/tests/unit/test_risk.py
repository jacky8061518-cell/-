"""三層風控（SPEC 7）。

SPEC 第 11 節 Phase 4 驗收條件：
- 風控測試必須涵蓋**每一條限額的觸發與未觸發邊界**
- 模擬回撤情境，驗證分層停損確實逐級生效
- 心跳中斷模擬，驗證系統進入只平倉模式
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market, Severity
from trading_intel.core.ids import make_entity_id
from trading_intel.core.settings import ModelRiskSettings, MonitoringSettings, RiskLimits
from trading_intel.risk.model_risk import (
    SignalHealth,
    SignalMonitor,
    agent_consistency_acceptable,
    information_coefficient,
)
from trading_intel.risk.monitoring import (
    OrderGuard,
    TradingMode,
    TriggerReason,
    check_drawdown,
    check_heartbeat,
    check_realized_volatility,
    check_reconciliation,
    combine_alerts,
)
from trading_intel.risk.pretrade import (
    Breach,
    LimitCode,
    PortfolioState,
    check_pretrade,
)

A = make_entity_id(Market.TW, "1111")
B = make_entity_id(Market.TW, "2222")
C = make_entity_id(Market.TW, "3333")
D = make_entity_id(Market.TW, "4444")
E = make_entity_id(Market.TW, "5555")
F = make_entity_id(Market.TW, "6666")

T0 = datetime(2024, 3, 19, 9, 0, tzinfo=UTC)

LIMITS = RiskLimits(
    max_position_weight=0.05,
    max_sector_weight=0.25,
    max_gross_exposure=1.0,
    max_net_exposure=0.6,
    drawdown_derisk=0.08,
    drawdown_flatten=0.15,
    adv_participation_cap=0.05,
    top5_concentration_cap=0.4,
)

MONITORING = MonitoringSettings(
    heartbeat_timeout_seconds=300,
    reconciliation_interval_minutes=15,
    realized_vol_multiple_for_derisk=1.5,
    max_orders_per_minute=10,
    max_price_deviation=0.05,
)

MODEL_RISK = ModelRiskSettings(
    ic_decay_periods=5,
    min_rolling_ic=0.0,
    max_live_backtest_tracking_error=0.05,
    min_agent_consistency=0.67,
)


def state(**overrides: object) -> PortfolioState:
    defaults: dict[str, object] = {
        "weights": {},
        "sectors": {A: "半導體", B: "半導體", C: "金融", D: "金融", E: "航運", F: "航運"},
        "adv_values": dict.fromkeys([A, B, C, D, E, F], Decimal("1000000000")),
        "equity": Decimal("10000000"),
    }
    defaults.update(overrides)
    return PortfolioState(**defaults)  # type: ignore[arg-type]


def codes(breaches: tuple[Breach, ...]) -> set[LimitCode]:
    return {breach.code for breach in breaches}


# --- 事前風控：每條限額的觸發與未觸發邊界 ----------------------------------


def test_all_limits_pass_on_a_clean_portfolio() -> None:
    verdict, breaches = check_pretrade({A: 0.04, C: 0.04}, state(), LIMITS, now=T0)
    assert verdict.approved
    assert breaches == ()
    assert verdict.breached_limits == ()


def test_position_limit_at_the_boundary_passes() -> None:
    """剛好等於上限應通過——邊界是包含的。"""
    verdict, breaches = check_pretrade({A: 0.05}, state(), LIMITS, now=T0)
    assert verdict.approved
    assert LimitCode.MAX_POSITION_WEIGHT not in codes(breaches)


def test_position_limit_just_over_the_boundary_fails() -> None:
    verdict, breaches = check_pretrade({A: 0.0501}, state(), LIMITS, now=T0)
    assert not verdict.approved
    assert LimitCode.MAX_POSITION_WEIGHT in codes(breaches)


def test_position_limit_applies_to_short_positions() -> None:
    _verdict, breaches = check_pretrade({A: -0.06}, state(), LIMITS, now=T0)
    assert LimitCode.MAX_POSITION_WEIGHT in codes(breaches)


def test_sector_limit_at_the_boundary_passes() -> None:
    weights = {A: 0.05, B: 0.05, C: 0.05, D: 0.05, E: 0.05}
    _verdict, breaches = check_pretrade(weights, state(), LIMITS, now=T0)
    assert LimitCode.MAX_SECTOR_WEIGHT not in codes(breaches)


def test_sector_limit_over_the_boundary_fails() -> None:
    """半導體兩檔各 0.05 只有 0.1，要超過 0.25 需要更多檔。"""
    sectors = dict.fromkeys([A, B, C, D, E, F], "半導體")
    weights = {A: 0.05, B: 0.05, C: 0.05, D: 0.05, E: 0.05, F: 0.05}
    _verdict, breaches = check_pretrade(weights, state(sectors=sectors), LIMITS, now=T0)
    assert LimitCode.MAX_SECTOR_WEIGHT in codes(breaches)


def test_gross_exposure_boundary() -> None:
    loose = RiskLimits(
        **{
            **LIMITS.model_dump(),
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "top5_concentration_cap": 1.0,
            "max_net_exposure": 1.0,
        }
    )
    at_limit = {A: 0.5, B: 0.5}
    over = {A: 0.5, B: 0.51}
    assert LimitCode.MAX_GROSS_EXPOSURE not in codes(
        check_pretrade(at_limit, state(), loose, now=T0)[1]
    )
    assert LimitCode.MAX_GROSS_EXPOSURE in codes(check_pretrade(over, state(), loose, now=T0)[1])


def test_net_exposure_boundary() -> None:
    loose = RiskLimits(
        **{
            **LIMITS.model_dump(),
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "top5_concentration_cap": 1.0,
        }
    )
    assert LimitCode.MAX_NET_EXPOSURE not in codes(
        check_pretrade({A: 0.6}, state(), loose, now=T0)[1]
    )
    assert LimitCode.MAX_NET_EXPOSURE in codes(check_pretrade({A: 0.61}, state(), loose, now=T0)[1])


def test_net_exposure_allows_offsetting_shorts() -> None:
    """多空相抵後的淨曝險才是受限的對象。"""
    loose = RiskLimits(
        **{
            **LIMITS.model_dump(),
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "top5_concentration_cap": 1.0,
        }
    )
    _verdict, breaches = check_pretrade({A: 0.5, B: -0.4}, state(), loose, now=T0)
    assert LimitCode.MAX_NET_EXPOSURE not in codes(breaches)


def test_adv_participation_boundary() -> None:
    """流動性：部位不得超過 20 日均量的設定比例。"""
    # 權益一千萬，均量一億，5% 權重即 50 萬，佔均量 0.5%，通過。
    assert LimitCode.ADV_PARTICIPATION not in codes(
        check_pretrade({A: 0.05}, state(), LIMITS, now=T0)[1]
    )
    # 均量縮到五百萬，50 萬即佔 10%，超過 5% 上限。
    thin = state(adv_values={A: Decimal("5000000")})
    breaches = check_pretrade({A: 0.05}, thin, LIMITS, now=T0)[1]
    assert LimitCode.ADV_PARTICIPATION in codes(breaches)
    assert "無法在實盤重現" in next(
        item.detail for item in breaches if item.code is LimitCode.ADV_PARTICIPATION
    )


def test_missing_adv_data_skips_the_check() -> None:
    _verdict, breaches = check_pretrade({A: 0.05}, state(adv_values={}), LIMITS, now=T0)
    assert LimitCode.ADV_PARTICIPATION not in codes(breaches)


def test_top5_concentration_boundary() -> None:
    loose = RiskLimits(
        **{**LIMITS.model_dump(), "max_position_weight": 0.1, "max_sector_weight": 1.0}
    )
    at_limit = {A: 0.08, B: 0.08, C: 0.08, D: 0.08, E: 0.08}
    over = {A: 0.09, B: 0.09, C: 0.09, D: 0.09, E: 0.09}
    assert LimitCode.TOP5_CONCENTRATION not in codes(
        check_pretrade(at_limit, state(), loose, now=T0)[1]
    )
    assert LimitCode.TOP5_CONCENTRATION in codes(check_pretrade(over, state(), loose, now=T0)[1])


def test_restricted_list_blocks_new_positions() -> None:
    _verdict, breaches = check_pretrade({A: 0.03}, state(), LIMITS, restricted=[A], now=T0)
    assert LimitCode.RESTRICTED_LIST in codes(breaches)


def test_restricted_list_allows_closing_to_zero() -> None:
    """禁止清單擋的是建立與加碼，不擋平倉。"""
    _verdict, breaches = check_pretrade({A: 0.0}, state(), LIMITS, restricted=[A], now=T0)
    assert LimitCode.RESTRICTED_LIST not in codes(breaches)


def test_all_breaches_are_reported_not_just_the_first() -> None:
    """交易員需要知道違反了哪幾條，而不是「有一條違規，修好再來」。"""
    sectors = dict.fromkeys([A, B, C], "半導體")
    weights = {A: 0.3, B: 0.3, C: 0.3}
    _verdict, breaches = check_pretrade(weights, state(sectors=sectors), LIMITS, now=T0)
    assert len(codes(breaches)) >= 3


def test_verdict_contract_requires_breaches_when_rejected() -> None:
    """Phase 0 的型別契約：否決時必須列出至少一項違反的限額。"""
    verdict, _breaches = check_pretrade({A: 0.5}, state(), LIMITS, now=T0)
    assert not verdict.approved
    assert verdict.breached_limits


def test_intent_hash_is_deterministic() -> None:
    first, _ = check_pretrade({A: 0.03, B: 0.02}, state(), LIMITS, now=T0)
    second, _ = check_pretrade({B: 0.02, A: 0.03}, state(), LIMITS, now=T0)
    assert first.intent_hash == second.intent_hash


# --- 分層停損：逐級生效 ----------------------------------------------------


def curve(values: list[str]) -> dict[datetime, Decimal]:
    return {T0 + timedelta(days=index): Decimal(value) for index, value in enumerate(values)}


def test_no_drawdown_no_alert() -> None:
    assert check_drawdown(curve(["100", "105", "110"]), LIMITS, now=T0) is None


def test_drawdown_below_the_first_threshold_does_not_trigger() -> None:
    # 從 110 跌到 103，回撤 6.4%，未達 8%。
    assert check_drawdown(curve(["100", "110", "103"]), LIMITS, now=T0) is None


def test_first_tier_halves_exposure() -> None:
    """回撤 8% 自動減半曝險。"""
    # 110 → 100，回撤 9.1%。
    alert = check_drawdown(curve(["100", "110", "100"]), LIMITS, now=T0)
    assert alert is not None
    assert alert.reason is TriggerReason.DRAWDOWN_DERISK
    assert alert.mode is TradingMode.REDUCE_ONLY
    assert alert.exposure_multiplier == 0.5


def test_second_tier_flattens_and_escalates() -> None:
    """回撤 15% 全部平倉並進入人工審核。"""
    # 110 → 90，回撤 18.2%。
    alert = check_drawdown(curve(["100", "110", "90"]), LIMITS, now=T0)
    assert alert is not None
    assert alert.reason is TriggerReason.DRAWDOWN_FLATTEN
    assert alert.mode is TradingMode.CLOSE_ONLY
    assert alert.exposure_multiplier == 0.0
    assert "人工審核" in alert.detail


def test_drawdown_tiers_escalate_in_order() -> None:
    """模擬逐步惡化的回撤，驗證分層確實逐級生效。"""
    peak = ["100", "110"]
    stages = [
        (["105"], None),
        (["100"], TriggerReason.DRAWDOWN_DERISK),
        (["90"], TriggerReason.DRAWDOWN_FLATTEN),
    ]
    for tail, expected in stages:
        alert = check_drawdown(curve(peak + tail), LIMITS, now=T0)
        assert (alert.reason if alert else None) is expected


def test_empty_curve_is_safe() -> None:
    assert check_drawdown({}, LIMITS, now=T0) is None


# --- 心跳：dead man's switch ----------------------------------------------


def test_fresh_heartbeat_passes() -> None:
    assert check_heartbeat(T0, MONITORING, now=T0 + timedelta(seconds=299)) is None


def test_heartbeat_at_the_boundary_passes() -> None:
    assert check_heartbeat(T0, MONITORING, now=T0 + timedelta(seconds=300)) is None


def test_lost_heartbeat_enters_close_only_mode() -> None:
    """SPEC 驗收：心跳中斷模擬，驗證系統進入只平倉模式。"""
    alert = check_heartbeat(T0, MONITORING, now=T0 + timedelta(seconds=301))
    assert alert is not None
    assert alert.reason is TriggerReason.HEARTBEAT_LOST
    assert alert.mode is TradingMode.CLOSE_ONLY
    assert alert.severity is Severity.P0
    assert alert.exposure_multiplier == 0.0


# --- 波動率 ----------------------------------------------------------------


def test_volatility_within_target_does_not_trigger() -> None:
    assert check_realized_volatility(0.15, 0.15, MONITORING, now=T0) is None


def test_volatility_at_the_multiple_boundary_passes() -> None:
    assert check_realized_volatility(0.225, 0.15, MONITORING, now=T0) is None


def test_volatility_spike_reduces_exposure() -> None:
    alert = check_realized_volatility(0.30, 0.15, MONITORING, now=T0)
    assert alert is not None
    assert alert.reason is TriggerReason.VOLATILITY_SPIKE
    assert alert.exposure_multiplier == pytest.approx(0.5)


# --- 對帳 ------------------------------------------------------------------


def test_matching_positions_pass() -> None:
    positions = {A: Decimal("1000"), B: Decimal("500")}
    assert check_reconciliation(positions, dict(positions), now=T0) is None


def test_mismatch_halts_trading() -> None:
    """對帳不符代表我們對自己的部位認知是錯的，直接凍結。"""
    alert = check_reconciliation({A: Decimal("1000")}, {A: Decimal("900")}, now=T0)
    assert alert is not None
    assert alert.mode is TradingMode.HALTED
    assert alert.reason is TriggerReason.RECONCILIATION_MISMATCH


def test_position_missing_at_broker_is_a_mismatch() -> None:
    alert = check_reconciliation({A: Decimal("1000")}, {}, now=T0)
    assert alert is not None


def test_tiny_difference_within_tolerance_passes() -> None:
    assert check_reconciliation({A: Decimal("1000.00001")}, {A: Decimal("1000")}, now=T0) is None


# --- 下單防護 --------------------------------------------------------------


def test_order_rate_limit() -> None:
    guard = OrderGuard(settings=MONITORING)
    for index in range(10):
        assert guard.record_order(T0, f"key-{index}")
    alert = guard.check_rate(T0)
    assert alert is not None
    assert alert.reason is TriggerReason.ORDER_RATE_LIMIT


def test_order_rate_window_slides() -> None:
    guard = OrderGuard(settings=MONITORING)
    for index in range(10):
        guard.record_order(T0, f"key-{index}")
    assert guard.check_rate(T0 + timedelta(minutes=2)) is None


def test_duplicate_order_is_rejected() -> None:
    guard = OrderGuard(settings=MONITORING)
    assert guard.record_order(T0, "same-key")
    assert not guard.record_order(T0, "same-key")


def test_price_deviation_within_tolerance_passes() -> None:
    guard = OrderGuard(settings=MONITORING)
    assert guard.check_price(Decimal("104"), Decimal("100"), now=T0) is None


def test_price_deviation_beyond_tolerance_triggers() -> None:
    guard = OrderGuard(settings=MONITORING)
    alert = guard.check_price(Decimal("110"), Decimal("100"), now=T0)
    assert alert is not None
    assert alert.reason is TriggerReason.PRICE_DEVIATION


# --- 彙整：取最嚴格者 ------------------------------------------------------


def test_combine_takes_the_strictest_mode() -> None:
    """不做平均、不做投票：任一個要求停止就停止。"""
    derisk = check_drawdown(curve(["100", "110", "100"]), LIMITS, now=T0)
    halt = check_reconciliation({A: Decimal("1")}, {A: Decimal("2")}, now=T0)
    mode, multiplier = combine_alerts([derisk, halt])
    assert mode is TradingMode.HALTED
    assert multiplier == 0.0


def test_combine_with_no_alerts_is_normal() -> None:
    mode, multiplier = combine_alerts([None, None])
    assert mode is TradingMode.NORMAL
    assert multiplier == 1.0


# --- 模型風險 --------------------------------------------------------------


def test_ic_of_perfect_ranking_is_one() -> None:
    assert information_coefficient([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_ic_of_inverted_ranking_is_minus_one() -> None:
    assert information_coefficient([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_ic_uses_rank_not_magnitude() -> None:
    """排序對就好，絕對值高估不影響 IC。"""
    assert information_coefficient([1, 2, 3], [100, 200, 300]) == pytest.approx(1.0)


def test_ic_of_too_few_points_is_nan() -> None:
    import math

    assert math.isnan(information_coefficient([1, 2], [1, 2]))


def test_healthy_signal_stays_healthy() -> None:
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.06, settings=MODEL_RISK)
    for _ in range(10):
        monitor.record(0.05)
    assert monitor.health is SignalHealth.HEALTHY
    assert monitor.tradable


def test_one_bad_period_moves_to_watch() -> None:
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.06, settings=MODEL_RISK)
    monitor.record(0.05)
    assert monitor.record(-0.01) is SignalHealth.WATCH


def test_consecutive_bad_periods_quarantine_the_signal() -> None:
    """連續 N 期低於下限自動隔離（SPEC 7.3）。"""
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.06, settings=MODEL_RISK)
    for _ in range(4):
        monitor.record(-0.01)
    assert monitor.health is SignalHealth.WATCH
    assert monitor.record(-0.01) is SignalHealth.QUARANTINED
    assert not monitor.tradable


def test_recovery_resets_the_counter() -> None:
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.06, settings=MODEL_RISK)
    for _ in range(4):
        monitor.record(-0.01)
    monitor.record(0.05)
    assert monitor.health is SignalHealth.HEALTHY


def test_tracking_error_detects_live_underperformance() -> None:
    """回測 IC 0.08、上線 0.02，即使仍為正也代表有東西不對。"""
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.08, settings=MODEL_RISK)
    for _ in range(10):
        monitor.record(0.02)
    assert monitor.live_vs_backtest_gap() == pytest.approx(0.06)
    assert monitor.tracking_error_breached()


def test_small_gap_does_not_breach() -> None:
    monitor = SignalMonitor(signal_name="mom", backtest_ic=0.06, settings=MODEL_RISK)
    for _ in range(10):
        monitor.record(0.05)
    assert not monitor.tracking_error_breached()


def test_agent_consistency_gate() -> None:
    assert agent_consistency_acceptable([1.0, 1.0, 0.67], MODEL_RISK)
    assert not agent_consistency_acceptable([0.5, 0.4], MODEL_RISK)
    assert not agent_consistency_acceptable([], MODEL_RISK)
