"""晨報與風險儀表板：組裝已算好的數字，不重新計算。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.agents.schemas import AttackFinding, AttackSeverity, AttackVector, RedTeamReport
from trading_intel.core.clock import UTC
from trading_intel.core.enums import DecisionLevel, Direction, Horizon, Market
from trading_intel.core.ids import EntityId, EvidenceId, SignalId, make_entity_id
from trading_intel.core.settings import RiskLimits
from trading_intel.core.types import Signal
from trading_intel.models.regime import LiquidityState, MarketRegime, TrendState, VolatilityState
from trading_intel.risk.model_risk import SignalHealth
from trading_intel.risk.monitoring import TradingMode
from trading_intel.risk.pretrade import PortfolioState
from trading_intel.surface.attribution import AttributionResult
from trading_intel.surface.dashboard import build_dashboard
from trading_intel.surface.morning_brief import (
    CalendarEvent,
    RiskSnapshot,
    build_morning_brief,
)
from trading_intel.surface.signal_card import SignalCard, build_signal_card

TSMC = make_entity_id(Market.TW, "2330")
UMC = make_entity_id(Market.TW, "2303")
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


def make_card() -> SignalCard:
    signal = Signal(
        event_time=T0,
        ingest_time=T0 + timedelta(minutes=1),
        signal_id=SignalId("sig1"),
        entity_id=TSMC,
        direction=Direction.LONG,
        score=0.5,
        confidence=0.7,
        half_life_days=10.0,
        invalidation_condition="月營收年增率轉為負成長",
        horizon=Horizon.WEEKS,
        evidence_ids=(EvidenceId("ev1"),),
        model_version="v1",
        decision_level=DecisionLevel.CONFIRM,
    )
    report = RedTeamReport(
        findings=tuple(
            AttackFinding(vector=v, severity=AttackSeverity.LOW, detail="通過")
            for v in AttackVector
        ),
        confidence=0.8,
        reasoning_digest="已檢視",
    )
    return build_signal_card(signal, report, generated_at=T0)


NORMAL_REGIME = MarketRegime(
    trend=TrendState.RANGING,
    volatility=VolatilityState.NORMAL,
    liquidity=LiquidityState.NORMAL,
    cross_sectional_correlation=0.3,
)


# --- 晨報 --------------------------------------------------------------------


def test_brief_includes_all_spec_sections() -> None:
    """SPEC 8.1：昨日歸因、regime、待觸發訊號、風險快照、行事曆。"""
    attribution = AttributionResult(
        total_return=0.01,
        beta_return=0.008,
        alpha_return=0.002,
        style_return=0.0,
        cost_drag=0.0,
        timing_return=0.0,
    )
    brief = build_morning_brief(
        trade_date=date(2024, 3, 19),
        yesterday_attribution=attribution,
        regime=NORMAL_REGIME,
        pending_signals=(make_card(),),
        risk_snapshot=RiskSnapshot(gross_exposure=0.6, net_exposure=0.3, top5_concentration=0.35),
        calendar=(
            CalendarEvent(entity_id=TSMC, description="法說會", event_date=date(2024, 3, 20)),
        ),
    )
    text = brief.to_display_text()
    assert "昨日歸因" in text
    assert "今日 regime" in text
    assert "待觸發訊號" in text
    assert "風險曝險快照" in text
    assert "今日行事曆" in text
    assert "法說會" in text


def test_brief_handles_missing_prior_attribution() -> None:
    """第一個交易日沒有昨日歸因，不該崩潰。"""
    brief = build_morning_brief(
        trade_date=date(2024, 1, 1),
        yesterday_attribution=None,
        regime=NORMAL_REGIME,
        pending_signals=(),
        risk_snapshot=RiskSnapshot(gross_exposure=0.0, net_exposure=0.0, top5_concentration=0.0),
        calendar=(),
    )
    assert "無資料" in brief.to_display_text()


def test_risk_off_regime_is_flagged() -> None:
    extreme = MarketRegime(
        trend=TrendState.TRENDING_DOWN,
        volatility=VolatilityState.EXTREME,
        liquidity=LiquidityState.STRESSED,
        cross_sectional_correlation=0.9,
    )
    brief = build_morning_brief(
        trade_date=date(2024, 3, 19),
        yesterday_attribution=None,
        regime=extreme,
        pending_signals=(),
        risk_snapshot=RiskSnapshot(gross_exposure=0.0, net_exposure=0.0, top5_concentration=0.0),
        calendar=(),
    )
    assert "risk-off" in brief.to_display_text()


def test_blocking_counter_argument_is_flagged_in_the_brief() -> None:
    card = make_card()
    flagged_card = build_signal_card(
        Signal(
            event_time=T0,
            ingest_time=T0 + timedelta(minutes=1),
            signal_id=SignalId("sig2"),
            entity_id=UMC,
            direction=Direction.SHORT,
            score=-0.4,
            confidence=0.6,
            half_life_days=5.0,
            invalidation_condition="轉為正向財測大幅修正",
            horizon=Horizon.DAYS,
            evidence_ids=(EvidenceId("ev2"),),
            model_version="v1",
            decision_level=DecisionLevel.RESEARCH_ONLY,
        ),
        RedTeamReport(
            findings=(
                AttackFinding(
                    vector=AttackVector.DATA_LEAKAGE,
                    severity=AttackSeverity.CRITICAL,
                    detail="洩漏",
                ),
                *[
                    AttackFinding(vector=v, severity=AttackSeverity.LOW, detail="通過")
                    for v in list(AttackVector)[1:]
                ],
            ),
            confidence=0.5,
            reasoning_digest="有疑慮",
        ),
        generated_at=T0,
    )
    brief = build_morning_brief(
        trade_date=date(2024, 3, 19),
        yesterday_attribution=None,
        regime=NORMAL_REGIME,
        pending_signals=(card, flagged_card),
        risk_snapshot=RiskSnapshot(gross_exposure=0.0, net_exposure=0.0, top5_concentration=0.0),
        calendar=(),
    )
    assert "有反方高風險意見" in brief.to_display_text()


# --- 風險儀表板 ----------------------------------------------------------------


def portfolio_state(
    *,
    weights: dict[EntityId, float] | None = None,
    sectors: dict[EntityId, str] | None = None,
    adv_values: dict[EntityId, Decimal] | None = None,
    equity: Decimal = Decimal("10000000"),
) -> PortfolioState:
    return PortfolioState(
        weights=weights if weights is not None else {TSMC: 0.05, UMC: 0.04},
        sectors=sectors if sectors is not None else {TSMC: "半導體", UMC: "半導體"},
        adv_values=adv_values if adv_values is not None else {},
        equity=equity,
    )


def test_dashboard_reports_limit_usage_ratios() -> None:
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(),
        limits=LIMITS,
        trading_mode=TradingMode.NORMAL,
        signal_healths={},
        equity_curve={date(2024, 3, 18): Decimal("1000000"), date(2024, 3, 19): Decimal("1000000")},
    )
    position_usage = next(u for u in dashboard.limit_usages if "2330" in u.name)
    assert position_usage.usage_ratio == pytest.approx(1.0)  # 0.05 / 0.05


def test_dashboard_flags_near_limit() -> None:
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(weights={TSMC: 0.045}),
        limits=LIMITS,
        trading_mode=TradingMode.NORMAL,
        signal_healths={},
        equity_curve={},
    )
    position_usage = next(u for u in dashboard.limit_usages if "2330" in u.name)
    assert position_usage.is_near_limit
    assert position_usage.usage_ratio < 1.0


def test_dashboard_computes_drawdown_from_peak() -> None:
    curve = {
        date(2024, 3, 1): Decimal("1000000"),
        date(2024, 3, 10): Decimal("1100000"),
        date(2024, 3, 19): Decimal("990000"),
    }
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(),
        limits=LIMITS,
        trading_mode=TradingMode.NORMAL,
        signal_healths={},
        equity_curve=curve,
    )
    assert dashboard.drawdown_from_peak == pytest.approx((1100000 - 990000) / 1100000)


def test_dashboard_summarizes_signal_health() -> None:
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(),
        limits=LIMITS,
        trading_mode=TradingMode.REDUCE_ONLY,
        signal_healths={
            "a": SignalHealth.HEALTHY,
            "b": SignalHealth.QUARANTINED,
            "c": SignalHealth.HEALTHY,
        },
        equity_curve={},
    )
    assert dashboard.signal_health.healthy == 2
    assert dashboard.signal_health.quarantined == 1
    assert dashboard.signal_health.total == 3


def test_breached_limits_are_identified() -> None:
    breached = LIMITS.model_copy(update={"max_position_weight": 0.03})
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(),
        limits=breached,
        trading_mode=TradingMode.NORMAL,
        signal_healths={},
        equity_curve={},
    )
    assert len(dashboard.breached_limits) > 0


def test_display_text_shows_trading_mode() -> None:
    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio_state(),
        limits=LIMITS,
        trading_mode=TradingMode.HALTED,
        signal_healths={},
        equity_curve={},
    )
    assert "HALTED" in dashboard.to_display_text()
