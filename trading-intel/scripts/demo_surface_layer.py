"""用真正的 surface 層程式碼組出一組示範資料，輸出成 JSON 供展示頁面使用。

股票代號、金額、文字內容都是虛構的範例（不是即時資料），但物件的組裝、
歸因恆等式、限額使用率、分級告警的去重與速率上限、紙上交易週報——
全部呼叫 ``src/trading_intel/surface`` 底下真正上線的函式，
不是另外手刻的假邏輯。

用法：``uv run python scripts/demo_surface_layer.py > /tmp/demo.json``
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

from trading_intel.agents.schemas import AttackFinding, AttackSeverity, AttackVector, RedTeamReport
from trading_intel.core.clock import UTC
from trading_intel.core.enums import DecisionLevel, Direction, Horizon, Market, Severity
from trading_intel.core.ids import EntityId, EvidenceId, SignalId, make_entity_id
from trading_intel.core.settings import AlertingSettings, RiskLimits
from trading_intel.core.types import OrderIntent, RiskVerdict, Signal
from trading_intel.models.regime import LiquidityState, MarketRegime, TrendState, VolatilityState
from trading_intel.risk.model_risk import SignalHealth
from trading_intel.risk.monitoring import TradingMode
from trading_intel.risk.pretrade import PortfolioState
from trading_intel.surface.alerts import Alert, AlertRouter
from trading_intel.surface.attribution import AttributionResult
from trading_intel.surface.dashboard import build_dashboard
from trading_intel.surface.morning_brief import CalendarEvent, RiskSnapshot, build_morning_brief
from trading_intel.surface.paper_trading import PaperTradingBook
from trading_intel.surface.signal_card import build_signal_card

T0 = datetime(2026, 9, 11, 0, 30, tzinfo=UTC)  # 台北早上 08:30

TSMC = make_entity_id(Market.TW, "2330")
MEDIATEK = make_entity_id(Market.TW, "2454")
DELTA = make_entity_id(Market.TW, "2308")
LARGAN = make_entity_id(Market.TW, "3008")


def _make_signal(entity: EntityId, **overrides: object) -> Signal:
    defaults: dict[str, object] = {
        "event_time": T0 - timedelta(hours=2),
        "ingest_time": T0 - timedelta(hours=1, minutes=50),
        "signal_id": SignalId(f"sig-{entity}"),
        "entity_id": entity,
        "direction": Direction.LONG,
        "score": 0.6,
        "confidence": 0.72,
        "half_life_days": 12.0,
        "invalidation_condition": "月營收年增率轉為負成長",
        "horizon": Horizon.WEEKS,
        "evidence_ids": (EvidenceId("ev1"), EvidenceId("ev2")),
        "model_version": "v1.0.0",
        "decision_level": DecisionLevel.CONFIRM,
    }
    defaults.update(overrides)
    return Signal(**defaults)


def _make_redteam(findings: tuple[AttackFinding, ...]) -> RedTeamReport:
    covered = {f.vector for f in findings}
    padded = findings + tuple(
        AttackFinding(vector=v, severity=AttackSeverity.NOT_APPLICABLE, detail="不適用")
        for v in AttackVector
        if v not in covered
    )
    return RedTeamReport(
        findings=padded, confidence=0.81, reasoning_digest="八項攻擊向量已逐一檢視"
    )


def build_demo() -> dict[str, object]:
    card_tsmc = build_signal_card(
        _make_signal(
            TSMC,
            confidence=0.78,
            invalidation_condition="月營收年增率轉為負成長，或法人連續三日賣超逾五千張",
        ),
        _make_redteam(
            (
                AttackFinding(
                    vector=AttackVector.CROWDING,
                    severity=AttackSeverity.MEDIUM,
                    detail="同類訊號近期被多個因子模型同時偵測到，擁擠度偏高",
                ),
            )
        ),
        generated_at=T0,
        entry_range=(985.0, 1005.0),
    )
    card_mtk = build_signal_card(
        _make_signal(
            MEDIATEK,
            direction=Direction.SHORT,
            confidence=0.64,
            horizon=Horizon.DAYS,
            invalidation_condition="客戶端庫存回補力道優於預期，或股價站回月線",
        ),
        _make_redteam(
            (
                AttackFinding(
                    vector=AttackVector.SINGLE_REGIME,
                    severity=AttackSeverity.HIGH,
                    detail="此訊號歷史上僅在降息循環中有效，目前 regime 判讀尚未確認轉向",
                ),
            )
        ),
        generated_at=T0,
        entry_range=(1180.0, 1210.0),
    )

    attribution = AttributionResult(
        total_return=0.0083,
        beta_return=0.0041,
        alpha_return=0.0052,
        style_return=-0.0006,
        cost_drag=-0.0012,
        timing_return=0.0008,
    )

    regime = MarketRegime(
        trend=TrendState.TRENDING_UP,
        volatility=VolatilityState.NORMAL,
        liquidity=LiquidityState.NORMAL,
        cross_sectional_correlation=0.34,
    )

    brief = build_morning_brief(
        trade_date=date(2026, 9, 11),
        yesterday_attribution=attribution,
        regime=regime,
        pending_signals=(card_tsmc, card_mtk),
        risk_snapshot=RiskSnapshot(
            gross_exposure=0.58,
            net_exposure=0.31,
            top5_concentration=0.36,
            limit_usage={"單一標的／2330": 0.76, "產業／半導體": 0.88, "前五大集中度": 0.9},
        ),
        calendar=(
            CalendarEvent(entity_id=TSMC, description="法人說明會", event_date=date(2026, 9, 11)),
            CalendarEvent(
                entity_id=None, description="美國 CPI 公布", event_date=date(2026, 9, 12)
            ),
            CalendarEvent(
                entity_id=DELTA, description="8 月營收公布", event_date=date(2026, 9, 10)
            ),
        ),
    )

    portfolio = PortfolioState(
        weights={TSMC: 0.048, MEDIATEK: -0.022, DELTA: 0.031, LARGAN: 0.018},
        sectors={TSMC: "半導體", MEDIATEK: "半導體", DELTA: "電子零組件", LARGAN: "光學元件"},
        adv_values={
            TSMC: Decimal("8000000000"),
            MEDIATEK: Decimal("3200000000"),
            DELTA: Decimal("900000000"),
            LARGAN: Decimal("400000000"),
        },
        equity=Decimal("50000000"),
    )
    limits = RiskLimits(
        max_position_weight=0.05,
        max_sector_weight=0.25,
        max_gross_exposure=1.0,
        max_net_exposure=0.6,
        drawdown_derisk=0.08,
        drawdown_flatten=0.15,
        adv_participation_cap=0.05,
        top5_concentration_cap=0.4,
    )

    base_equity = 48_500_000
    equity_curve = {}
    for i in range(30):
        d = date(2026, 8, 1) + timedelta(days=i)
        drift = int(base_equity * (1 + 0.0009 * i - (0.0004 * max(0, i - 20))))
        equity_curve[d] = Decimal(drift)

    dashboard = build_dashboard(
        as_of=T0,
        portfolio=portfolio,
        limits=limits,
        trading_mode=TradingMode.NORMAL,
        signal_healths={
            "signal-momentum-tw": SignalHealth.HEALTHY,
            "signal-flow-foreign": SignalHealth.HEALTHY,
            "signal-earnings-surprise": SignalHealth.WATCH,
            "signal-news-guidance": SignalHealth.QUARANTINED,
        },
        equity_curve=equity_curve,
    )

    router = AlertRouter(
        settings=AlertingSettings(
            max_alerts_per_hour=5, novelty_window_hours=24, dedup_similarity_threshold=0.85
        )
    )
    raw_alerts = [
        (
            TSMC,
            Severity.P1,
            "外資近三日轉為賣超",
            "留意是否轉為連續五日以上，屆時觸發訊號重新評估",
            T0,
        ),
        (
            TSMC,
            Severity.P2,
            "外資賣超力道持平",
            "尚未達重新評估門檻，先觀察即可",
            T0 + timedelta(minutes=20),
        ),
        (
            MEDIATEK,
            Severity.P0,
            "客戶端急單消息未經證實，股價已劇烈波動",
            "立即檢視部位並準備人工複核",
            T0 + timedelta(minutes=25),
        ),
        (
            DELTA,
            Severity.P2,
            "8 月營收年增率優於市場預期",
            "符合既有多方訊號，維持現有部位",
            T0 + timedelta(minutes=40),
        ),
        (
            TSMC,
            Severity.P1,
            "外資賣超力道加大",
            "接近失效條件門檻，建議提前準備複核",
            T0 + timedelta(minutes=55),
        ),
        (
            TSMC,
            Severity.P1,
            "外資賣超力道持續加大",
            "已逼近失效條件門檻，建議提前準備複核",
            T0 + timedelta(minutes=75),
        ),
        (
            TSMC,
            Severity.P1,
            "外資賣超尚未減緩",
            "持續觀察中，尚未達重新評估門檻",
            T0 + timedelta(minutes=95),
        ),
        (
            TSMC,
            Severity.P1,
            "外資賣超略為趨緩",
            "風險降低，可維持現有部位",
            T0 + timedelta(minutes=115),
        ),
    ]
    alert_events = []
    for entity, sev, summary, action, ts in raw_alerts:
        result = router.submit(
            Alert(entity_id=entity, severity=sev, summary=summary, action=action, created_at=ts)
        )
        alert_events.append((ts, result))

    book = PaperTradingBook()
    for i in range(5):
        d = date(2026, 9, 1) + timedelta(days=i)
        now = datetime(d.year, d.month, d.day, 1, 0, tzinfo=UTC)
        intent = OrderIntent(
            event_time=now,
            ingest_time=now,
            entity_id=TSMC,
            direction=Direction.LONG,
            target_weight=0.05,
            source_signal_ids=(SignalId(f"sig-{i}"),),
            decision_level=DecisionLevel.CONFIRM,
        )
        verdict = RiskVerdict(
            event_time=now,
            ingest_time=now,
            intent_hash=f"hash-{i}",
            approved=(i != 3),
            breached_limits=() if i != 3 else ("單一標的上限",),
        )
        book.record_order(TSMC, intent, verdict, reference_price=Decimal("995.0"), now=now)
        book.record_agent_cost(d, 42000 + i * 1500)
        book.record_alert_outcome(d, sent=(i % 2 == 0))
        book.record_alert_outcome(d, sent=False)
    weekly = book.weekly_summary(date(2026, 9, 1), date(2026, 9, 5))

    return {
        "generated_at": T0.isoformat(),
        "morning_brief_text": brief.to_display_text(),
        "signal_cards": [
            {
                "entity": str(c.entity_id),
                "direction": c.direction.value,
                "confidence": c.confidence,
                "horizon": c.horizon.value,
                "entry_range": c.entry_range,
                "invalidation": c.invalidation_condition,
                "decision_level": c.decision_level.name,
                "has_blocking": c.has_blocking_counter_argument,
                "counter_arguments": [
                    {"severity": a.severity.value, "text": a.text}
                    for a in c.counter_arguments
                    if a.severity is not AttackSeverity.NOT_APPLICABLE
                ],
                "display": c.to_display_text(),
            }
            for c in (card_tsmc, card_mtk)
        ],
        "attribution": {
            "total_return": attribution.total_return,
            "beta_return": attribution.beta_return,
            "alpha_return": attribution.alpha_return,
            "style_return": attribution.style_return,
            "cost_drag": attribution.cost_drag,
            "timing_return": attribution.timing_return,
        },
        "dashboard": {
            "as_of": dashboard.as_of.isoformat(),
            "trading_mode": dashboard.trading_mode.name,
            "gross_exposure": dashboard.gross_exposure,
            "net_exposure": dashboard.net_exposure,
            "top5_concentration": dashboard.top5_concentration,
            "drawdown_from_peak": dashboard.drawdown_from_peak,
            "sector_weights": dashboard.sector_weights,
            "limit_usages": [
                {
                    "name": u.name,
                    "usage_ratio": u.usage_ratio,
                    "current": u.current,
                    "limit": u.limit,
                }
                for u in dashboard.limit_usages
            ],
            "signal_health": {
                "healthy": dashboard.signal_health.healthy,
                "watch": dashboard.signal_health.watch,
                "quarantined": dashboard.signal_health.quarantined,
            },
            "equity_curve": {k.isoformat(): float(v) for k, v in dashboard.equity_curve.items()},
        },
        "weekly_summary": {
            "start": weekly.start.isoformat(),
            "end": weekly.end.isoformat(),
            "total_orders": weekly.total_orders,
            "rejected_orders": weekly.rejected_orders,
            "total_agent_cost_tokens": weekly.total_agent_cost_tokens,
            "alert_noise_ratio": weekly.alert_noise_ratio,
            "display": weekly.to_display_text(),
        },
        "alerts_timeline": [
            {
                "at": ts.isoformat(),
                "kind": type(result).__name__ if result is not None else "SUPPRESSED",
                "text": result.to_display_text() if result is not None else None,
            }
            for ts, result in alert_events
        ],
    }


if __name__ == "__main__":
    print(json.dumps(build_demo(), ensure_ascii=False, indent=2))
