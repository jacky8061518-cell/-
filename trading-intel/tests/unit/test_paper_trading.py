"""紙上交易模式（SPEC 8.1 第 5 點）：完整走完決策路徑但不下單，每日記錄。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import DecisionLevel, Direction, Market
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import SignalId, make_entity_id
from trading_intel.core.types import OrderIntent, RiskVerdict
from trading_intel.surface.paper_trading import PaperOrderStatus, PaperTradingBook

TSMC = make_entity_id(Market.TW, "2330")
UMC = make_entity_id(Market.TW, "2303")
T0 = datetime(2024, 3, 19, 9, 0, tzinfo=UTC)


def intent(**overrides: object) -> OrderIntent:
    defaults: dict[str, object] = {
        "event_time": T0,
        "ingest_time": T0 + timedelta(minutes=1),
        "entity_id": TSMC,
        "direction": Direction.LONG,
        "target_weight": 0.03,
        "source_signal_ids": (SignalId("sig1"),),
        "decision_level": DecisionLevel.CONFIRM,
    }
    defaults.update(overrides)
    return OrderIntent(**defaults)


def approved_verdict() -> RiskVerdict:
    return RiskVerdict(
        event_time=T0,
        ingest_time=T0 + timedelta(minutes=1),
        intent_hash="h1",
        approved=True,
    )


def rejected_verdict() -> RiskVerdict:
    return RiskVerdict(
        event_time=T0,
        ingest_time=T0 + timedelta(minutes=1),
        intent_hash="h2",
        approved=False,
        breached_limits=("MAX_POSITION_WEIGHT",),
    )


# --- 記錄決策，不管通過與否 --------------------------------------------------


def test_approved_order_is_recorded() -> None:
    book = PaperTradingBook()
    order = book.record_order(
        TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0
    )
    assert order.status is PaperOrderStatus.RECORDED


def test_rejected_order_is_also_recorded() -> None:
    """被否決的決策同樣是資料——紙上交易要記錄「完整走過的路徑」，不只是成功的。"""
    book = PaperTradingBook()
    order = book.record_order(
        TSMC, intent(), rejected_verdict(), reference_price=Decimal("100"), now=T0
    )
    assert order.status is PaperOrderStatus.REJECTED
    assert book.logs[T0.date()].rejected_count == 1


def test_no_real_order_is_ever_sent() -> None:
    """PaperOrder 沒有任何送出委託的方法或欄位——這件事只能靠設計保證，
    這裡驗證的是它至少不包含任何看起來像下單確認的欄位。"""
    book = PaperTradingBook()
    order = book.record_order(
        TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0
    )
    assert not hasattr(order, "broker_order_id")
    assert not hasattr(order, "submitted")


def test_orders_accumulate_within_a_day() -> None:
    book = PaperTradingBook()
    book.record_order(TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0)
    book.record_order(
        UMC,
        intent(entity_id=UMC),
        approved_verdict(),
        reference_price=Decimal("50"),
        now=T0 + timedelta(hours=1),
    )
    assert len(book.logs[T0.date()].orders) == 2


def test_orders_on_different_days_are_separated() -> None:
    book = PaperTradingBook()
    book.record_order(TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0)
    next_day = T0 + timedelta(days=1)
    book.record_order(
        TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=next_day
    )
    assert len(book.logs) == 2
    assert len(book.logs[T0.date()].orders) == 1


# --- agent 成本記錄 -----------------------------------------------------------


def test_agent_cost_accumulates() -> None:
    """SPEC：紙上交易要能回答「agent 成本的實際數字」。"""
    book = PaperTradingBook()
    book.record_agent_cost(T0.date(), 1500)
    book.record_agent_cost(T0.date(), 800)
    assert book.logs[T0.date()].agent_cost_tokens == 2300


def test_negative_token_cost_is_rejected() -> None:
    book = PaperTradingBook()
    with pytest.raises(SchemaValidationError, match="不得為負數"):
        book.record_agent_cost(T0.date(), -1)


# --- 告警訊噪比 -----------------------------------------------------------


def test_alert_noise_ratio_tracks_suppressed_fraction() -> None:
    """SPEC：紙上交易要能回答「告警的訊噪比」。"""
    book = PaperTradingBook()
    day = T0.date()
    book.record_alert_outcome(day, sent=True)
    book.record_alert_outcome(day, sent=False)
    book.record_alert_outcome(day, sent=False)
    book.record_alert_outcome(day, sent=False)
    assert book.logs[day].alert_noise_ratio == pytest.approx(0.75)


def test_noise_ratio_of_no_alerts_is_zero() -> None:
    book = PaperTradingBook()
    book.record_agent_cost(T0.date(), 100)  # 觸發建立當天紀錄，但無告警
    assert book.logs[T0.date()].alert_noise_ratio == 0.0


# --- 每週摘要 -----------------------------------------------------------------


def test_weekly_summary_aggregates_across_days() -> None:
    """SPEC：紙上交易至少三個月，期間每週檢視這三項。"""
    book = PaperTradingBook()
    monday = date(2024, 3, 18)
    for offset in range(5):
        day = monday + timedelta(days=offset)
        stamp = datetime.combine(day, T0.timetz())
        book.record_order(
            TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=stamp
        )
        book.record_agent_cost(day, 1000)
        book.record_alert_outcome(day, sent=True)
        book.record_alert_outcome(day, sent=False)

    summary = book.weekly_summary(monday, monday + timedelta(days=6))
    assert summary.total_orders == 5
    assert summary.total_agent_cost_tokens == 5000
    assert summary.alert_noise_ratio == pytest.approx(0.5)


def test_weekly_summary_excludes_days_outside_range() -> None:
    book = PaperTradingBook()
    book.record_order(TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0)
    outside = T0 + timedelta(days=30)
    book.record_order(
        TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=outside
    )

    summary = book.weekly_summary(T0.date(), T0.date() + timedelta(days=6))
    assert summary.total_orders == 1


def test_weekly_summary_counts_rejected_orders_separately() -> None:
    book = PaperTradingBook()
    book.record_order(TSMC, intent(), approved_verdict(), reference_price=Decimal("100"), now=T0)
    book.record_order(TSMC, intent(), rejected_verdict(), reference_price=Decimal("100"), now=T0)

    summary = book.weekly_summary(T0.date(), T0.date())
    assert summary.total_orders == 2
    assert summary.rejected_orders == 1


def test_summary_display_text() -> None:
    book = PaperTradingBook()
    book.record_agent_cost(T0.date(), 5000)
    summary = book.weekly_summary(T0.date(), T0.date())
    text = summary.to_display_text()
    assert "5,000" in text
    assert "紙上交易週報" in text
