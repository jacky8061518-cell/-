"""紙上交易模式（SPEC 8.1 第 5 點）。

「完整走完決策路徑但不下單，每日記錄」——這句話的關鍵是**完整**。
紙上交易不是回測的另一個名字：回測讀的是歷史資料、跑在模擬時鐘上；
紙上交易讀的是即時資料、跑在真實時鐘上，唯一被拿掉的只有「真的送出委託」
這一步。中間的訊號生成、風控檢查、組合建構全部照走，這樣累積下來的
紀錄才能拿去回答 SPEC 提出的三個問題：live 訊號與回測訊號的差異來源、
告警的訊噪比、agent 成本的實際數字。

**因此這裡刻意不接 core.sandbox.backtest_mode。** 那個 context manager
是為回測設計的：凍結時鐘、切斷網路。紙上交易兩者都要反過來——用真實時鐘、
可以連網——只在「產生委託之後」這一步驟停手，改記錄不送出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from trading_intel.core.clock import ensure_utc
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import EntityId
from trading_intel.core.types import OrderIntent, RiskVerdict


class PaperOrderStatus(StrEnum):
    """紙上單的狀態。沒有 FILLED 之外的部分成交等真實成交狀態——
    紙上交易假設全額即時成交於決策當下的參考價，因為重點是驗證決策路徑，
    不是模擬成交機制（那是 backtest 引擎的職責，見 backtest/engine.py）。
    """

    RECORDED = "RECORDED"
    #: 風控否決，不成立紙上單。
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class PaperOrder:
    """一筆紙上交易紀錄。"""

    entity_id: EntityId
    intent: OrderIntent
    verdict: RiskVerdict
    status: PaperOrderStatus
    reference_price: Decimal
    recorded_at: datetime

    @property
    def notional(self) -> Decimal:
        return abs(Decimal(str(self.intent.target_weight))) * self.reference_price


@dataclass(frozen=True)
class DailyPaperLog:
    """一天的紙上交易紀錄，供每週檢視使用。"""

    trade_date: date
    orders: tuple[PaperOrder, ...]
    #: agent 呼叫的實際成本（token），累計到當天。SPEC：「agent 成本的實際數字」。
    agent_cost_tokens: int
    #: 當天送出的告警則數，供訊噪比分析。
    alerts_sent: int
    alerts_suppressed: int

    @property
    def rejected_count(self) -> int:
        return sum(1 for order in self.orders if order.status is PaperOrderStatus.REJECTED)

    @property
    def alert_noise_ratio(self) -> float:
        """告警的訊噪比：被抑制的比例。過高代表告警設計得太吵。"""
        total = self.alerts_sent + self.alerts_suppressed
        if total == 0:
            return 0.0
        return self.alerts_suppressed / total


@dataclass
class PaperTradingBook:
    """紙上交易的累積帳本。

    刻意不做任何「事後才知道」的判斷（例如即時算報酬）——那需要之後的
    真實價格才能算，屬於盤後歸因的工作（見 surface/attribution.py）。
    這裡只負責忠實記錄「當時決定了什麼」。
    """

    logs: dict[date, DailyPaperLog] = field(default_factory=dict)

    def record_order(
        self,
        entity_id: EntityId,
        intent: OrderIntent,
        verdict: RiskVerdict,
        *,
        reference_price: Decimal,
        now: datetime,
    ) -> PaperOrder:
        """記錄一筆決策——不管風控通過與否都要記，被否決的決策同樣是資料。"""
        stamp = ensure_utc(now)
        status = PaperOrderStatus.RECORDED if verdict.approved else PaperOrderStatus.REJECTED
        order = PaperOrder(
            entity_id=entity_id,
            intent=intent,
            verdict=verdict,
            status=status,
            reference_price=reference_price,
            recorded_at=stamp,
        )
        trade_date = stamp.date()
        existing = self.logs.get(trade_date)
        if existing is None:
            self.logs[trade_date] = DailyPaperLog(
                trade_date=trade_date,
                orders=(order,),
                agent_cost_tokens=0,
                alerts_sent=0,
                alerts_suppressed=0,
            )
        else:
            self.logs[trade_date] = _replace_orders(existing, (*existing.orders, order))
        return order

    def record_agent_cost(self, trade_date: date, tokens: int) -> None:
        if tokens < 0:
            raise SchemaValidationError("token 成本不得為負數", tokens=tokens)
        self._ensure_day(trade_date)
        existing = self.logs[trade_date]
        self.logs[trade_date] = _replace(
            existing, agent_cost_tokens=existing.agent_cost_tokens + tokens
        )

    def record_alert_outcome(self, trade_date: date, *, sent: bool) -> None:
        self._ensure_day(trade_date)
        existing = self.logs[trade_date]
        if sent:
            self.logs[trade_date] = _replace(existing, alerts_sent=existing.alerts_sent + 1)
        else:
            self.logs[trade_date] = _replace(
                existing, alerts_suppressed=existing.alerts_suppressed + 1
            )

    def _ensure_day(self, trade_date: date) -> None:
        if trade_date not in self.logs:
            self.logs[trade_date] = DailyPaperLog(
                trade_date=trade_date,
                orders=(),
                agent_cost_tokens=0,
                alerts_sent=0,
                alerts_suppressed=0,
            )

    def weekly_summary(self, start: date, end: date) -> WeeklySummary:
        """每週檢視所需的三項統計（SPEC 8.1）。"""
        in_range = [log for day, log in self.logs.items() if start <= day <= end]
        total_orders = sum(len(log.orders) for log in in_range)
        total_rejected = sum(log.rejected_count for log in in_range)
        total_tokens = sum(log.agent_cost_tokens for log in in_range)
        total_sent = sum(log.alerts_sent for log in in_range)
        total_suppressed = sum(log.alerts_suppressed for log in in_range)
        total_alerts = total_sent + total_suppressed

        return WeeklySummary(
            start=start,
            end=end,
            total_orders=total_orders,
            rejected_orders=total_rejected,
            total_agent_cost_tokens=total_tokens,
            alert_noise_ratio=(total_suppressed / total_alerts if total_alerts > 0 else 0.0),
        )


@dataclass(frozen=True)
class WeeklySummary:
    """每週檢視摘要，對應 SPEC 8.1「紙上交易至少執行三個月，期間每週檢視」的三項。"""

    start: date
    end: date
    total_orders: int
    rejected_orders: int
    total_agent_cost_tokens: int
    alert_noise_ratio: float

    def to_display_text(self) -> str:
        return (
            f"═══ 紙上交易週報　{self.start.isoformat()} ~ {self.end.isoformat()} ═══\n"
            f"總決策數 {self.total_orders}（風控否決 {self.rejected_orders}）\n"
            f"Agent 成本：{self.total_agent_cost_tokens:,} tokens\n"
            f"告警訊噪比：{self.alert_noise_ratio:.0%}（被抑制的比例）"
        )


def _replace_orders(log: DailyPaperLog, orders: tuple[PaperOrder, ...]) -> DailyPaperLog:
    return DailyPaperLog(
        trade_date=log.trade_date,
        orders=orders,
        agent_cost_tokens=log.agent_cost_tokens,
        alerts_sent=log.alerts_sent,
        alerts_suppressed=log.alerts_suppressed,
    )


def _replace(
    log: DailyPaperLog,
    *,
    agent_cost_tokens: int | None = None,
    alerts_sent: int | None = None,
    alerts_suppressed: int | None = None,
) -> DailyPaperLog:
    return DailyPaperLog(
        trade_date=log.trade_date,
        orders=log.orders,
        agent_cost_tokens=agent_cost_tokens
        if agent_cost_tokens is not None
        else log.agent_cost_tokens,
        alerts_sent=alerts_sent if alerts_sent is not None else log.alerts_sent,
        alerts_suppressed=(
            alerts_suppressed if alerts_suppressed is not None else log.alerts_suppressed
        ),
    )
