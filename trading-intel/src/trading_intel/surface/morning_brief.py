"""盤前晨報（SPEC 8.1）。

內容清單直接來自 SPEC：昨日歸因、今日 regime 判讀、待觸發訊號清單、
風險曝險快照、行事曆。本模組只負責**組裝**——把已經算好的東西排版成
一份晨報，不在這裡重新計算任何統計量（那些是 backtest／models／risk
層的職責）。這個分工本身就是 SPEC 1「LLM 不碰數字」精神的延伸：
不只是 LLM 不該算數字，連組裝報告的程式碼也不該重算數字，一律引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from trading_intel.core.ids import EntityId
from trading_intel.models.regime import MarketRegime
from trading_intel.surface.attribution import AttributionResult
from trading_intel.surface.signal_card import SignalCard


@dataclass(frozen=True)
class CalendarEvent:
    """行事曆項目：財報、法說、總經數據。"""

    entity_id: EntityId | None
    description: str
    event_date: date


@dataclass(frozen=True)
class RiskSnapshot:
    """風險曝險快照。數字皆由 risk 層算好傳入，本模組不重算。"""

    gross_exposure: float
    net_exposure: float
    top5_concentration: float
    limit_usage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MorningBrief:
    """一份完整的盤前晨報。"""

    trade_date: date
    yesterday_attribution: AttributionResult | None
    regime: MarketRegime
    pending_signals: tuple[SignalCard, ...]
    risk_snapshot: RiskSnapshot
    calendar: tuple[CalendarEvent, ...]

    def to_display_text(self) -> str:
        lines = [f"═══ 盤前晨報　{self.trade_date.isoformat()} ═══", ""]

        lines.append("【昨日歸因】")
        if self.yesterday_attribution is not None:
            a = self.yesterday_attribution
            lines.append(
                f"總報酬 {a.total_return:+.2%}"
                f"（阿爾法 {a.alpha_return:+.2%}　成本 {a.cost_drag:+.2%}）"
            )
        else:
            lines.append("（無資料，可能是首個交易日）")
        lines.append("")

        lines.append("【今日 regime 判讀】")
        lines.append(f"{self.regime.label}" + ("　⚠ risk-off" if self.regime.is_risk_off else ""))
        lines.append("")

        lines.append(f"【待觸發訊號】共 {len(self.pending_signals)} 檔")
        for card in self.pending_signals:
            flag = "　⚠ 有反方高風險意見" if card.has_blocking_counter_argument else ""
            lines.append(
                f"  {card.entity_id}　{card.direction.value}　信心 {card.confidence:.0%}{flag}"
            )
        lines.append("")

        lines.append("【風險曝險快照】")
        r = self.risk_snapshot
        lines.append(
            f"總曝險 {r.gross_exposure:.1%}　淨曝險 {r.net_exposure:.1%}"
            f"　前五大集中度 {r.top5_concentration:.1%}"
        )
        if r.limit_usage:
            for name, usage in sorted(r.limit_usage.items(), key=lambda item: -item[1]):
                lines.append(f"  {name}：使用率 {usage:.0%}")
        lines.append("")

        lines.append(f"【今日行事曆】共 {len(self.calendar)} 項")
        for event in sorted(self.calendar, key=lambda item: item.event_date):
            target = f"（{event.entity_id}）" if event.entity_id else ""
            lines.append(f"  {event.event_date.isoformat()}{target}　{event.description}")

        return "\n".join(lines)


def build_morning_brief(
    *,
    trade_date: date,
    yesterday_attribution: AttributionResult | None,
    regime: MarketRegime,
    pending_signals: tuple[SignalCard, ...],
    risk_snapshot: RiskSnapshot,
    calendar: tuple[CalendarEvent, ...],
) -> MorningBrief:
    """組裝晨報。純函式，方便測試——輸入什麼就組出什麼，沒有隱藏狀態。"""
    return MorningBrief(
        trade_date=trade_date,
        yesterday_attribution=yesterday_attribution,
        regime=regime,
        pending_signals=pending_signals,
        risk_snapshot=risk_snapshot,
        calendar=calendar,
    )
