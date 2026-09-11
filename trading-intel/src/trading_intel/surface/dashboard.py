"""風險儀表板（SPEC 8.1）。

即時曝險、因子暴露、回撤路徑、限額使用率——四項都是把 risk 層已經算好的
數字組成一個可讀的快照，本模組不重新計算任何統計量，理由與晨報模組相同。

儀表板的核心設計是**限額使用率而非限額本身**：交易員需要知道的是
「離觸發還有多遠」，不是再看一次限額數字是多少（那在 configs 裡）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from trading_intel.core.settings import RiskLimits
from trading_intel.risk.model_risk import SignalHealth
from trading_intel.risk.monitoring import TradingMode
from trading_intel.risk.pretrade import PortfolioState


@dataclass(frozen=True)
class LimitUsage:
    """單一限額的使用率。1.0 代表剛好用滿，>1.0 代表已違規。"""

    name: str
    current: float
    limit: float

    @property
    def usage_ratio(self) -> float:
        if self.limit == 0:
            return 0.0
        return self.current / self.limit

    @property
    def is_near_limit(self) -> bool:
        """使用率超過八成視為警戒，讓交易員在觸發前就看到。"""
        return self.usage_ratio >= 0.8


@dataclass(frozen=True)
class SignalHealthSummary:
    """訊號健康度統計，供儀表板顯示分布而非逐一列出。"""

    healthy: int
    watch: int
    quarantined: int

    @property
    def total(self) -> int:
        return self.healthy + self.watch + self.quarantined


@dataclass(frozen=True)
class RiskDashboard:
    """完整的風險儀表板快照。"""

    as_of: datetime
    gross_exposure: float
    net_exposure: float
    top5_concentration: float
    sector_weights: dict[str, float]
    limit_usages: tuple[LimitUsage, ...]
    drawdown_from_peak: float
    trading_mode: TradingMode
    signal_health: SignalHealthSummary
    equity_curve: dict[date, Decimal]

    @property
    def breached_limits(self) -> tuple[LimitUsage, ...]:
        return tuple(item for item in self.limit_usages if item.usage_ratio >= 1.0)

    @property
    def near_limit(self) -> tuple[LimitUsage, ...]:
        return tuple(
            item for item in self.limit_usages if item.is_near_limit and item.usage_ratio < 1.0
        )

    def to_display_text(self) -> str:
        lines = [
            f"═══ 風險儀表板　{self.as_of.isoformat()} ═══",
            f"交易模式：{self.trading_mode.name}",
            f"總曝險 {self.gross_exposure:.1%}　淨曝險 {self.net_exposure:.1%}"
            f"　前五大集中度 {self.top5_concentration:.1%}",
            f"回撤（自高點）{self.drawdown_from_peak:.2%}",
            "",
            "【限額使用率】",
        ]
        for item in sorted(self.limit_usages, key=lambda x: -x.usage_ratio):
            marker = "🔴" if item.usage_ratio >= 1.0 else "🟡" if item.is_near_limit else "🟢"
            lines.append(f"  {marker} {item.name}：{item.usage_ratio:.0%}")
        lines.append("")
        lines.append(
            f"【訊號健康度】健康 {self.signal_health.healthy}"
            f"　觀察中 {self.signal_health.watch}"
            f"　已隔離 {self.signal_health.quarantined}"
        )
        if self.sector_weights:
            lines.append("")
            lines.append("【產業曝險】")
            for sector, weight in sorted(self.sector_weights.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {sector}：{weight:+.1%}")
        return "\n".join(lines)


def build_dashboard(
    *,
    as_of: datetime,
    portfolio: PortfolioState,
    limits: RiskLimits,
    trading_mode: TradingMode,
    signal_healths: dict[str, SignalHealth],
    equity_curve: dict[date, Decimal],
) -> RiskDashboard:
    """組裝風險儀表板。純函式，輸入已算好的狀態，輸出格式化快照。"""
    limit_usages = (
        LimitUsage("總曝險", portfolio.gross_exposure, limits.max_gross_exposure),
        LimitUsage("淨曝險", abs(portfolio.net_exposure), limits.max_net_exposure),
        LimitUsage("前五大集中度", portfolio.top_n_concentration(5), limits.top5_concentration_cap),
        *(
            LimitUsage(f"產業／{sector}", abs(weight), limits.max_sector_weight)
            for sector, weight in portfolio.sector_weights().items()
        ),
        *(
            LimitUsage(f"單一標的／{entity_id}", abs(weight), limits.max_position_weight)
            for entity_id, weight in portfolio.weights.items()
        ),
    )

    drawdown = 0.0
    if equity_curve:
        ordered = [equity_curve[key] for key in sorted(equity_curve)]
        peak = max(ordered)
        if peak > 0:
            drawdown = float((peak - ordered[-1]) / peak)

    health_counts = {SignalHealth.HEALTHY: 0, SignalHealth.WATCH: 0, SignalHealth.QUARANTINED: 0}
    for health in signal_healths.values():
        health_counts[health] += 1

    return RiskDashboard(
        as_of=as_of,
        gross_exposure=portfolio.gross_exposure,
        net_exposure=portfolio.net_exposure,
        top5_concentration=portfolio.top_n_concentration(5),
        sector_weights=portfolio.sector_weights(),
        limit_usages=limit_usages,
        drawdown_from_peak=drawdown,
        trading_mode=trading_mode,
        signal_health=SignalHealthSummary(
            healthy=health_counts[SignalHealth.HEALTHY],
            watch=health_counts[SignalHealth.WATCH],
            quarantined=health_counts[SignalHealth.QUARANTINED],
        ),
        equity_curve=dict(equity_curve),
    )
