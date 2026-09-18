"""Risk engine. The gate every signal must pass, and the only place sizing happens.

The engine is deliberately separate from the agents and from the decision layer.
Agents propose; this module decides how much, or whether at all. Nothing
upstream can widen a limit, and every rejection records which constraint bound,
so a trader can always answer "why was this only 1% instead of 3%".

Sizing follows risk budgets rather than share counts: a 10% position in a
placid utility and a 10% position in a small-cap semiconductor are not the same
bet, and only volatility targeting makes the two comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .contracts import Position, RiskBudget
from .statistics import correlation_clusters, effective_positions


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits. Deployed and reviewed separately from any strategy code."""

    target_portfolio_vol: float = 0.12
    max_weight_per_name: float = 0.05
    max_weight_per_cluster: float = 0.20
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.80
    max_positions: int = 20
    min_weight: float = 0.005
    max_loss_per_trade_nav: float = 0.01
    kelly_fraction: float = 0.25
    stop_atr_multiple: float = 2.0
    reward_risk_floor: float = 1.5
    min_conviction: float = 0.55

    def __post_init__(self) -> None:
        if not 0 < self.kelly_fraction <= 0.5:
            raise ValueError("fractional Kelly above 0.5 is not survivable under estimation error")
        if self.max_weight_per_name > self.max_weight_per_cluster:
            raise ValueError("a single name cannot be allowed more than its cluster")


@dataclass(frozen=True)
class BreakerStatus:
    """Result of evaluating the account-level circuit breakers."""

    tripped: bool
    level: str
    reason: str
    action: str
    recovery: str

    @property
    def blocks_new_entries(self) -> bool:
        return self.tripped


@dataclass
class RiskEngine:
    """Stateless with respect to signals, stateful with respect to the book."""

    limits: RiskLimits = field(default_factory=RiskLimits)

    # --- Position sizing --------------------------------------------------

    def size(
        self,
        *,
        conviction: float,
        annual_volatility: float,
        last_price: float,
        direction: str,
        atr: float | None = None,
        regime_multiplier: float = 1.0,
        breaker: BreakerStatus | None = None,
    ) -> RiskBudget:
        """Turn a confidence score into an authorised weight, stop and target."""
        if breaker is not None and breaker.blocks_new_entries:
            return self._rejected(f"熔斷中：{breaker.reason}")
        if conviction < self.limits.min_conviction:
            return self._rejected(
                f"信心 {conviction:.2f} 低於下限 {self.limits.min_conviction:.2f}"
            )
        if not np.isfinite(annual_volatility) or annual_volatility <= 0:
            return self._rejected("波動率無法估計，拒絕定量")
        if not np.isfinite(last_price) or last_price <= 0:
            return self._rejected("價格資料無效")

        constraints: dict[str, float] = {}

        # Volatility targeting makes weights comparable across names. The budget
        # is per position, not per portfolio: sizing every name to the full
        # portfolio target would leave the book several times over-risked once
        # twenty of them are on. Dividing by sqrt(max_positions) is the
        # low-correlation approximation of each name's risk contribution.
        vol_budget = self.limits.target_portfolio_vol / np.sqrt(self.limits.max_positions)
        constraints["波動率目標"] = vol_budget / annual_volatility

        # Confidence scaling. Below 0.5 the meta-model says do not take it, so
        # the ramp starts there rather than at zero.
        confidence_scale = max(0.0, 2.0 * conviction - 1.0)
        if confidence_scale <= 0:
            return self._rejected("信心不足以形成部位")
        constraints["信心縮放"] = constraints["波動率目標"] * confidence_scale

        # Fractional Kelly. Full Kelly assumes the true edge is known; it is not.
        edge = 2.0 * conviction - 1.0
        kelly = edge / max(annual_volatility**2, 1e-6)
        constraints["分數凱利"] = self.limits.kelly_fraction * kelly

        constraints["單一標的上限"] = self.limits.max_weight_per_name

        weight = min(constraints.values()) * regime_multiplier
        binding = min(constraints, key=lambda name: constraints[name])

        stop_distance = self._stop_distance(atr, annual_volatility, last_price)
        stop_level = self._stop_level(last_price, stop_distance, direction)

        # A stop the position cannot afford means the position is too large,
        # not that the stop should be widened.
        loss_cap_weight = self.limits.max_loss_per_trade_nav / max(stop_distance, 1e-6)
        if loss_cap_weight < weight:
            weight = loss_cap_weight
            binding = "單筆最大虧損"

        if weight < self.limits.min_weight:
            return self._rejected(f"計算權重 {weight:.2%} 低於最小交易門檻，成本不划算")

        target_distance = stop_distance * self.limits.reward_risk_floor
        target_level = (
            last_price * (1 + target_distance)
            if direction == "long"
            else last_price * (1 - target_distance)
        )

        return RiskBudget(
            suggested_weight_pct=round(float(weight) * 100, 2),
            stop_level=round(float(stop_level), 2),
            stop_basis=(
                f"{self.limits.stop_atr_multiple:g}×ATR"
                if atr and np.isfinite(atr)
                else "2×日波動（ATR 不可得）"
            ),
            max_loss_pct_nav=round(float(weight * stop_distance) * 100, 3),
            target_level=round(float(target_level), 2),
            reward_risk=self.limits.reward_risk_floor,
            binding_constraint=binding,
        )

    def _stop_distance(self, atr: float | None, annual_volatility: float, price: float) -> float:
        """Fractional distance to the stop, floored so it cannot be trivially tight."""
        if atr is not None and np.isfinite(atr) and atr > 0 and price > 0:
            distance = self.limits.stop_atr_multiple * atr / price
        else:
            daily_vol = annual_volatility / np.sqrt(252)
            distance = self.limits.stop_atr_multiple * daily_vol
        return float(np.clip(distance, 0.02, 0.35))

    def _stop_level(self, price: float, distance: float, direction: str) -> float:
        return price * (1 - distance) if direction == "long" else price * (1 + distance)

    def _rejected(self, reason: str) -> RiskBudget:
        return RiskBudget(
            suggested_weight_pct=0.0,
            stop_level=None,
            stop_basis="—",
            max_loss_pct_nav=0.0,
            target_level=None,
            reward_risk=None,
            binding_constraint=reason,
        )

    # --- Portfolio-level checks -------------------------------------------

    def apply_portfolio_limits(
        self,
        proposals: list[tuple[str, str, RiskBudget]],
        clusters: dict[str, str],
        existing: list[Position] | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Trim proposed weights until every portfolio-level limit holds.

        Limits bind on correlation clusters, not tickers: twenty names that fall
        together are one bet, however the position list is written.
        """
        existing = existing or []
        approved: dict[str, float] = {}
        notes: list[str] = []

        cluster_used: dict[str, float] = {}
        for position in existing:
            cluster_used[position.cluster] = (
                cluster_used.get(position.cluster, 0.0) + abs(position.weight_pct)
            )
        gross = sum(abs(position.weight_pct) for position in existing)
        count = len(existing)

        ordered = sorted(proposals, key=lambda item: -item[2].suggested_weight_pct)
        for symbol, _direction, budget in ordered:
            weight = budget.suggested_weight_pct
            if weight <= 0:
                continue
            if count >= self.limits.max_positions:
                notes.append(f"{symbol}：已達最大持倉數 {self.limits.max_positions}，未納入")
                continue

            cluster = clusters.get(symbol, symbol)
            cluster_room = self.limits.max_weight_per_cluster * 100 - cluster_used.get(cluster, 0.0)
            gross_room = self.limits.max_gross_exposure * 100 - gross

            allowed = min(weight, cluster_room, gross_room)
            if allowed < self.limits.min_weight * 100:
                reason = "相關性叢集額度已滿" if cluster_room <= gross_room else "總曝險額度已滿"
                notes.append(f"{symbol}：{reason}，未納入")
                continue
            if allowed < weight:
                notes.append(
                    f"{symbol}：{weight:.2f}% 縮減為 {allowed:.2f}%（叢集 {cluster} 額度限制）"
                )

            approved[symbol] = round(allowed, 2)
            cluster_used[cluster] = cluster_used.get(cluster, 0.0) + allowed
            gross += allowed
            count += 1

        return approved, notes

    # --- Account-level breakers -------------------------------------------

    def evaluate_breakers(
        self,
        *,
        daily_pnl_pct: float,
        rolling_5d_pnl_pct: float,
        drawdown_pct: float,
        data_delay_seconds: float = 0.0,
        reject_rate: float = 0.0,
    ) -> BreakerStatus:
        """Check the account-level halts, most severe first.

        Recovery is deliberately harder than tripping: technical conditions
        clear themselves, but anything driven by losses needs a human.
        """
        if drawdown_pct <= -20:
            return BreakerStatus(
                True, "P0", f"回撤 {drawdown_pct:.1f}% 超過 20%",
                "全平倉並停止系統", "人工重新驗證全流程",
            )
        if rolling_5d_pnl_pct <= -6:
            return BreakerStatus(
                True, "P0", f"滾動 5 日虧損 {rolling_5d_pnl_pct:.1f}%",
                "全系統暫停", "人工覆盤後恢復",
            )
        if daily_pnl_pct <= -4:
            return BreakerStatus(
                True, "P0", f"當日虧損 {daily_pnl_pct:.1f}% 超過 4%",
                "平倉至 50% 並停止交易", "人工核准",
            )
        if reject_rate > 0.20:
            return BreakerStatus(
                True, "P0", f"券商拒單率 {reject_rate:.0%}",
                "停止下單", "人工排除後恢復",
            )
        if data_delay_seconds > 3:
            return BreakerStatus(
                True, "P1", f"行情延遲 {data_delay_seconds:.1f} 秒",
                "只出不進", "延遲恢復後自動",
            )
        if daily_pnl_pct <= -2:
            return BreakerStatus(
                True, "P1", f"當日虧損 {daily_pnl_pct:.1f}% 超過 2%",
                "停止新進場，既有部位保留", "隔日自動",
            )
        if drawdown_pct <= -10:
            return BreakerStatus(
                True, "P1", f"回撤 {drawdown_pct:.1f}% 超過 10%",
                "部位規模全面 ×0.5", "回撤修復至 5% 內自動恢復",
            )
        return BreakerStatus(False, "OK", "所有帳戶層限制均在範圍內", "正常運作", "—")


def build_clusters(returns: pd.DataFrame, threshold: float = 0.7) -> dict[str, str]:
    """Map each symbol to its correlation cluster label."""
    groups = correlation_clusters(returns, threshold=threshold)
    mapping: dict[str, str] = {}
    for leader, members in groups.items():
        for member in members:
            mapping[member] = leader
    return mapping


def portfolio_risk_summary(
    positions: list[Position],
    last_prices: pd.Series,
    clusters: dict[str, str] | None = None,
) -> dict[str, object]:
    """Headline risk numbers for the trader dashboard."""
    if not positions:
        return {
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "effective_positions": 0.0,
            "cluster_exposure": {},
            "unrealised_pct": 0.0,
            "positions": 0,
        }
    clusters = clusters or {}
    weights = pd.Series(
        {position.symbol: position.weight_pct for position in positions}, dtype="float64"
    )
    signed = pd.Series(
        {
            position.symbol: position.weight_pct
            * (1 if position.direction == "long" else -1)
            for position in positions
        },
        dtype="float64",
    )
    cluster_exposure: dict[str, float] = {}
    for position in positions:
        label = clusters.get(position.symbol, position.cluster)
        cluster_exposure[label] = cluster_exposure.get(label, 0.0) + abs(position.weight_pct)

    unrealised = 0.0
    for position in positions:
        price = float(last_prices.get(position.symbol, np.nan))
        if np.isfinite(price):
            unrealised += position.unrealised_pct(price) * position.weight_pct / 100.0

    return {
        "gross_exposure": float(weights.abs().sum()),
        "net_exposure": float(signed.sum()),
        "effective_positions": round(effective_positions(weights), 2),
        "cluster_exposure": dict(
            sorted(cluster_exposure.items(), key=lambda item: -item[1])
        ),
        "unrealised_pct": round(unrealised * 100, 3),
        "positions": len(positions),
    }
