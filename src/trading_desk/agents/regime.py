"""Regime agent: what kind of market are we in right now.

Deterministic. Most strategies do not fail because the model is wrong; they
fail because the regime changed and the model was never told. Every signal
downstream is scaled by this reading, and in a hostile regime long signals are
demoted rather than silently emitted at full conviction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..blackboard import Blackboard
from ..contracts import RegimeState
from ..statistics import max_drawdown, realised_volatility, volatility_regime_ratio
from .base import Agent, MarketContext

TREND_WINDOW = 200
BREADTH_WINDOW = 50
HIGH_VOL_RATIO = 1.4
LOW_VOL_RATIO = 0.8

# Conviction multipliers by regime. A long idea in a risk-off tape is worth
# materially less than the identical idea in a calm uptrend.
REGIME_MULTIPLIERS = {
    "順風": {"long": 1.10, "short": 0.75},
    "中性": {"long": 1.00, "short": 1.00},
    "震盪": {"long": 0.85, "short": 0.95},
    "逆風": {"long": 0.65, "short": 1.10},
}


class RegimeAgent(Agent):
    name = "regime"
    version = "1.0"
    sources = ("prices",)

    def run(self, context: MarketContext, board: Blackboard) -> int:
        prices = context.visible_prices()
        benchmark = context.benchmark
        if benchmark not in prices.columns:
            board.record_degradation(f"regime: 基準 {benchmark} 缺漏，退回中性判定")
            board.regime = self._neutral(context, "基準資料缺漏")
            return 0

        series = prices[benchmark].ffill().dropna()
        if len(series) < BREADTH_WINDOW + 5:
            board.regime = self._neutral(context, "基準歷史不足")
            return 0

        returns = series.pct_change(fill_method=None)
        trend, trend_gap = self._trend(series)
        volatility, vol_ratio, vol_level = self._volatility(returns)
        breadth = self._breadth(prices)
        appetite, drawdown = self._risk_appetite(series, returns)
        label = self._label(trend, volatility, breadth, appetite)

        detail = (
            f"基準 {benchmark} 距 {TREND_WINDOW} 日均線 {trend_gap:+.1%}；"
            f"年化波動 {vol_level:.1%}（短長比 {vol_ratio:.2f}）；"
            f"{BREADTH_WINDOW} 日均線上方個股佔 {breadth:.0%}；"
            f"近一年最大回撤 {drawdown:.1%}。"
        )

        board.regime = RegimeState(
            label=label,
            trend=trend,
            volatility=volatility,
            breadth=breadth,
            risk_appetite=appetite,
            detail=detail,
            as_of=pd.Timestamp(context.as_of).to_pydatetime(),
        )
        board.market["regime_multipliers"] = REGIME_MULTIPLIERS[label]
        board.market["benchmark_drawdown"] = drawdown
        board.market["benchmark_volatility"] = vol_level
        return 1

    def _trend(self, series: pd.Series) -> tuple[str, float]:
        window = min(TREND_WINDOW, len(series) - 1)
        average = float(series.rolling(window, min_periods=window // 2).mean().iloc[-1])
        if not np.isfinite(average) or average == 0:
            return "未知", float("nan")
        gap = float(series.iloc[-1]) / average - 1.0
        if gap > 0.03:
            return "多頭", gap
        if gap < -0.03:
            return "空頭", gap
        return "盤整", gap

    def _volatility(self, returns: pd.Series) -> tuple[str, float, float]:
        ratio = volatility_regime_ratio(returns)
        level = realised_volatility(returns, 60)
        if not np.isfinite(ratio):
            return "未知", float("nan"), level
        if ratio >= HIGH_VOL_RATIO:
            return "高波動", ratio, level
        if ratio <= LOW_VOL_RATIO:
            return "低波動", ratio, level
        return "常態波動", ratio, level

    def _breadth(self, prices: pd.DataFrame) -> float:
        """Share of the universe trading above its own 50-day average.

        Breadth is what separates a broad advance from an index dragged up by a
        handful of names, and the two call for very different position sizing.
        """
        filled = prices.ffill()
        if len(filled) < BREADTH_WINDOW:
            return float("nan")
        average = filled.rolling(BREADTH_WINDOW, min_periods=BREADTH_WINDOW // 2).mean().iloc[-1]
        latest = filled.iloc[-1]
        comparable = (latest.notna()) & (average.notna())
        if comparable.sum() == 0:
            return float("nan")
        return float((latest[comparable] > average[comparable]).mean())

    def _risk_appetite(self, series: pd.Series, returns: pd.Series) -> tuple[str, float]:
        drawdown = max_drawdown(series.tail(252))
        recent = float(series.pct_change(20, fill_method=None).iloc[-1])
        if drawdown <= -0.15 or recent <= -0.08:
            return "risk-off", drawdown
        if drawdown >= -0.07 and recent >= 0.03:
            return "risk-on", drawdown
        return "中性", drawdown

    def _label(self, trend: str, volatility: str, breadth: float, appetite: str) -> str:
        if appetite == "risk-off" or trend == "空頭":
            return "逆風"
        if volatility == "高波動":
            return "震盪"
        if trend == "多頭" and appetite == "risk-on" and (np.isnan(breadth) or breadth >= 0.5):
            return "順風"
        return "中性"

    def _neutral(self, context: MarketContext, reason: str) -> RegimeState:
        return RegimeState(
            label="中性",
            trend="未知",
            volatility="未知",
            breadth=float("nan"),
            risk_appetite="中性",
            detail=f"降級判定：{reason}",
            as_of=pd.Timestamp(context.as_of).to_pydatetime(),
        )
