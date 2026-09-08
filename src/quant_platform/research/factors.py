"""Factor library. Each factor is a hypothesis stated precisely enough to be wrong.

A factor here is not "momentum works". It is a named, versioned function from a
point-in-time price panel to a cross-sectional score, together with the economic
claim it rests on and the conditions under which that claim should fail. Writing
the failure condition down at definition time is what stops a dead factor from
being quietly re-parameterised until it looks alive again.

All three baselines are deliberately simple. Simple baselines are what a
complicated model has to beat before it earns the right to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from trading_desk.statistics import (
    market_neutral_returns,
    rank_to_normal,
    realised_volatility_frame,
    robust_zscore_frame,
)

MIN_HISTORY = 260


@dataclass(frozen=True)
class Factor:
    """One testable hypothesis."""

    name: str
    version: str
    claim: str
    fails_when: str
    compute: Callable[[pd.DataFrame], pd.Series]
    direction: int = 1
    min_history: int = MIN_HISTORY

    def score(self, visible_prices: pd.DataFrame) -> pd.Series:
        """Cross-sectional score at the last visible date. Higher is more attractive."""
        if len(visible_prices) < self.min_history:
            return pd.Series(dtype="float64")
        usable = visible_prices.loc[:, visible_prices.notna().sum() >= self.min_history]
        if usable.shape[1] < 20:
            return pd.Series(dtype="float64")
        raw = self.compute(usable.ffill())
        return (raw * self.direction).dropna()

    @property
    def id(self) -> str:
        return f"{self.name}_v{self.version}"


def _mean_reversion(prices: pd.DataFrame) -> pd.Series:
    """Short-horizon reversal on the idiosyncratic part of the move.

    The market factor is removed first: a name that fell because everything
    fell has not become cheap relative to its peers, and buying it is a beta
    bet wearing a reversal costume.
    """
    returns = prices.pct_change(fill_method=None)
    residual = market_neutral_returns(returns.tail(120))
    cumulative = residual.tail(5).sum()
    scores = robust_zscore_frame(residual, window=60, min_sigma=0.006).iloc[-1]
    combined = scores.where(scores.notna(), rank_to_normal(cumulative))
    return -rank_to_normal(combined)


def _momentum(prices: pd.DataFrame) -> pd.Series:
    """Twelve-month trend excluding the most recent month.

    The one-month gap is not decoration: recent returns reverse, and a momentum
    factor that includes them is a momentum factor fighting a reversal factor.
    """
    if len(prices) < 260:
        return pd.Series(dtype="float64")
    lagged = prices.shift(21)
    long_return = lagged.pct_change(231, fill_method=None).iloc[-1]
    medium_return = lagged.pct_change(126, fill_method=None).iloc[-1]
    return 0.6 * rank_to_normal(long_return) + 0.4 * rank_to_normal(medium_return)


def _low_volatility(prices: pd.DataFrame) -> pd.Series:
    """Low-risk anomaly: quieter names have historically not been paid less."""
    returns = prices.pct_change(fill_method=None).tail(260)
    volatility = realised_volatility_frame(returns, 120)
    return -rank_to_normal(volatility)


FACTORS: dict[str, Factor] = {
    "mean_reversion": Factor(
        name="mean_reversion",
        version="1",
        claim="扣除市場因子後，短期超跌的個股在數週內傾向回補。",
        fails_when="下跌源自真實資訊而非流動性衝擊時，價格會永久重定價而非回補；"
                   "高波動或趨勢型 regime 下失效最明顯。",
        compute=_mean_reversion,
    ),
    "momentum": Factor(
        name="momentum",
        version="1",
        claim="跳過最近一個月後，過去一年的相對強弱會延續。",
        fails_when="市場轉折點會出現動能崩潰（momentum crash），"
                   "空方部位在反彈時虧損最劇。",
        compute=_momentum,
    ),
    "low_volatility": Factor(
        name="low_volatility",
        version="1",
        claim="低波動個股的風險調整後報酬長期優於高波動個股。",
        fails_when="強勁多頭與高槓桿環境下，高 beta 會持續領先，此因子長期落後。",
        compute=_low_volatility,
    ),
}
