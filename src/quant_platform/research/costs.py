"""Transaction cost model for Taiwan-listed equities.

Most backtests die here. A strategy with a 0.3% average edge and a 0.47%
round-trip cost is not a marginal strategy, it is a losing one, and the gap
between those two numbers is invisible unless the costs are modelled with the
market's actual fee schedule rather than a round "10 bps" placeholder.

Four components, each separately adjustable so a stress test can triple one
without distorting the others:

* **Commission** — 0.1425% per side by statute, discounted by the broker.
  A minimum ticket charge applies, which makes small orders disproportionately
  expensive and quietly rules out a whole class of high-turnover strategies.
* **Transaction tax** — levied on the sell side only: 0.3% for shares, 0.1%
  for ETFs, 0.15% for same-day round trips.
* **Spread** — derived from the exchange's price-banded tick sizes rather than
  assumed, since a NT$18 stock and a NT$1,200 stock have very different
  relative spreads.
* **Market impact** — square-root law in participation rate.

The honest limitation: the bundled database has no volume, so participation
cannot be measured and is supplied as an assumption. Every result that depends
on impact must therefore be read together with the cost stress test, which is
why :func:`stressed` exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# Taiwan Stock Exchange tick sizes by price band. Real numbers, not a guess:
# they set the minimum spread a taker can possibly pay.
TICK_BANDS = (
    (10.0, 0.01),
    (50.0, 0.05),
    (100.0, 0.10),
    (500.0, 0.50),
    (1000.0, 1.00),
    (float("inf"), 5.00),
)

STATUTORY_COMMISSION = 0.001425
TAX_SHARES = 0.003
TAX_ETF = 0.001
TAX_DAY_TRADE = 0.0015


def tick_size(price: float) -> float:
    """Minimum price increment for a given price level."""
    for ceiling, tick in TICK_BANDS:
        if price < ceiling:
            return tick
    return TICK_BANDS[-1][1]


def tick_size_array(prices: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype="float64")
    ticks = np.full(values.shape, TICK_BANDS[-1][1])
    previous = 0.0
    for ceiling, tick in TICK_BANDS:
        mask = (values >= previous) & (values < ceiling)
        ticks[mask] = tick
        previous = ceiling
    return ticks


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost assumptions, all expressed as fractions of notional."""

    commission_discount: float = 0.60
    min_commission_twd: float = 20.0
    tax_rate: float = TAX_SHARES
    spread_capture: float = 0.5
    extra_slippage_bps: float = 3.0
    impact_coefficient: float = 0.4
    participation: float = 0.01
    label: str = "base"

    def __post_init__(self) -> None:
        if not 0 < self.commission_discount <= 1:
            raise ValueError("手續費折扣必須介於 0 與 1 之間")
        if self.participation <= 0:
            raise ValueError("參與率必須為正；下不了單的策略不需要成本模型")

    @property
    def commission_rate(self) -> float:
        return STATUTORY_COMMISSION * self.commission_discount

    def spread_cost(self, price: float | np.ndarray) -> np.ndarray:
        """Half-spread as a fraction of price, floored at one tick.

        Crossing the spread costs at least half a tick; on a NT$15 stock that
        is 17 bps before any other cost, which is the whole reason cheap and
        expensive names are not interchangeable in a signal.
        """
        prices = np.asarray(price, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_tick = np.where(prices > 0, tick_size_array(prices) / prices, np.nan)
        return relative_tick * self.spread_capture

    def impact_cost(self, volatility: float | np.ndarray) -> np.ndarray:
        """Square-root market impact, scaled by the name's own volatility.

        impact = k · σ_daily · sqrt(participation). Trading 1% of a day's
        volume in a 3%-daily-vol name costs roughly 12 bps at k = 0.4.
        """
        daily_vol = np.asarray(volatility, dtype="float64") / np.sqrt(252.0)
        return self.impact_coefficient * daily_vol * np.sqrt(self.participation)

    def one_way(
        self,
        price: float | np.ndarray,
        volatility: float | np.ndarray,
        is_sell: bool = False,
        is_etf: bool = False,
    ) -> np.ndarray:
        """Total one-way cost as a fraction of notional."""
        cost = np.full(np.shape(np.asarray(price, dtype="float64")), self.commission_rate)
        cost = cost + self.spread_cost(price)
        cost = cost + self.extra_slippage_bps / 10_000.0
        cost = cost + self.impact_cost(volatility)
        if is_sell:
            cost = cost + (TAX_ETF if is_etf else self.tax_rate)
        return cost

    def round_trip(
        self,
        price: float | np.ndarray,
        volatility: float | np.ndarray,
        is_etf: bool = False,
    ) -> np.ndarray:
        """Buy plus sell. This is the number an edge has to clear."""
        return self.one_way(price, volatility, is_sell=False, is_etf=is_etf) + self.one_way(
            price, volatility, is_sell=True, is_etf=is_etf
        )

    def minimum_viable_notional(self) -> float:
        """Order size below which the flat commission floor dominates.

        Below this, the minimum ticket alone exceeds the percentage commission,
        so the strategy is paying a fixed tax on being small.
        """
        return self.min_commission_twd / self.commission_rate

    def stressed(self, multiple: float = 3.0) -> "CostModel":
        """A pessimistic variant for the stress gate.

        Statutory tax does not change under stress, so only the estimated,
        arguable components are multiplied. Tripling the tax as well would
        make the test look harsh while actually testing the wrong thing.
        """
        return replace(
            self,
            commission_discount=min(1.0, self.commission_discount * multiple),
            spread_capture=self.spread_capture * multiple,
            extra_slippage_bps=self.extra_slippage_bps * multiple,
            impact_coefficient=self.impact_coefficient * multiple,
            label=f"stress×{multiple:g}",
        )


DEFAULT_COSTS = CostModel()


def cost_report(model: CostModel, prices: pd.Series, volatility: pd.Series) -> pd.DataFrame:
    """Round-trip cost decomposition, for reading before trusting a backtest."""
    aligned = pd.concat([prices.rename("price"), volatility.rename("vol")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()
    commission = np.full(len(aligned), model.commission_rate * 2)
    spread = model.spread_cost(aligned["price"].to_numpy()) * 2
    slippage = np.full(len(aligned), model.extra_slippage_bps / 10_000.0 * 2)
    impact = model.impact_cost(aligned["vol"].to_numpy()) * 2
    tax = np.full(len(aligned), model.tax_rate)
    frame = pd.DataFrame(
        {
            "手續費": commission,
            "價差": spread,
            "滑價": slippage,
            "市場衝擊": impact,
            "證交稅": tax,
        },
        index=aligned.index,
    )
    frame["合計"] = frame.sum(axis=1)
    return frame * 10_000.0  # basis points
