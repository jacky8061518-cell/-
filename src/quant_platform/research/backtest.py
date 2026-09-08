"""Walk-forward, point-in-time, cost-aware backtest.

Three properties matter more than speed:

**No lookahead.** The factor sees only what was published by the decision date.
Positions open at the next tradeable session, never at the price that produced
the signal. Every read is checked against the point-in-time store rather than
trusted.

**Non-overlapping holdings.** The rebalance step equals the holding period, so
each return observation is independent. Overlapping windows inflate the
apparent sample size several-fold and make every significance test optimistic.

**Costs on the actual turnover.** A name carried from one period into the next
is not bought again, so cost is charged on the fraction of the book that
actually changes, priced at each name's own price level and volatility. Charging
a full round trip every rebalance is the single easiest way to make a genuinely
profitable low-turnover strategy look like a loser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.pit import PointInTimeStore, assert_no_lookahead
from .costs import DEFAULT_COSTS, CostModel
from .factors import Factor

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestSpec:
    """Everything that defines one experiment. Serialised into the run record."""

    factor_id: str
    holding_days: int = 21
    quantile: float = 0.1
    long_short: bool = True
    max_names: int = 50
    min_names: int = 10
    universe_label: str = "all"
    signal_delay_days: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not 0 < self.quantile <= 0.5:
            raise ValueError("分位必須介於 0 與 0.5 之間")
        if self.holding_days < 1:
            raise ValueError("持有期間至少一個交易日")


@dataclass
class BacktestResult:
    """Returns plus everything needed to argue about them."""

    spec: BacktestSpec
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    benchmark_returns: pd.Series
    names_held: pd.Series
    turnover: pd.Series
    factor_id: str
    universe_size: int

    @property
    def periods_per_year(self) -> float:
        return TRADING_DAYS / self.spec.holding_days

    @property
    def trades(self) -> int:
        return int(self.names_held.sum())

    def summary(self) -> dict[str, float]:
        net = self.net_returns.dropna()
        if net.empty:
            return {}
        scale = self.periods_per_year
        mean = float(net.mean())
        std = float(net.std(ddof=1))
        gross_mean = float(self.gross_returns.dropna().mean())
        equity = (1 + net).cumprod()
        drawdown = float((equity / equity.cummax() - 1).min())
        # A dollar-neutral book carries no index exposure, so its hurdle is cash,
        # not the index. Scoring a market-neutral strategy as "excess over 0050"
        # during a bull market measures the market, not the strategy.
        hurdle = (
            pd.Series(0.0, index=net.index)
            if self.spec.long_short
            else self.benchmark_returns.reindex(net.index).fillna(0.0)
        )
        excess = net - hurdle
        return {
            "periods": len(net),
            "trades": self.trades,
            "gross_return_per_period": gross_mean,
            "net_return_per_period": mean,
            "cost_per_period": float(self.costs.dropna().mean()),
            "cost_share_of_gross": (
                float(self.costs.dropna().mean() / abs(gross_mean)) if gross_mean else np.nan
            ),
            "annual_return": (1 + mean) ** scale - 1,
            "annual_volatility": std * np.sqrt(scale),
            "sharpe": (mean / std * np.sqrt(scale)) if std > 0 else np.nan,
            "hit_rate": float((net > 0).mean()),
            "max_drawdown": drawdown,
            "benchmark_annual": (1 + float(self.benchmark_returns.dropna().mean())) ** scale - 1,
            "hurdle": "cash" if self.spec.long_short else "benchmark",
            "excess_annual": (1 + float(excess.mean())) ** scale - 1,
            "information_ratio": (
                float(excess.mean() / excess.std(ddof=1) * np.sqrt(scale))
                if excess.std(ddof=1) > 0
                else np.nan
            ),
        }

    def by_year(self) -> pd.DataFrame:
        """Per-year breakdown. A factor that only worked in one year did not work."""
        net = self.net_returns.dropna()
        if net.empty:
            return pd.DataFrame()
        grouped = net.groupby(net.index.year)
        frame = pd.DataFrame(
            {
                "periods": grouped.size(),
                "total_return": grouped.apply(lambda s: (1 + s).prod() - 1),
                "mean_return": grouped.mean(),
                "hit_rate": grouped.apply(lambda s: float((s > 0).mean())),
                "worst": grouped.min(),
            }
        )
        benchmark = self.benchmark_returns.reindex(net.index).fillna(0.0)
        frame["benchmark"] = benchmark.groupby(benchmark.index.year).apply(
            lambda s: (1 + s).prod() - 1
        )
        return frame.reset_index(names="year")


def run_backtest(
    store: PointInTimeStore,
    factor: Factor,
    spec: BacktestSpec,
    costs: CostModel = DEFAULT_COSTS,
    benchmark: str = "0050.TW",
    universe: list[str] | None = None,
    check_lookahead: bool = True,
) -> BacktestResult:
    """Run one walk-forward experiment and return its full record."""
    panel = store.panel("prices")
    sessions = panel.frame.index
    prices = panel.frame

    if universe is not None:
        keep = [column for column in universe if column in prices.columns]
        columns = keep + ([benchmark] if benchmark in prices.columns and benchmark not in keep else [])
        prices = prices[columns]

    filled = prices.ffill()
    vol = filled.pct_change(fill_method=None).rolling(120, min_periods=60).std() * np.sqrt(
        TRADING_DAYS
    )

    start = pd.Timestamp(spec.start) if spec.start is not None else sessions[factor.min_history]
    end = pd.Timestamp(spec.end) if spec.end is not None else sessions[-1]

    step = spec.holding_days
    lag = panel.spec.tradeable_lag_sessions + spec.signal_delay_days

    rows: list[dict] = []
    position = int(np.searchsorted(sessions, start))
    position = max(position, factor.min_history)

    previous_names: set[str] = set()

    while position + lag + step < len(sessions):
        decision_date = sessions[position]
        if decision_date > end:
            break

        visible = panel.visible(decision_date)
        if universe is not None:
            visible = visible[[c for c in prices.columns if c in visible.columns]]
        if check_lookahead:
            assert_no_lookahead(store, "prices", decision_date, visible.index[-3:])

        scores = factor.score(visible.drop(columns=[benchmark], errors="ignore"))
        if scores.empty or len(scores) < spec.min_names * 2:
            position += step
            continue

        count = int(max(spec.min_names, min(spec.max_names, len(scores) * spec.quantile)))
        longs = list(scores.nlargest(count).index)
        shorts = list(scores.nsmallest(count).index) if spec.long_short else []

        entry_index = position + lag
        exit_index = entry_index + step
        entry_date, exit_date = sessions[entry_index], sessions[exit_index]

        entry = filled.loc[entry_date]
        exit_prices = filled.loc[exit_date]
        volatility = vol.loc[entry_date]

        long_return, long_round_trip = _leg(longs, entry, exit_prices, volatility, costs, +1)
        short_return, short_round_trip = _leg(shorts, entry, exit_prices, volatility, costs, -1)

        if np.isnan(long_return) and np.isnan(short_return):
            position += step
            continue

        if spec.long_short and shorts:
            gross = np.nanmean([long_return, short_return])
            round_trip = np.nanmean([long_round_trip, short_round_trip])
        else:
            gross, round_trip = long_return, long_round_trip

        held = set(longs) | set(shorts)
        # Only the part of the book that changed is traded, so only that part
        # pays. The first period is a full build, hence turnover of one.
        turnover = 1.0 if not previous_names else len(held - previous_names) / max(len(held), 1)
        cost = round_trip * turnover
        previous_names = held

        benchmark_return = np.nan
        if benchmark in filled.columns:
            base = float(entry.get(benchmark, np.nan))
            final = float(exit_prices.get(benchmark, np.nan))
            if np.isfinite(base) and np.isfinite(final) and base > 0:
                benchmark_return = final / base - 1

        rows.append(
            {
                "date": exit_date,
                "gross": gross,
                "cost": cost,
                "net": gross - cost,
                "benchmark": benchmark_return,
                "names": len(held),
                "turnover": turnover,
            }
        )
        position += step

    if not rows:
        empty = pd.Series(dtype="float64")
        return BacktestResult(
            spec, empty, empty, empty, empty, empty, empty, factor.id, prices.shape[1]
        )

    frame = pd.DataFrame(rows).set_index("date")
    return BacktestResult(
        spec=spec,
        gross_returns=frame["gross"],
        net_returns=frame["net"],
        costs=frame["cost"],
        benchmark_returns=frame["benchmark"],
        names_held=frame["names"],
        turnover=frame["turnover"],
        factor_id=factor.id,
        universe_size=prices.shape[1],
    )


def _leg(
    names: list[str],
    entry: pd.Series,
    exit_prices: pd.Series,
    volatility: pd.Series,
    costs: CostModel,
    sign: int,
) -> tuple[float, float]:
    """Equal-weighted return, and the round-trip cost of fully replacing this leg.

    The caller scales the cost by realised turnover; this function only prices
    what a complete rebuild would cost.
    """
    if not names:
        return np.nan, np.nan
    entry_prices = entry.reindex(names).astype("float64")
    final_prices = exit_prices.reindex(names).astype("float64")
    vols = volatility.reindex(names).astype("float64").fillna(0.35)
    valid = entry_prices.notna() & final_prices.notna() & (entry_prices > 0)
    if not valid.any():
        return np.nan, np.nan
    raw = (final_prices[valid] / entry_prices[valid] - 1.0) * sign
    cost = costs.round_trip(entry_prices[valid].to_numpy(), vols[valid].to_numpy())
    return float(raw.mean()), float(np.mean(cost))
