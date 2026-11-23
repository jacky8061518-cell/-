"""Operator console helpers.

This module sits *above* the four planes rather than inside one. An interface a
human drives has to read the research plane's results and the control plane's
registry in the same screen, which is exactly the coupling the plane boundaries
forbid between the planes themselves. Keeping it out of every plane directory is
the point: the isolation rules apply to the machinery, not to the window a
person looks through.

What it must never do is give the interface powers the planes withhold. It
calls the same public functions a script would, and every promotion still goes
through the registry's gates, so a button here cannot approve what the CLI
would refuse.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .control.registry import ModelRegistry, validation_record
from .data.pit import TAIWAN_SOURCES, PointInTimeStore
from .research.backtest import BacktestSpec, run_backtest
from .research.costs import CostModel
from .research.factors import FACTORS, Factor
from .research.validation import sharpe_ratio, validate

DATABASE = Path(__file__).resolve().parents[2] / "data" / "databases" / "tw"


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    frame = pd.read_parquet(DATABASE / "adjusted-prices.parquet")
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_master() -> pd.DataFrame:
    return pd.read_csv(DATABASE / "security-master.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def build_universe(min_market_cap: float, industries: tuple[str, ...]) -> list[str]:
    """Names that clear the size floor, using issued shares times last close.

    A proxy, and a loud one: the database carries no volume, so this is standing
    in for liquidity. It is good enough to keep untradeable micro caps out of a
    backtest and not good enough to size a real order.
    """
    prices, master = load_prices(), load_master()
    stocks = master[master["Asset type"] == "股票"]
    if industries:
        stocks = stocks[stocks["Industry"].isin(industries)]
    last = prices.ffill().iloc[-1]
    cap = pd.to_numeric(stocks["Issued shares"], errors="coerce") * stocks["Yahoo ticker"].map(last)
    tickers = stocks.loc[cap >= min_market_cap, "Yahoo ticker"]
    return [ticker for ticker in tickers if ticker in prices.columns]


def industries() -> list[str]:
    master = load_master()
    values = master.loc[master["Asset type"] == "股票", "Industry"].dropna().unique()
    return sorted(str(value) for value in values)


def _store() -> PointInTimeStore:
    store = PointInTimeStore()
    store.register("prices", load_prices(), TAIWAN_SOURCES["prices"])
    return store


def _cost_model(discount: float, slippage_bps: float, participation: float) -> CostModel:
    return CostModel(
        commission_discount=discount,
        extra_slippage_bps=slippage_bps,
        participation=participation,
    )


@st.cache_data(ttl=1800, show_spinner=False, max_entries=40)
def run_experiment(
    factor_key: str,
    holding_days: int,
    quantile: float,
    long_short: bool,
    respect_limits: bool,
    signal_delay_days: int,
    min_market_cap: float,
    industries_selected: tuple[str, ...],
    commission_discount: float,
    slippage_bps: float,
    participation: float,
    cost_multiple: float,
    start_year: int,
) -> dict:
    """One backtest, returned as plain data so Streamlit can cache it."""
    factor = FACTORS[factor_key]
    universe = build_universe(min_market_cap, industries_selected)
    if len(universe) < 40:
        return {"error": f"universe 只有 {len(universe)} 檔，樣本太小，無法做橫斷面排序。"}

    spec = BacktestSpec(
        factor_id=factor.id,
        holding_days=holding_days,
        quantile=quantile,
        long_short=long_short,
        respect_price_limits=respect_limits,
        signal_delay_days=signal_delay_days,
        start=pd.Timestamp(f"{start_year}-01-01"),
    )
    costs = _cost_model(commission_discount, slippage_bps, participation)
    if cost_multiple != 1.0:
        costs = costs.stressed(cost_multiple)

    result = run_backtest(_store(), factor, spec, costs=costs, universe=universe)
    summary = result.summary()
    if not summary:
        return {"error": "這組參數沒有產生任何交易期間。可能是持有期間太長或樣本期間太短。"}

    equity = (1 + result.net_returns.fillna(0)).cumprod()
    benchmark_equity = (1 + result.benchmark_returns.fillna(0)).cumprod()
    return {
        "summary": summary,
        "spec": asdict(spec),
        "universe_size": len(universe),
        "turnover": float(result.turnover.mean()),
        "blocked_per_period": (
            float(result.blocked_names.mean()) if result.blocked_names is not None else 0.0
        ),
        "net_returns": result.net_returns,
        "equity": equity,
        "benchmark_equity": benchmark_equity,
        "by_year": result.by_year(),
        "sharpe": sharpe_ratio(result.net_returns, result.periods_per_year),
    }


@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def run_validation(
    factor_key: str,
    holding_days: int,
    quantile: float,
    long_short: bool,
    respect_limits: bool,
    min_market_cap: float,
    industries_selected: tuple[str, ...],
    commission_discount: float,
    slippage_bps: float,
    participation: float,
    start_year: int,
    trials: int,
    null_draws: int,
) -> dict:
    """The full gate: base run, cost stress, execution delay, randomised null."""
    factor = FACTORS[factor_key]
    universe = build_universe(min_market_cap, industries_selected)
    store = _store()
    spec = BacktestSpec(
        factor_id=factor.id,
        holding_days=holding_days,
        quantile=quantile,
        long_short=long_short,
        respect_price_limits=respect_limits,
        start=pd.Timestamp(f"{start_year}-01-01"),
    )
    costs = _cost_model(commission_discount, slippage_bps, participation)

    base = run_backtest(store, factor, spec, costs=costs, universe=universe)
    if not base.summary():
        return {"error": "回測沒有產生任何交易期間。"}

    stressed = run_backtest(store, factor, spec, costs=costs.stressed(3), universe=universe)
    delayed = run_backtest(
        store, factor, replace(spec, signal_delay_days=1), costs=costs, universe=universe
    )

    nulls: list[float] = []
    for seed in range(null_draws):
        rng = np.random.default_rng(seed)

        def permuted(prices, _factor=factor, _rng=rng):
            scores = _factor.compute(prices)
            return pd.Series(_rng.permutation(scores.to_numpy()), index=scores.index)

        shuffled = Factor(
            name=f"{factor.name}_null{seed}",
            version=factor.version,
            claim="null",
            fails_when="null",
            compute=permuted,
            min_history=factor.min_history,
        )
        null_result = run_backtest(
            store, shuffled, spec, costs=costs, universe=universe, check_lookahead=False
        )
        value = sharpe_ratio(null_result.net_returns, null_result.periods_per_year)
        if np.isfinite(value):
            nulls.append(value)

    report = validate(base, stressed, delayed, nulls, trials=trials)
    return {
        "record": validation_record(report),
        "verdict": report.verdict(),
        "gates": report.to_frame(),
        "metrics": report.metrics,
        "nulls": nulls,
        "spec": asdict(spec),
        "universe_size": len(universe),
        "stressed_net": stressed.summary().get("net_return_per_period"),
        "delayed_net": delayed.summary().get("net_return_per_period"),
        "base_net": base.summary().get("net_return_per_period"),
    }


def registry() -> ModelRegistry:
    return ModelRegistry()


def model_id_for(factor_key: str, holding: int, quantile: float, limits: bool) -> str:
    suffix = "" if limits else "_nolimit"
    return f"{FACTORS[factor_key].id}_h{holding}_q{int(quantile * 100)}{suffix}"
