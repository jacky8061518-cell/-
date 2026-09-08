"""Point-in-time, cost and validation-gate tests.

The lookahead tests carry the most weight. Every other failure here costs a bad
signal; a lookahead leak costs a strategy that backtests beautifully and loses
money from the first day it is live.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.control.registry import ModelRegistry, PromotionRefused, Stage
from quant_platform.data.pit import (
    LookaheadError,
    PointInTimeStore,
    SourceSpec,
    TAIWAN_SOURCES,
    assert_no_lookahead,
)
from quant_platform.research.backtest import BacktestSpec, run_backtest
from quant_platform.research.costs import CostModel, tick_size
from quant_platform.research.factors import FACTORS
from quant_platform.research.validation import (
    bootstrap_sharpe_interval,
    deflated_sharpe,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
    sharpe_ratio,
    validate,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2019-01-01", periods=900)
    frame = pd.DataFrame(index=dates)
    for i in range(60):
        drift = 0.0006 if i < 20 else (-0.0004 if i < 40 else 0.0)
        frame[f"S{i:02d}"] = 100 * np.cumprod(1 + rng.normal(drift, 0.018, len(dates)))
    frame["0050.TW"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.009, len(dates)))
    return frame


@pytest.fixture
def store(panel):
    store = PointInTimeStore()
    store.register("prices", panel, TAIWAN_SOURCES["prices"])
    return store


# --- Point in time --------------------------------------------------------


def test_visible_excludes_unpublished_rows(store, panel):
    as_of = panel.index[500]
    visible = store.visible("prices", as_of)
    assert visible.index.max() == as_of
    assert len(visible) == 501


def test_publication_lag_delays_visibility(panel):
    store = PointInTimeStore()
    lagged = SourceSpec("flows", publication_lag=timedelta(days=2), tradeable_lag_sessions=1)
    store.register("flows", panel, lagged)
    as_of = panel.index[500]
    visible = store.visible("flows", as_of)
    # Two days of publication lag means the newest rows are not yet usable.
    assert visible.index.max() < as_of


def test_tradeable_index_is_the_next_session(store, panel):
    as_of = panel.index[500]
    assert store.panel("prices").tradeable_index(as_of) == panel.index[501]


def test_tradeable_index_is_none_at_the_end_of_sample(store, panel):
    assert store.panel("prices").tradeable_index(panel.index[-1]) is None


def test_lookahead_guard_raises_on_future_data(store, panel):
    with pytest.raises(LookaheadError):
        assert_no_lookahead(store, "prices", panel.index[300], [panel.index[301]])


def test_lookahead_guard_accepts_published_data(store, panel):
    assert_no_lookahead(store, "prices", panel.index[300], [panel.index[300]])


def test_backtest_result_is_identical_when_future_data_is_removed(store, panel):
    """Truncating the panel after the last exit must not change any return."""
    factor = FACTORS["momentum"]
    spec = BacktestSpec(factor_id=factor.id, holding_days=21, end=panel.index[700])
    full = run_backtest(store, factor, spec, benchmark="0050.TW")

    truncated_store = PointInTimeStore()
    truncated_store.register("prices", panel.loc[: panel.index[760]], TAIWAN_SOURCES["prices"])
    partial = run_backtest(truncated_store, factor, spec, benchmark="0050.TW")

    pd.testing.assert_series_equal(full.net_returns, partial.net_returns)


def test_signal_delay_changes_the_result(store, panel):
    """If delaying execution changes nothing, the delay is not being applied."""
    factor = FACTORS["momentum"]
    base = run_backtest(store, factor, BacktestSpec(factor_id=factor.id), benchmark="0050.TW")
    delayed = run_backtest(
        store, factor, BacktestSpec(factor_id=factor.id, signal_delay_days=2), benchmark="0050.TW"
    )
    assert not base.net_returns.equals(delayed.net_returns)


# --- Costs ----------------------------------------------------------------


def test_tick_size_follows_the_exchange_bands():
    assert tick_size(8.0) == 0.01
    assert tick_size(15.0) == 0.05
    assert tick_size(88.0) == 0.10
    assert tick_size(250.0) == 0.50
    assert tick_size(640.0) == 1.00
    assert tick_size(1200.0) == 5.00


def test_sell_side_carries_the_transaction_tax():
    model = CostModel()
    buy = model.one_way(100.0, 0.3, is_sell=False)
    sell = model.one_way(100.0, 0.3, is_sell=True)
    assert float(sell - buy) == pytest.approx(model.tax_rate)


def test_etf_tax_is_lower_than_share_tax():
    model = CostModel()
    shares = model.round_trip(100.0, 0.3, is_etf=False)
    etf = model.round_trip(100.0, 0.3, is_etf=True)
    assert float(etf) < float(shares)


def test_cheap_stocks_pay_a_larger_relative_spread():
    model = CostModel()
    assert float(model.spread_cost(12.0)) > float(model.spread_cost(300.0))


def test_stress_does_not_inflate_statutory_tax():
    """Tripling a legislated rate would test the wrong thing."""
    model = CostModel()
    assert model.stressed(3).tax_rate == model.tax_rate
    assert model.stressed(3).commission_rate > model.commission_rate


def test_higher_costs_reduce_net_returns(store):
    factor = FACTORS["momentum"]
    spec = BacktestSpec(factor_id=factor.id)
    cheap = run_backtest(store, factor, spec, costs=CostModel(), benchmark="0050.TW")
    dear = run_backtest(store, factor, spec, costs=CostModel().stressed(3), benchmark="0050.TW")
    assert dear.net_returns.mean() < cheap.net_returns.mean()


def test_persistent_holdings_are_not_charged_twice(store):
    """Turnover, not headcount, drives cost. This is the bug that kills momentum."""
    factor = FACTORS["momentum"]
    result = run_backtest(store, factor, BacktestSpec(factor_id=factor.id), benchmark="0050.TW")
    assert result.turnover.mean() < 1.0
    round_trip_if_fully_rebuilt = result.costs / result.turnover.replace(0, np.nan)
    assert (result.costs <= round_trip_if_fully_rebuilt + 1e-12).all()


# --- Validation statistics ------------------------------------------------


def test_normal_helpers_match_known_values():
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-4)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    """Search hard enough and a worthless strategy looks good. This prices that."""
    assert expected_max_sharpe(1, 0.04) == 0.0
    assert expected_max_sharpe(10, 0.04) < expected_max_sharpe(1000, 0.04)


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.004, 0.02, 200))
    assert deflated_sharpe(returns, trials=2) > deflated_sharpe(returns, trials=500)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.004, 0.02, 300))
    point = sharpe_ratio(returns, 12)
    low, high = bootstrap_sharpe_interval(returns, 12)
    assert low < point < high


def test_pure_noise_does_not_pass_the_gate():
    """The gate's real job: rejecting something that only looks like a strategy."""
    from quant_platform.research.backtest import BacktestResult

    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", periods=150, freq="21B")
    noise = pd.Series(rng.normal(0.0005, 0.03, 150), index=index)
    spec = BacktestSpec(factor_id="noise_v1", holding_days=21)
    result = BacktestResult(
        spec=spec,
        gross_returns=noise + 0.002,
        net_returns=noise,
        costs=pd.Series(0.002, index=index),
        benchmark_returns=pd.Series(0.0, index=index),
        names_held=pd.Series(20, index=index),
        turnover=pd.Series(0.5, index=index),
        factor_id="noise_v1",
        universe_size=200,
    )
    # Shuffled variants score about as well as the real thing, which is exactly
    # what noise looks like from the inside.
    report = validate(result, result, result, [0.4, 0.5, 0.6, 0.55], trials=100)
    assert not report.approved
    assert any("Deflated Sharpe" in gate.name for gate in report.hard_failures)


# --- Registry -------------------------------------------------------------


def test_registry_refuses_to_skip_stages(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": True, "hard_failures": []})
    with pytest.raises(PromotionRefused, match="不能從"):
        registry.promote("m1", Stage.PRODUCTION)


def test_registry_refuses_a_failed_validation(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": False, "hard_failures": ["Deflated Sharpe"]})
    with pytest.raises(PromotionRefused, match="驗證閘門"):
        registry.promote("m1", Stage.VALIDATED)


def test_registry_enforces_minimum_time_in_stage(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": True, "hard_failures": []})
    registry.promote("m1", Stage.VALIDATED)
    registry.promote("m1", Stage.SHADOW)
    with pytest.raises(PromotionRefused, match="需滿"):
        registry.promote("m1", Stage.PAPER)


def test_real_money_stages_require_a_named_approver(tmp_path):
    from datetime import datetime, timedelta, timezone

    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": True, "hard_failures": []})
    registry.promote("m1", Stage.VALIDATED)
    registry.promote("m1", Stage.SHADOW)
    later = datetime.now(timezone.utc) + timedelta(days=120)
    registry.promote("m1", Stage.PAPER, now=later)
    with pytest.raises(PromotionRefused, match="人工核准"):
        registry.promote("m1", Stage.CANARY, now=later + timedelta(days=120))


def test_only_canary_and_production_are_deployable(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": True, "hard_failures": []})
    registry.promote("m1", Stage.VALIDATED)
    assert registry.deployable() == []


def test_registry_rebuilds_state_from_the_append_only_log(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m1", "f_v1", {}, {"approved": True, "hard_failures": []})
    registry.promote("m1", Stage.VALIDATED)
    reopened = ModelRegistry(tmp_path)
    assert reopened.get("m1").stage is Stage.VALIDATED
    assert len(reopened.get("m1").history) == 2
