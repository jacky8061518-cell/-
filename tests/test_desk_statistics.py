"""Property tests for the robust estimators.

These are the functions every agent depends on, so they are tested for the
properties that actually matter in production: immunity to outliers, correct
behaviour when data is missing, and the scale invariance that lets scores from
different names be compared at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_desk import statistics as stats


def test_robust_zscore_ignores_a_single_outlier():
    rng = np.random.default_rng(11)
    calm = pd.Series(list(rng.normal(0, 0.005, 60)) + [0.5])
    naive = (calm.iloc[-1] - calm.mean()) / calm.std()
    robust = stats.robust_zscore(calm, window=60).iloc[-1]
    # The outlier inflates the sample std it is measured against, so a naive
    # score caps out near sqrt(n); MAD is unaffected and keeps scaling.
    assert robust > naive * 5


def test_robust_zscore_degrades_instead_of_returning_nan():
    """A flat series must still produce a score, not silently drop out."""
    flat = pd.Series([0.0] * 60 + [0.02])
    score = stats.robust_zscore(flat, window=60, min_sigma=0.006).iloc[-1]
    assert np.isfinite(score)
    assert 1 < abs(score) < 10


def test_rank_to_normal_is_invariant_to_monotone_transforms():
    values = pd.Series([1.0, 2.0, 3.0, 10.0, 100.0])
    direct = stats.rank_to_normal(values)
    logged = stats.rank_to_normal(np.log(values))
    pd.testing.assert_series_equal(direct, logged)


def test_rank_to_normal_is_symmetric_around_zero():
    scores = stats.rank_to_normal(pd.Series(range(11), dtype="float64"))
    assert scores.iloc[5] == pytest.approx(0.0, abs=1e-9)
    assert scores.iloc[0] == pytest.approx(-scores.iloc[-1])


def test_rank_to_normal_preserves_missing_values():
    scores = stats.rank_to_normal(pd.Series([1.0, np.nan, 3.0]))
    assert np.isnan(scores.iloc[1])
    assert scores.notna().sum() == 2


def test_market_neutral_returns_removes_the_common_move():
    frame = pd.DataFrame(
        {"a": [-0.05, 0.01], "b": [-0.05, 0.01], "c": [-0.05, 0.01], "d": [-0.01, 0.01]}
    )
    residual = stats.market_neutral_returns(frame)
    # Three names moved identically, so the market factor absorbs the whole move.
    assert residual.loc[0, "a"] == pytest.approx(0.0)
    assert residual.loc[0, "d"] > 0, "the name that fell less must show positive residual"


def test_sigma_floor_stops_illiquid_names_printing_absurd_scores():
    quiet = pd.DataFrame({"x": [0.0] * 60 + [0.02]})
    unfloored = stats.robust_zscore_frame(quiet, window=60).iloc[-1, 0]
    floored = stats.robust_zscore_frame(quiet, window=60, min_sigma=0.006).iloc[-1, 0]
    assert not np.isfinite(unfloored) or abs(unfloored) > abs(floored)
    assert abs(floored) < 5


def test_ewma_volatility_reacts_faster_than_trailing_volatility():
    calm = [0.001] * 200
    stormy = [0.05, -0.05] * 15
    returns = pd.Series(calm + stormy)
    assert stats.ewma_volatility(returns) > stats.realised_volatility(returns, 252)


def test_ewma_volatility_returns_nan_rather_than_guessing():
    assert np.isnan(stats.ewma_volatility(pd.Series([0.01, 0.02])))


def test_effective_positions_counts_concentration_not_names():
    spread = pd.Series([0.25, 0.25, 0.25, 0.25])
    concentrated = pd.Series([0.97, 0.01, 0.01, 0.01])
    assert stats.effective_positions(spread) == pytest.approx(4.0)
    assert stats.effective_positions(concentrated) < 1.1


def test_correlation_clusters_group_names_that_move_together():
    rng = np.random.default_rng(7)
    driver = rng.normal(0, 0.02, 200)
    frame = pd.DataFrame(
        {
            "a": driver,
            "b": driver * 1.02 + rng.normal(0, 0.001, 200),
            "c": rng.normal(0, 0.02, 200),
        }
    )
    clusters = stats.correlation_clusters(frame, threshold=0.7)
    members = {frozenset(group) for group in clusters.values()}
    assert frozenset({"a", "b"}) in members
    assert frozenset({"c"}) in members


def test_population_stability_index_flags_a_shifted_distribution():
    rng = np.random.default_rng(3)
    baseline = pd.Series(rng.normal(0, 1, 2000))
    same = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(3, 1, 2000))
    assert stats.population_stability_index(baseline, same) < 0.1
    assert stats.population_stability_index(baseline, shifted) > 0.25
