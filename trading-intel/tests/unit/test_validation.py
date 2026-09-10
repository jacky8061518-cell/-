"""驗證協定：多重檢定與過度配適的防線。

這一組測試的核心是「用已知答案的資料驗證」：
純雜訊的策略必須被判為無效，真有 alpha 的策略必須通過。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_intel.backtest.validation import (
    capacity_curve,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    iter_split_returns,
    probability_of_backtest_overfitting,
    purged_walk_forward,
    sharpe_ratio,
)

SEED = 20240319


# --- Purged walk-forward ---------------------------------------------------


def test_walk_forward_train_always_precedes_test() -> None:
    for split in purged_walk_forward(1000, n_splits=5):
        assert max(split.train) < min(split.test)


def test_embargo_creates_a_gap() -> None:
    """留白的寬度必須等於指定的 embargo。"""
    embargo = 10
    for split in purged_walk_forward(1000, n_splits=4, embargo=embargo):
        gap = min(split.test) - max(split.train) - 1
        assert gap == embargo


def test_no_embargo_means_adjacent() -> None:
    for split in purged_walk_forward(1000, n_splits=4, embargo=0):
        assert min(split.test) == max(split.train) + 1


def test_train_and_test_never_overlap() -> None:
    for split in purged_walk_forward(500, n_splits=5, embargo=5):
        assert not (set(split.train) & set(split.test))


def test_training_set_grows() -> None:
    """walk-forward 的訓練集應逐次擴大。"""
    splits = purged_walk_forward(1000, n_splits=5)
    sizes = [len(split.train) for split in splits]
    assert sizes == sorted(sizes)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1}, "至少為 2"),
        ({"embargo": -1}, "不得為負數"),
    ],
)
def test_walk_forward_rejects_bad_arguments(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        purged_walk_forward(1000, **kwargs)


def test_walk_forward_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="不足以切成"):
        purged_walk_forward(3, n_splits=10)


# --- Combinatorial purged CV -----------------------------------------------


def test_cpcv_produces_multiple_paths() -> None:
    """C(6,2) = 15 條路徑，而不是一條回測曲線。"""
    splits = combinatorial_purged_cv(1200, n_groups=6, n_test_groups=2)
    assert len(splits) == 15


def test_cpcv_splits_do_not_overlap() -> None:
    for split in combinatorial_purged_cv(1200, n_groups=6, n_test_groups=2):
        assert not (set(split.train) & set(split.test))


def test_cpcv_embargo_removes_neighbours() -> None:
    """留白會讓訓練集變小，因為測試集周邊的樣本被剔除。"""
    without = combinatorial_purged_cv(1200, n_groups=6, n_test_groups=2, embargo=0)
    with_embargo = combinatorial_purged_cv(1200, n_groups=6, n_test_groups=2, embargo=20)
    assert len(with_embargo[0].train) < len(without[0].train)


def test_cpcv_rejects_bad_group_counts() -> None:
    with pytest.raises(ValueError, match="測試組數"):
        combinatorial_purged_cv(1200, n_groups=4, n_test_groups=4)


# --- Sharpe ----------------------------------------------------------------


def test_sharpe_of_known_series() -> None:
    """日報酬平均 0.001、標準差 0.01，年化 Sharpe 約 0.1*sqrt(252)。"""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(0.001, 0.01, 10_000)
    expected = 0.001 / 0.01 * math.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=0.1)


def test_sharpe_of_constant_series_is_nan() -> None:
    """浮點誤差讓常數序列的標準差是 1.7e-18 而非 0。

    若不用相對門檻把關，Sharpe 會算出 9e16——一個荒謬但不報錯的數字，
    而且它會在任何排序中排第一。
    """
    assert math.isnan(sharpe_ratio([0.01] * 100))


def test_sharpe_of_near_constant_series_is_nan() -> None:
    values = [0.01 + index * 1e-18 for index in range(100)]
    assert math.isnan(sharpe_ratio(values))


def test_sharpe_of_genuinely_small_but_real_volatility_is_finite() -> None:
    """真實的低波動不該被誤判為無波動。"""
    rng = np.random.default_rng(SEED)
    assert math.isfinite(sharpe_ratio(rng.normal(0.0001, 0.0005, 500)))


def test_sharpe_needs_two_observations() -> None:
    assert math.isnan(sharpe_ratio([0.01]))


# --- Deflated Sharpe -------------------------------------------------------
#
# 以下所有 Sharpe 皆為「每期」非年化值。0.1 的日 Sharpe 約等於年化 1.59。
# 試驗變異數 0.0025 代表那些試驗的 Sharpe 標準差為 0.05。

TRIAL_VAR = 0.0025


def test_more_trials_lowers_the_deflated_sharpe() -> None:
    """同一個 Sharpe，試越多次越不可信——這是本函式的全部重點。"""
    single = deflated_sharpe_ratio(
        0.10, n_trials=1, n_observations=1000, variance_of_trial_sharpes=TRIAL_VAR
    )
    many = deflated_sharpe_ratio(
        0.10, n_trials=1000, n_observations=1000, variance_of_trial_sharpes=TRIAL_VAR
    )
    assert single > many


def test_a_strong_sharpe_survives_few_trials() -> None:
    assert (
        deflated_sharpe_ratio(
            0.15, n_trials=1, n_observations=2000, variance_of_trial_sharpes=TRIAL_VAR
        )
        > 0.95
    )


def test_a_mediocre_sharpe_dies_under_many_trials() -> None:
    """試一千次挑出來的普通 Sharpe，幾乎必然是運氣。"""
    assert (
        deflated_sharpe_ratio(
            0.05, n_trials=1000, n_observations=500, variance_of_trial_sharpes=0.01
        )
        < 0.5
    )


def test_more_observations_raises_confidence() -> None:
    """觀測越多，同一個 Sharpe 的估計越可信。"""
    short = deflated_sharpe_ratio(
        0.10, n_trials=10, n_observations=100, variance_of_trial_sharpes=TRIAL_VAR
    )
    long = deflated_sharpe_ratio(
        0.10, n_trials=10, n_observations=5000, variance_of_trial_sharpes=TRIAL_VAR
    )
    assert long > short


def test_wider_trial_dispersion_lowers_confidence() -> None:
    """試驗之間的 Sharpe 越分散，最大值越可能只是運氣。"""
    tight = deflated_sharpe_ratio(
        0.10, n_trials=50, n_observations=1000, variance_of_trial_sharpes=0.0001
    )
    wide = deflated_sharpe_ratio(
        0.10, n_trials=50, n_observations=1000, variance_of_trial_sharpes=0.01
    )
    assert tight > wide


def test_deflated_sharpe_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="試驗次數"):
        deflated_sharpe_ratio(
            0.1, n_trials=0, n_observations=100, variance_of_trial_sharpes=TRIAL_VAR
        )
    with pytest.raises(ValueError, match="觀測數"):
        deflated_sharpe_ratio(
            0.1, n_trials=1, n_observations=2, variance_of_trial_sharpes=TRIAL_VAR
        )
    with pytest.raises(ValueError, match="變異數必須為正數"):
        deflated_sharpe_ratio(0.1, n_trials=1, n_observations=100, variance_of_trial_sharpes=0.0)


def test_deflated_sharpe_handles_nan() -> None:
    assert math.isnan(
        deflated_sharpe_ratio(
            float("nan"), n_trials=1, n_observations=100, variance_of_trial_sharpes=TRIAL_VAR
        )
    )


# --- PBO -------------------------------------------------------------------


def test_pure_noise_strategies_have_high_pbo() -> None:
    """全是雜訊的策略群，樣本內冠軍在樣本外應該隨機——PBO 接近 0.5。"""
    rng = np.random.default_rng(SEED)
    trials = [rng.normal(0, 0.01, 1200) for _ in range(20)]
    pbo = probability_of_backtest_overfitting(trials, n_groups=8)
    assert 0.25 < pbo < 0.75, f"純雜訊的 PBO 應接近 0.5，實得 {pbo}"


def test_a_genuinely_superior_strategy_has_low_pbo() -> None:
    """其中一個策略真有 alpha 時，它在樣本內外都會贏，PBO 應該低。"""
    rng = np.random.default_rng(SEED)
    trials = [rng.normal(0, 0.01, 1200) for _ in range(9)]
    trials.append(rng.normal(0.004, 0.01, 1200))
    pbo = probability_of_backtest_overfitting(trials, n_groups=8)
    assert pbo < 0.25, f"真有 alpha 的策略群 PBO 應偏低，實得 {pbo}"


def test_pbo_rejects_single_trial() -> None:
    with pytest.raises(ValueError, match="至少需要兩組"):
        probability_of_backtest_overfitting([np.zeros(100)])


def test_pbo_rejects_odd_group_count() -> None:
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="偶數"):
        probability_of_backtest_overfitting(
            [rng.normal(0, 0.01, 100) for _ in range(3)], n_groups=7
        )


# --- 容量曲線 --------------------------------------------------------------


def test_capacity_decays_with_size() -> None:
    """資金越大，報酬被市場衝擊吃掉越多。"""
    points = capacity_curve(
        0.15,
        capital_levels=[1e6, 1e7, 1e8, 1e9],
        average_daily_volume_value=5e8,
        participation_cap=0.05,
        impact_coefficient=0.1,
        turnover_per_year=12.0,
    )
    returns = [point.annualized_return for point in points]
    assert returns == sorted(returns, reverse=True)
    assert points[0].decay < points[-1].decay


def test_zero_capital_has_no_decay() -> None:
    points = capacity_curve(
        0.15,
        capital_levels=[0.0],
        average_daily_volume_value=5e8,
        participation_cap=0.05,
        impact_coefficient=0.1,
        turnover_per_year=12.0,
    )
    assert points[0].annualized_return == pytest.approx(0.15)
    assert points[0].decay == pytest.approx(0.0)


def test_capacity_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="日均量金額"):
        capacity_curve(
            0.1,
            capital_levels=[1e6],
            average_daily_volume_value=0,
            participation_cap=0.05,
            impact_coefficient=0.1,
            turnover_per_year=1.0,
        )
    with pytest.raises(ValueError, match="參與率上限"):
        capacity_curve(
            0.1,
            capital_levels=[1e6],
            average_daily_volume_value=1e8,
            participation_cap=1.5,
            impact_coefficient=0.1,
            turnover_per_year=1.0,
        )


# --- 切分取值 --------------------------------------------------------------


def test_iter_split_returns_matches_the_splits() -> None:
    returns = np.arange(100, dtype=float)
    splits = purged_walk_forward(100, n_splits=3)
    chunks = list(iter_split_returns(returns, splits))
    assert len(chunks) == len(splits)
    for chunk, split in zip(chunks, splits, strict=True):
        assert chunk.size == len(split.test)
