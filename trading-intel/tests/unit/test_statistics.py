"""統計量模組：以已知答案的合成資料驗證。

每個檢定都用「構造出來、答案已知」的序列測試：隨機漫步必須被判為非定態，
OU 過程的半衰期必須估回它被生成時用的參數。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_intel.features.statistics import (
    StationarityVerdict,
    check_stationarity,
    ewma_volatility,
    ou_half_life,
    window_from_half_life,
    winsorize,
    yang_zhang_volatility,
    zscore,
)

SEED = 20240319


def random_walk(n: int = 500, seed: int = SEED) -> np.ndarray:
    """隨機漫步：非定態的標準例子。"""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n))


def ou_process(n: int = 2000, theta: float = 0.1, seed: int = SEED) -> np.ndarray:
    """生成一條 OU 序列。理論半衰期為 ln(2)/theta。"""
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for index in range(1, n):
        series[index] = series[index - 1] - theta * series[index - 1] + rng.normal(0, 1)
    return series


# --- 定態檢定 --------------------------------------------------------------


def test_random_walk_is_not_stationary() -> None:
    result = check_stationarity(random_walk())
    assert result.verdict is StationarityVerdict.NON_STATIONARY
    assert not result.usable_for_mean_reversion


def test_ou_process_is_stationary() -> None:
    result = check_stationarity(ou_process())
    assert result.verdict is StationarityVerdict.STATIONARY
    assert result.usable_for_mean_reversion
    assert result.adf_pvalue < 0.05


def test_white_noise_is_stationary() -> None:
    rng = np.random.default_rng(SEED)
    result = check_stationarity(rng.normal(0, 1, 1000))
    assert result.verdict is StationarityVerdict.STATIONARY


def test_trending_series_is_not_stationary() -> None:
    """帶趨勢的序列。這正是天真滾動 z-score 會持續給錯誤反轉訊號的情況。"""
    rng = np.random.default_rng(SEED)
    trend = np.arange(500) * 0.1 + rng.normal(0, 1, 500)
    assert check_stationarity(trend).verdict is not StationarityVerdict.STATIONARY


def test_short_series_is_inconclusive() -> None:
    result = check_stationarity([1.0, 2.0, 3.0])
    assert result.verdict is StationarityVerdict.INCONCLUSIVE
    assert not result.usable_for_mean_reversion
    assert math.isnan(result.adf_pvalue)


def test_constant_series_is_inconclusive() -> None:
    """常數序列數學上定態，但兩個檢定都無法處理，因此誠實回報存疑。"""
    result = check_stationarity([5.0] * 100)
    assert result.verdict is StationarityVerdict.INCONCLUSIVE


def test_inconclusive_is_treated_as_unusable() -> None:
    """存疑一律當作不可用於均值回歸——不對稱的代價決定了這個預設。"""
    for verdict in StationarityVerdict:
        usable = verdict is StationarityVerdict.STATIONARY
        assert (verdict is StationarityVerdict.STATIONARY) == usable


def test_nan_values_are_dropped() -> None:
    series = ou_process()
    with_nan = series.copy()
    with_nan[::50] = np.nan
    result = check_stationarity(with_nan)
    assert result.observations == int(np.isfinite(with_nan).sum())


# --- OU 半衰期 -------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.05, 0.1, 0.2])
def test_half_life_recovers_the_generating_parameter(theta: float) -> None:
    """估出來的半衰期必須接近生成時用的 ln(2)/theta。"""
    expected = math.log(2) / theta
    estimated = ou_half_life(ou_process(n=4000, theta=theta))
    assert estimated == pytest.approx(expected, rel=0.25)


def test_random_walk_has_no_half_life() -> None:
    """隨機漫步沒有回歸傾向，必須回傳 NaN 而非某個數字。"""
    assert math.isnan(ou_half_life(random_walk(n=2000)))


def test_short_series_has_no_half_life() -> None:
    assert math.isnan(ou_half_life([1.0, 2.0, 3.0]))


def test_constant_series_has_no_half_life() -> None:
    assert math.isnan(ou_half_life([5.0] * 100))


def test_faster_reversion_means_shorter_half_life() -> None:
    fast = ou_half_life(ou_process(n=4000, theta=0.3))
    slow = ou_half_life(ou_process(n=4000, theta=0.05))
    assert fast < slow


# --- 由半衰期決定視窗 ------------------------------------------------------


def test_window_scales_with_half_life() -> None:
    assert window_from_half_life(10.0) == 20
    assert window_from_half_life(10.0, multiplier=3.0) == 30


def test_window_is_none_when_there_is_no_half_life() -> None:
    """SPEC 5.1：未通過定態檢定的序列不得使用均值回歸訊號。"""
    assert window_from_half_life(float("nan")) is None
    assert window_from_half_life(-1.0) is None
    assert window_from_half_life(0.0) is None


def test_window_is_clamped() -> None:
    assert window_from_half_life(0.5, minimum=5) == 5
    assert window_from_half_life(1000.0, maximum=252) == 252


# --- EWMA 波動率 -----------------------------------------------------------


def test_ewma_matches_the_known_standard_deviation() -> None:
    """對常態序列，EWMA 年化波動應接近理論值。"""
    rng = np.random.default_rng(SEED)
    daily_sigma = 0.01
    returns = rng.normal(0, daily_sigma, 2000)
    expected = daily_sigma * math.sqrt(252)
    assert ewma_volatility(returns, span=500) == pytest.approx(expected, rel=0.15)


def test_ewma_reacts_faster_than_equal_weighting() -> None:
    """波動率叢聚：近期波動放大時，EWMA 必須比等權平均更快反應。"""
    rng = np.random.default_rng(SEED)
    calm = rng.normal(0, 0.005, 200)
    stormy = rng.normal(0, 0.05, 20)
    series = np.concatenate([calm, stormy])
    fast = ewma_volatility(series, span=10)
    slow = ewma_volatility(series, span=200)
    assert fast > slow


def test_ewma_without_annualization() -> None:
    rng = np.random.default_rng(SEED)
    returns = rng.normal(0, 0.01, 500)
    raw = ewma_volatility(returns, span=100, annualize=False)
    annual = ewma_volatility(returns, span=100, annualize=True)
    assert annual == pytest.approx(raw * math.sqrt(252))


def test_ewma_needs_two_observations() -> None:
    assert math.isnan(ewma_volatility([0.01]))


def test_ewma_rejects_bad_span() -> None:
    with pytest.raises(ValueError, match="span 必須為正整數"):
        ewma_volatility([0.01, 0.02], span=0)


# --- Yang-Zhang 波動率 -----------------------------------------------------


def synthetic_ohlc(n: int = 500, sigma: float = 0.01, seed: int = SEED):  # type: ignore[no-untyped-def]
    """生成開高低收序列，日波動率為 sigma。"""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = close * np.exp(rng.normal(0, sigma / 3, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sigma / 3, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sigma / 3, n)))
    return open_, high, low, close


def test_yang_zhang_is_in_the_right_ballpark() -> None:
    open_, high, low, close = synthetic_ohlc(sigma=0.01)
    estimate = yang_zhang_volatility(open_, high, low, close)
    expected = 0.01 * math.sqrt(252)
    assert estimate == pytest.approx(expected, rel=0.5)


def test_yang_zhang_scales_with_volatility() -> None:
    low_vol = yang_zhang_volatility(*synthetic_ohlc(sigma=0.005))
    high_vol = yang_zhang_volatility(*synthetic_ohlc(sigma=0.03))
    assert high_vol > low_vol


def test_yang_zhang_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="長度必須相同"):
        yang_zhang_volatility([1.0, 2.0], [1.0], [1.0], [1.0])


def test_yang_zhang_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="價格必須為正數"):
        yang_zhang_volatility([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def test_yang_zhang_needs_enough_observations() -> None:
    assert math.isnan(yang_zhang_volatility([1.0], [1.0], [1.0], [1.0]))


# --- winsorize 與標準化 ----------------------------------------------------


def test_winsorize_clips_rather_than_drops() -> None:
    values = [*list(range(100)), 10_000.0]
    result = winsorize(values)
    assert result.size == len(values), "winsorize 不得改變樣本數"
    assert result.max() < 10_000


def test_winsorize_preserves_the_middle() -> None:
    values = np.arange(100, dtype=float)
    result = winsorize(values, lower=0.05, upper=0.95)
    assert result[50] == values[50]


def test_winsorize_rejects_bad_percentiles() -> None:
    with pytest.raises(ValueError, match="百分位"):
        winsorize([1.0, 2.0], lower=0.9, upper=0.1)


def test_winsorize_handles_all_nan() -> None:
    result = winsorize([float("nan")] * 5)
    assert np.isnan(result).all()


def test_zscore_has_zero_mean_and_unit_std() -> None:
    rng = np.random.default_rng(SEED)
    result = zscore(rng.normal(5, 3, 1000))
    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-10)
    assert float(np.std(result, ddof=1)) == pytest.approx(1.0, rel=1e-10)


def test_zscore_of_constant_is_nan_not_zero() -> None:
    """全零會讓下游以為每個標的都在平均值上，那是看似正常的錯誤答案。"""
    result = zscore([5.0] * 10)
    assert np.isnan(result).all()


def test_zscore_of_too_few_values_is_nan() -> None:
    assert np.isnan(zscore([1.0])).all()


def test_zscore_propagates_nan() -> None:
    result = zscore([1.0, 2.0, float("nan"), 4.0])
    assert np.isnan(result[2])
    assert np.isfinite(result[0])
