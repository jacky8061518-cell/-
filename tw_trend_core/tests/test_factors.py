"""
tests/test_factors.py
針對 core/factors.py 四個量化強化因子的單元測試：趨勢/震盪制度判別、
產業相對 Z-Score、成交量 Z-Score、流動性篩選，以及合成 Edge Score。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.factors import (
    EDGE_SCORE_WEIGHTS,
    apply_liquidity_filter,
    attach_industry_relative_zscore,
    classify_regime,
    compute_edge_score,
    compute_industry_median_zscores,
    compute_liquidity,
    compute_trend_efficiency,
    compute_volume_zscore,
)


def _series_df(columns: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    idx = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.DataFrame(columns, index=idx)


# ---------------------------------------------------------------------------
# 趨勢/震盪制度判別
# ---------------------------------------------------------------------------

def test_compute_trend_efficiency_straight_line_is_near_one():
    """單邊直線上漲：淨變動 = 總路徑長度，效率比應接近 1。"""
    prices = _series_df({"TREND.TW": [100.0 + i for i in range(25)]})
    er = compute_trend_efficiency(prices, length=20)
    assert er["TREND.TW"] == pytest.approx(1.0, abs=1e-9)


def test_compute_trend_efficiency_oscillation_is_near_zero():
    """來回震盪、首尾價格相同：淨變動趨近 0，效率比應趨近 0。"""
    base = [100.0, 105.0] * 13
    prices = _series_df({"CHOP.TW": base[:25]})
    er = compute_trend_efficiency(prices, length=20)
    assert er["CHOP.TW"] < 0.2


def test_compute_trend_efficiency_flat_price_is_nan_not_error():
    """完全無波動（路徑長度為 0）不應拋出除以 0 的例外，應回傳 NaN。"""
    prices = _series_df({"FLAT.TW": [100.0] * 25})
    er = compute_trend_efficiency(prices, length=20)
    assert pd.isna(er["FLAT.TW"])


def test_classify_regime_labels_trending_up_and_down():
    prices = _series_df({
        "UP.TW": [100.0 + i for i in range(25)],
        "DOWN.TW": [125.0 - i for i in range(25)],
    })
    regimes = classify_regime(prices, length=20)
    assert regimes.loc["UP.TW", "regime"] == "TRENDING_UP"
    assert regimes.loc["DOWN.TW", "regime"] == "TRENDING_DOWN"


def test_classify_regime_falls_back_to_range_bound_on_flat_price():
    prices = _series_df({"FLAT.TW": [100.0] * 25})
    regimes = classify_regime(prices, length=20)
    assert regimes.loc["FLAT.TW", "regime"] == "RANGE_BOUND"


def test_classify_regime_low_efficiency_is_range_bound():
    base = [100.0, 105.0] * 13
    prices = _series_df({"CHOP.TW": base[:25]})
    regimes = classify_regime(prices, length=20)
    assert regimes.loc["CHOP.TW", "regime"] == "RANGE_BOUND"


# ---------------------------------------------------------------------------
# 產業相對 Z-Score
# ---------------------------------------------------------------------------

def test_industry_median_and_relative_zscore():
    full_latest = pd.DataFrame(
        {"z_score": [2.5, 2.3, 0.1, -0.2]},
        index=["A.TW", "B.TW", "C.TW", "D.TW"],
    )
    industry_by_ticker = {"A.TW": "半導體業", "B.TW": "半導體業", "C.TW": "半導體業", "D.TW": "食品工業"}

    median_by_group = compute_industry_median_zscores(full_latest, industry_by_ticker)
    assert median_by_group["半導體業"] == pytest.approx(2.3)

    anomalies = pd.DataFrame({"z_score": [2.5], "industry": ["半導體業"]}, index=["A.TW"])
    enriched = attach_industry_relative_zscore(anomalies, median_by_group, industry_by_ticker)
    assert enriched.loc["A.TW", "industry_median_z"] == pytest.approx(2.3)
    assert enriched.loc["A.TW", "industry_relative_z"] == pytest.approx(0.2)


def test_industry_relative_zscore_near_zero_means_sector_wide_move():
    """個股 Z-Score 跟產業中位數幾乎一樣，代表這是整個產業在動，不是個股獨立訊號。"""
    full_latest = pd.DataFrame({"z_score": [2.5, 2.4, 2.6]}, index=["A.TW", "B.TW", "C.TW"])
    industry_by_ticker = {"A.TW": "半導體業", "B.TW": "半導體業", "C.TW": "半導體業"}
    median_by_group = compute_industry_median_zscores(full_latest, industry_by_ticker)

    anomalies = pd.DataFrame({"z_score": [2.5], "industry": ["半導體業"]}, index=["A.TW"])
    enriched = attach_industry_relative_zscore(anomalies, median_by_group, industry_by_ticker)
    assert abs(enriched.loc["A.TW", "industry_relative_z"]) < 0.2


# ---------------------------------------------------------------------------
# 成交量 Z-Score
# ---------------------------------------------------------------------------

def test_compute_volume_zscore_matches_manual_calculation():
    volumes = [1000.0] * 20 + [5000.0]
    df = _series_df({"A.TW": volumes})
    result = compute_volume_zscore(df, length=20)

    window = pd.Series(volumes[-20:])
    expected_mean = window.mean()
    expected_std = window.std()
    expected_z = (volumes[-1] - expected_mean) / expected_std

    assert result.loc["A.TW", "volume_z_score"] == pytest.approx(expected_z)
    assert result.loc["A.TW", "avg_volume"] == pytest.approx(expected_mean)


def test_compute_volume_zscore_high_volume_gives_positive_z():
    volumes = [1000.0] * 20 + [10000.0]
    df = _series_df({"SPIKE.TW": volumes})
    result = compute_volume_zscore(df, length=20)
    assert result.loc["SPIKE.TW", "volume_z_score"] > 2


# ---------------------------------------------------------------------------
# 流動性
# ---------------------------------------------------------------------------

def test_compute_liquidity_is_volume_times_price():
    avg_volume = pd.Series({"A.TW": 100_000.0})
    price = pd.Series({"A.TW": 50.0})
    turnover = compute_liquidity(avg_volume, price)
    assert turnover["A.TW"] == pytest.approx(5_000_000.0)


def test_apply_liquidity_filter_excludes_thin_names():
    df = pd.DataFrame({"avg_daily_turnover": [10_000_000.0, 500_000.0, 3_000_000.0]}, index=["LIQUID", "THIN", "BORDERLINE"])
    filtered = apply_liquidity_filter(df, min_turnover=3_000_000)
    assert "LIQUID" in filtered.index
    assert "THIN" not in filtered.index
    assert "BORDERLINE" in filtered.index  # 剛好等於門檻，應保留


def test_apply_liquidity_filter_treats_missing_turnover_as_zero():
    df = pd.DataFrame({"avg_daily_turnover": [10_000_000.0, float("nan")]}, index=["LIQUID", "UNKNOWN"])
    filtered = apply_liquidity_filter(df, min_turnover=3_000_000)
    assert "UNKNOWN" not in filtered.index


# ---------------------------------------------------------------------------
# Edge Score
# ---------------------------------------------------------------------------

def test_compute_edge_score_weights_sum_to_one():
    assert sum(EDGE_SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_compute_edge_score_extreme_values_cap_at_100():
    df = pd.DataFrame({
        "z_score": [10.0],
        "volume_z_score": [10.0],
        "avg_daily_turnover": [1_000_000_000.0],
    }, index=["MAX.TW"])
    score = compute_edge_score(df)
    assert score["MAX.TW"] == pytest.approx(100.0)


def test_compute_edge_score_weak_signal_scores_low():
    df = pd.DataFrame({
        "z_score": [2.01],
        "volume_z_score": [-1.0],
        "avg_daily_turnover": [0.0],
    }, index=["WEAK.TW"])
    score = compute_edge_score(df)
    assert score["WEAK.TW"] < 30


def test_compute_edge_score_handles_missing_volume_and_liquidity():
    """成交量/流動性資料缺失時（例如抓取失敗）不應讓分數計算整個掛掉。"""
    df = pd.DataFrame({
        "z_score": [2.5],
        "volume_z_score": [float("nan")],
        "avg_daily_turnover": [float("nan")],
    }, index=["A.TW"])
    score = compute_edge_score(df)
    assert not pd.isna(score["A.TW"])
    assert score["A.TW"] > 0  # 至少統計極端度那部分仍有貢獻
