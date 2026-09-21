"""
tests/test_quant_engine.py
針對 core/quant_engine.py 向量化計算的單元測試：確保對「整個寬表」
一次計算出的 Z-Score，與逐檔計算的結果一致，且能正確篩出異常標的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quant_engine import BB_LENGTH, compute_latest_zscores, get_anomalies


def _wide_prices(columns: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    idx = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.DataFrame(columns, index=idx)


def test_compute_latest_zscores_matches_manual_calculation_per_ticker():
    prices_a = [100.0] * 25
    prices_b = [100.0] * 20 + [140.0, 145.0, 150.0, 155.0, 160.0]
    df = _wide_prices({"A.TW": prices_a, "B.TW": prices_b})

    latest = compute_latest_zscores(df)

    window_b = pd.Series(prices_b[-BB_LENGTH:])
    expected_mean_b = window_b.mean()
    expected_std_b = window_b.std()
    expected_z_b = (prices_b[-1] - expected_mean_b) / expected_std_b

    assert latest.loc["B.TW", "z_score"] == pytest.approx(expected_z_b)
    assert latest.loc["B.TW", "signal"] == "OVERBOUGHT"
    assert latest.loc["A.TW", "signal"] == "NEUTRAL"


def test_compute_latest_zscores_drops_tickers_with_insufficient_history():
    """新股歷史不足 20 個交易日時，應從結果中排除，而不是回傳 NaN 污染下游篩選。"""
    df = _wide_prices({
        "OLD.TW": [100.0] * 25,
        "NEW.TW": [100.0] * 5 + [np.nan] * 20,
    })
    latest = compute_latest_zscores(df)
    assert "OLD.TW" in latest.index
    assert "NEW.TW" not in latest.index


def test_compute_latest_zscores_oversold_case():
    prices = [100.0] * 20 + [60.0, 55.0, 50.0, 45.0, 40.0]
    df = _wide_prices({"C.TW": prices})
    latest = compute_latest_zscores(df)
    assert latest.loc["C.TW", "z_score"] < -2.0
    assert latest.loc["C.TW", "signal"] == "OVERSOLD"


def test_get_anomalies_filters_and_sorts_by_absolute_zscore():
    df = _wide_prices({
        "NEUTRAL.TW": [100.0] * 25,
        "MILD.TW": [100.0] * 20 + [140.0, 145.0, 150.0, 155.0, 160.0],
        "EXTREME.TW": [100.0] * 20 + [200.0, 250.0, 300.0, 350.0, 400.0],
    })
    latest = compute_latest_zscores(df)
    anomalies = get_anomalies(latest, threshold=2.0)

    assert "NEUTRAL.TW" not in anomalies.index
    assert list(anomalies.index) == ["EXTREME.TW", "MILD.TW"]


def test_get_anomalies_empty_when_all_neutral():
    df = _wide_prices({"A.TW": [100.0] * 25, "B.TW": [50.0] * 25})
    latest = compute_latest_zscores(df)
    anomalies = get_anomalies(latest)
    assert anomalies.empty


def test_compute_latest_zscores_is_vectorized_equivalent_to_per_column_loop():
    """驗證整張寬表一次計算的結果，與逐欄位分別計算完全一致（向量化正確性）。"""
    rng = np.random.default_rng(7)
    columns = {f"T{i}.TW": list(100 + rng.normal(0, 3, size=30)) for i in range(5)}
    df = _wide_prices(columns)

    latest = compute_latest_zscores(df)

    for ticker in columns:
        window = df[ticker].iloc[-BB_LENGTH:]
        expected_z = (df[ticker].iloc[-1] - window.mean()) / window.std()
        assert latest.loc[ticker, "z_score"] == pytest.approx(expected_z)
