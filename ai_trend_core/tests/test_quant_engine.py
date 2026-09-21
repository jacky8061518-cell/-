"""
tests/test_quant_engine.py
針對 core/quant_engine.py 的單元測試，涵蓋正常情況與極端情況
（停牌 / 數據缺失 / API 斷線），確保 24 小時運行時不會因單一標的失敗而崩潰。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.quant_engine import (
    BB_LENGTH,
    MarketSnapshot,
    compute_bollinger_and_zscore,
    detect_signal,
    fetch_hourly_data,
    get_anomalies,
    scan_market,
    scan_symbol,
)


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="h")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_bollinger_and_zscore / detect_signal
# ---------------------------------------------------------------------------


def test_compute_bollinger_and_zscore_normal_case():
    prices = [100.0] * 55 + [140.0, 145.0, 150.0, 155.0, 160.0]
    df = compute_bollinger_and_zscore(_make_ohlcv(prices))
    latest = df.iloc[-1]

    # 最後 20 根 K 線 = 15 根 100.0 + [140, 145, 150, 155, 160]
    assert latest["bb_mean"] == pytest.approx(112.5, rel=1e-3)
    assert latest["z_score"] > 2.0
    assert detect_signal(latest["z_score"]) == "OVERBOUGHT"


def test_compute_bollinger_and_zscore_oversold_case():
    prices = [100.0] * 55 + [60.0, 55.0, 50.0, 45.0, 40.0]
    df = compute_bollinger_and_zscore(_make_ohlcv(prices))
    latest = df.iloc[-1]

    assert latest["z_score"] < -2.0
    assert detect_signal(latest["z_score"]) == "OVERSOLD"


def test_compute_bollinger_and_zscore_neutral_case():
    rng = np.random.default_rng(42)
    prices = 100 + rng.normal(0, 0.5, size=40)
    df = compute_bollinger_and_zscore(_make_ohlcv(list(prices)))
    latest = df.iloc[-1]

    assert detect_signal(latest["z_score"]) == "NEUTRAL"


def test_compute_bollinger_and_zscore_insufficient_history_raises():
    """數據長度不足 20 根 K 線（例如新標的、停牌後資料不全）應明確拋出錯誤，而非靜默失敗。"""
    df = _make_ohlcv([100.0] * (BB_LENGTH - 1))
    with pytest.raises(RuntimeError):
        compute_bollinger_and_zscore(df)


def test_detect_signal_handles_nan_from_flat_prices():
    """價格完全持平（標準差為 0）會使 Z-Score 出現 0/0 = NaN，應視為中性而非拋出例外。"""
    df = compute_bollinger_and_zscore(_make_ohlcv([100.0] * 40))
    latest_z = df.iloc[-1]["z_score"]
    assert pd.isna(latest_z)
    assert detect_signal(latest_z) == "NEUTRAL"


def test_detect_signal_boundary_values():
    assert detect_signal(2.0) == "OVERBOUGHT"
    assert detect_signal(-2.0) == "OVERSOLD"
    assert detect_signal(1.999) == "NEUTRAL"
    assert detect_signal(-1.999) == "NEUTRAL"


# ---------------------------------------------------------------------------
# fetch_hourly_data：停牌 / 數據缺失 / API 斷線
# ---------------------------------------------------------------------------


def test_fetch_hourly_data_raises_on_empty_dataframe():
    """標的停牌或代碼錯誤時，yfinance 會回傳空的 DataFrame，應轉為明確的 ValueError。"""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("core.quant_engine.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError):
            fetch_hourly_data("DELISTED")


def test_fetch_hourly_data_propagates_network_error():
    """API 斷線 / 逾時等例外應直接往外拋，由呼叫端（scan_market）決定如何處理。"""
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = ConnectionError("network unreachable")
    with patch("core.quant_engine.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ConnectionError):
            fetch_hourly_data("BTC-USD")


# ---------------------------------------------------------------------------
# scan_symbol / scan_market：單一標的失敗不應中斷整體掃描
# ---------------------------------------------------------------------------


def test_scan_symbol_success():
    prices = [100.0] * 55 + [140.0, 145.0, 150.0, 155.0, 160.0]
    with patch("core.quant_engine.fetch_hourly_data", return_value=_make_ohlcv(prices)):
        snapshot = scan_symbol("BTC-USD")

    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.symbol == "BTC-USD"
    assert snapshot.signal == "OVERBOUGHT"


def test_scan_market_skips_failed_symbol_and_keeps_others():
    """其中一個標的因停牌/斷線失敗時，其餘標的仍應正常回傳（不可整批崩潰）。"""
    good_prices = [100.0] * 55 + [140.0, 145.0, 150.0, 155.0, 160.0]

    def fake_fetch(symbol: str, period: str = "60d"):
        if symbol == "BROKEN":
            raise ConnectionError("simulated API outage")
        return _make_ohlcv(good_prices)

    with patch("core.quant_engine.fetch_hourly_data", side_effect=fake_fetch):
        snapshots = scan_market(["BROKEN", "BTC-USD"])

    assert "BROKEN" not in snapshots
    assert "BTC-USD" in snapshots
    assert snapshots["BTC-USD"].signal == "OVERBOUGHT"


def test_scan_market_all_symbols_fail_returns_empty_dict():
    with patch("core.quant_engine.fetch_hourly_data", side_effect=ConnectionError("down")):
        snapshots = scan_market(["A", "B"])
    assert snapshots == {}


# ---------------------------------------------------------------------------
# get_anomalies
# ---------------------------------------------------------------------------


def _snapshot(symbol: str, signal: str, z: float = 0.0) -> MarketSnapshot:
    df = _make_ohlcv([100.0] * 20)
    return MarketSnapshot(
        symbol=symbol, price=100.0, mean=100.0, upper_band=110.0,
        lower_band=90.0, std_dev=5.0, z_score=z, signal=signal, history=df,
    )


def test_get_anomalies_filters_neutral_out():
    snapshots = {
        "A": _snapshot("A", "NEUTRAL"),
        "B": _snapshot("B", "OVERBOUGHT", z=2.5),
        "C": _snapshot("C", "OVERSOLD", z=-2.5),
    }
    anomalies = get_anomalies(snapshots)
    symbols = {snap.symbol for snap in anomalies}
    assert symbols == {"B", "C"}


def test_get_anomalies_empty_when_all_neutral():
    snapshots = {"A": _snapshot("A", "NEUTRAL")}
    assert get_anomalies(snapshots) == []
