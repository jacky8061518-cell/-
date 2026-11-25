"""
tests/test_regime.py
大盤體制判別（top-down 市場過濾器）的單元測試：加權指數乖離 + 市場寬度
的組合判斷邏輯，以及套用到異常清單上的 buy_eligible / short_candidate 標記。
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.regime import (
    RISK_OFF_OVERSOLD_RATIO,
    RISK_OFF_TAIEX_DEVIATION,
    apply_regime_policy,
    compute_market_regime,
)


def _full_latest(n_oversold: int, n_overbought: int, n_neutral: int = 0) -> pd.DataFrame:
    signals = ["OVERSOLD"] * n_oversold + ["OVERBOUGHT"] * n_overbought + ["NEUTRAL"] * n_neutral
    return pd.DataFrame({"signal": signals}, index=[f"T{i}" for i in range(len(signals))])


def _taiex_series(ma20_level: float, latest_price: float) -> pd.Series:
    """建一條前 19 天都在 ma20_level、最後一天跳到 latest_price 的序列，
    讓 20 期均線約等於 ma20_level。"""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [ma20_level] * 19 + [latest_price]
    return pd.Series(values, index=idx)


def _taiex_series_with_exact_deviation(base: float, deviation: float) -> pd.Series:
    """建一條序列，使其 20 期均線相對最後一天收盤價的乖離「精確」等於 deviation。

    20 期均線本身就包含最後一天，所以不能直接把前 19 天設成目標均線再接上
    目標價——那樣真正算出來的均線會被最後一天拉偏。這裡直接解方程式：
    令前 19 天 = base，最後一天 = p，則 ma = (19*base + p) / 20，
    要 (p - ma) / ma = deviation，解出 p = 19*base*(1+deviation) / (19-deviation)。
    """
    p = 19 * base * (1 + deviation) / (19 - deviation)
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [base] * 19 + [p]
    return pd.Series(values, index=idx)


def test_risk_off_when_taiex_and_breadth_both_bearish():
    full_latest = _full_latest(n_oversold=90, n_overbought=10)
    taiex = _taiex_series(ma20_level=20000, latest_price=20000 * (1 + RISK_OFF_TAIEX_DEVIATION - 0.01))
    regime = compute_market_regime(full_latest, taiex)
    assert regime.classification == "RISK_OFF"
    assert regime.oversold_ratio == pytest.approx(0.9)


def test_risk_on_when_taiex_above_ma_and_breadth_balanced():
    full_latest = _full_latest(n_oversold=40, n_overbought=60)
    taiex = _taiex_series(ma20_level=20000, latest_price=20400)
    regime = compute_market_regime(full_latest, taiex)
    assert regime.classification == "RISK_ON"


def test_neutral_when_signals_disagree():
    """寬度極端偏空，但大盤指數其實站在均線之上——訊號矛盾，保守給 NEUTRAL。"""
    full_latest = _full_latest(n_oversold=90, n_overbought=10)
    taiex = _taiex_series(ma20_level=20000, latest_price=20400)
    regime = compute_market_regime(full_latest, taiex)
    assert regime.classification == "NEUTRAL"


def test_risk_off_from_breadth_alone_when_taiex_missing():
    """抓不到大盤指數時（例如 API 限流），寬度單獨極端偏空仍保守判為 RISK_OFF。"""
    full_latest = _full_latest(n_oversold=95, n_overbought=5)
    regime = compute_market_regime(full_latest, taiex_history=None)
    assert regime.classification == "RISK_OFF"


def test_oversold_ratio_none_when_no_anomalies():
    full_latest = _full_latest(n_oversold=0, n_overbought=0, n_neutral=50)
    regime = compute_market_regime(full_latest, taiex_history=None)
    assert regime.oversold_ratio is None
    assert regime.classification == "NEUTRAL"


def test_boundary_exactly_at_threshold_counts_as_bearish():
    """剛好等於門檻（>=）應視為達標，不是嚴格大於才算。"""
    full_latest = _full_latest(n_oversold=70, n_overbought=30)
    assert full_latest["signal"].eq("OVERSOLD").mean() == pytest.approx(RISK_OFF_OVERSOLD_RATIO)
    taiex = _taiex_series_with_exact_deviation(base=20000, deviation=RISK_OFF_TAIEX_DEVIATION)
    regime = compute_market_regime(full_latest, taiex)
    assert regime.classification == "RISK_OFF"


def test_apply_regime_policy_blocks_buy_in_risk_off():
    enriched = pd.DataFrame({"signal": ["OVERSOLD", "OVERBOUGHT", "OVERSOLD"]}, index=["A", "B", "C"])
    full_latest = _full_latest(n_oversold=90, n_overbought=10)
    taiex = _taiex_series(ma20_level=20000, latest_price=20000 * (1 + RISK_OFF_TAIEX_DEVIATION - 0.01))
    regime = compute_market_regime(full_latest, taiex)
    result = apply_regime_policy(enriched, regime)
    assert list(result["buy_eligible"]) == [False, True, False]
    assert list(result["short_candidate"]) == [False, True, False]


def test_apply_regime_policy_no_restrictions_in_risk_on():
    enriched = pd.DataFrame({"signal": ["OVERSOLD", "OVERBOUGHT"]}, index=["A", "B"])
    full_latest = _full_latest(n_oversold=40, n_overbought=60)
    taiex = _taiex_series(ma20_level=20000, latest_price=20400)
    regime = compute_market_regime(full_latest, taiex)
    result = apply_regime_policy(enriched, regime)
    assert result["buy_eligible"].all()
    assert not result["short_candidate"].any()
