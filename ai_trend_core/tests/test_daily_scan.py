"""
tests/test_daily_scan.py
針對 daily_scan.py 的 build_anomaly_payload() 單元測試：確保異常標的的快照
能正確轉為可複製貼給 Claude Code 分析的結構化 JSON 資料。
"""

from __future__ import annotations

import pandas as pd

from core.quant_engine import MarketSnapshot
from daily_scan import build_anomaly_payload


def _snapshot(symbol: str, z: float, signal: str) -> MarketSnapshot:
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    history = pd.DataFrame({"close": [100.0] * 20}, index=idx)
    return MarketSnapshot(
        symbol=symbol, price=225.123, mean=210.456, upper_band=220.789,
        lower_band=200.111, std_dev=5.4321, z_score=z, signal=signal, history=history,
    )


def test_build_anomaly_payload_empty_list_returns_empty():
    assert build_anomaly_payload([]) == []


def test_build_anomaly_payload_includes_all_expected_fields():
    payload = build_anomaly_payload([_snapshot("NVDA", 2.3, "OVERBOUGHT")])
    assert len(payload) == 1
    entry = payload[0]
    assert entry["symbol"] == "NVDA"
    assert entry["signal"] == "OVERBOUGHT"
    assert entry["price"] == 225.12
    assert entry["mean"] == 210.46
    assert entry["upper_band"] == 220.79
    assert entry["lower_band"] == 200.11
    assert entry["z_score"] == 2.3


def test_build_anomaly_payload_preserves_order_and_handles_multiple():
    payload = build_anomaly_payload([
        _snapshot("BTC-USD", 2.5, "OVERBOUGHT"),
        _snapshot("TSLA", -2.1, "OVERSOLD"),
    ])
    assert [p["symbol"] for p in payload] == ["BTC-USD", "TSLA"]
    assert payload[1]["signal"] == "OVERSOLD"


def test_build_anomaly_payload_is_json_serializable():
    import json

    payload = build_anomaly_payload([_snapshot("AAPL", 2.0, "OVERBOUGHT")])
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "AAPL" in serialized
