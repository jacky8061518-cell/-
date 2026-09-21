"""
tests/test_main_loop.py
針對 main_loop.py 的 handle_anomaly() 單元測試。這個函式是 main_loop.py
的 24/7 迴圈與 manual_run.py 的手動單次執行共用的核心邏輯（分析、寫入
資料庫、視信心值推播 Telegram），必須確保兩邊共用時行為一致。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import main_loop
from core import database
from core.quant_engine import MarketSnapshot


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_signals.db")
    database.init_db()
    yield


def _snapshot(symbol: str = "NVDA", signal: str = "OVERBOUGHT", z: float = 2.3) -> MarketSnapshot:
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    history = pd.DataFrame({"close": [100.0] * 20}, index=idx)
    return MarketSnapshot(
        symbol=symbol, price=225.0, mean=210.0, upper_band=220.0,
        lower_band=200.0, std_dev=5.0, z_score=z, signal=signal, history=history,
    )


def test_handle_anomaly_persists_result_and_returns_it():
    fake_result = {
        "symbol": "NVDA", "action": "Sell", "entry": 225.0, "take_profit": 210.0,
        "stop_loss": 235.0, "confidence": 0.7,
        "final_report": "Action: Sell\nReasoning: 測試理由",
        "sentiment_report": "測試情緒報告",
    }
    with patch("agents.trading_crew.analyze_opportunity", return_value=fake_result) as mock_analyze:
        result = main_loop.handle_anomaly(_snapshot())

    mock_analyze.assert_called_once()
    assert result == fake_result

    rows = database.get_recent_signals(limit=10)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["action"] == "Sell"
    assert rows[0]["confidence"] == pytest.approx(0.7)


def test_handle_anomaly_returns_none_and_skips_db_write_on_analysis_failure():
    with patch("agents.trading_crew.analyze_opportunity", side_effect=RuntimeError("LLM 呼叫失敗")):
        result = main_loop.handle_anomaly(_snapshot())

    assert result is None
    assert database.get_recent_signals(limit=10) == []


def test_handle_anomaly_sends_telegram_when_confidence_above_threshold():
    fake_result = {
        "symbol": "NVDA", "action": "Buy", "entry": 100.0, "take_profit": 110.0,
        "stop_loss": 90.0, "confidence": 0.95, "final_report": "Reasoning: 高信心測試",
        "sentiment_report": "",
    }
    with patch("agents.trading_crew.analyze_opportunity", return_value=fake_result), \
         patch("core.notifier.send_telegram_message", return_value=True) as mock_send:
        main_loop.handle_anomaly(_snapshot())

    mock_send.assert_called_once()


def test_handle_anomaly_skips_telegram_when_confidence_below_threshold():
    fake_result = {
        "symbol": "NVDA", "action": "Hold", "entry": None, "take_profit": None,
        "stop_loss": None, "confidence": 0.4, "final_report": "Reasoning: 低信心測試",
        "sentiment_report": "",
    }
    with patch("agents.trading_crew.analyze_opportunity", return_value=fake_result), \
         patch("core.notifier.send_telegram_message") as mock_send:
        main_loop.handle_anomaly(_snapshot())

    mock_send.assert_not_called()


def test_handle_anomaly_skips_telegram_when_confidence_missing():
    """CrewAI 解析器抓不到 Confidence 欄位時應回傳 None，此時不該嘗試推播。"""
    fake_result = {
        "symbol": "NVDA", "action": "Hold", "entry": None, "take_profit": None,
        "stop_loss": None, "confidence": None, "final_report": "格式跑掉的輸出",
        "sentiment_report": "",
    }
    with patch("agents.trading_crew.analyze_opportunity", return_value=fake_result), \
         patch("core.notifier.send_telegram_message") as mock_send:
        result = main_loop.handle_anomaly(_snapshot())

    assert result == fake_result
    mock_send.assert_not_called()
