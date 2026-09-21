"""
tests/test_notifier.py
針對 core/notifier.py 的單元測試：確保 Telegram 推播在未設定憑證、
API 呼叫失敗等情況下能優雅降級，不會讓 main_loop.py 的主流程中斷。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from core.notifier import format_signal_alert, send_telegram_message


@pytest.fixture(autouse=True)
def clear_telegram_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


def test_send_telegram_message_skips_when_credentials_missing():
    assert send_telegram_message("測試訊息") is False


def test_send_telegram_message_skips_when_only_token_set(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    assert send_telegram_message("測試訊息") is False


def test_send_telegram_message_success(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch("core.notifier.requests.post", return_value=mock_response) as mock_post:
        result = send_telegram_message("測試訊息")

    assert result is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["chat_id"] == "12345"
    assert call_kwargs["json"]["text"] == "測試訊息"


def test_send_telegram_message_handles_network_failure(monkeypatch):
    """API 逾時 / 斷線等情況應被捕捉並回傳 False，而不是讓整個 main_loop 崩潰。"""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    with patch("core.notifier.requests.post", side_effect=requests.ConnectionError("down")):
        result = send_telegram_message("測試訊息")

    assert result is False


def test_send_telegram_message_handles_http_error_status(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

    with patch("core.notifier.requests.post", return_value=mock_response):
        result = send_telegram_message("測試訊息")

    assert result is False


def test_format_signal_alert_includes_all_fields():
    message = format_signal_alert(
        symbol="BTC-USD", price=68000.1234, z_score=2.4321, action="Sell",
        confidence=0.86, take_profit=64000.0, stop_loss=70000.0,
        reasoning_summary="Z-Score 顯著偏離，新聞情緒偏空。",
    )
    assert "BTC-USD" in message
    assert "68000.12" in message
    assert "+2.43" in message
    assert "Sell" in message
    assert "86%" in message
    assert "64000.00" in message
    assert "70000.00" in message
    assert "Z-Score 顯著偏離" in message


def test_format_signal_alert_omits_optional_fields_when_none():
    message = format_signal_alert(
        symbol="NVDA", price=950.0, z_score=2.1, action="Hold", confidence=0.35,
    )
    assert "停利" not in message
    assert "停損" not in message
    assert "理由" not in message
