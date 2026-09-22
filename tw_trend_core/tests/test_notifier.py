"""
tests/test_notifier.py
針對 core/notifier.py 的單元測試：確保只有信心評分超過門檻的訊號才會
觸發 Telegram 推播，且未設定憑證或網路失敗時能優雅降級。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from core.notifier import (
    TELEGRAM_CONFIDENCE_THRESHOLD,
    format_signal_alert,
    notify_if_high_confidence,
    send_telegram_message,
)


@pytest.fixture(autouse=True)
def clear_telegram_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


def _signal(confidence: float, **overrides):
    base = dict(
        code="3036", name="文曄", price=205.5, z_score=2.082,
        action="Buy", confidence=confidence, take_profit=280.0, stop_loss=184.0,
        core_reason="測試理由摘要",
    )
    base.update(overrides)
    return base


def test_notify_if_high_confidence_skips_when_below_threshold(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    with patch("core.notifier.requests.post") as mock_post:
        result = notify_if_high_confidence(_signal(0.72))
    assert result is False
    mock_post.assert_not_called()


def test_notify_if_high_confidence_skips_at_exact_threshold(monkeypatch):
    """信心值剛好等於門檻（0.80）不應觸發，與 ai_trend_core 的「> 門檻」語意一致。"""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    with patch("core.notifier.requests.post") as mock_post:
        result = notify_if_high_confidence(_signal(TELEGRAM_CONFIDENCE_THRESHOLD))
    assert result is False
    mock_post.assert_not_called()


def test_notify_if_high_confidence_sends_when_above_threshold(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    with patch("core.notifier.requests.post", return_value=mock_response) as mock_post:
        result = notify_if_high_confidence(_signal(0.85))
    assert result is True
    mock_post.assert_called_once()
    body = mock_post.call_args.kwargs["json"]
    assert body["chat_id"] == "12345"
    assert "文曄" in body["text"]
    assert "3036" in body["text"]


def test_notify_if_high_confidence_handles_missing_confidence():
    result = notify_if_high_confidence(_signal(None))
    assert result is False


def test_notify_if_high_confidence_skips_when_credentials_missing():
    """憑證未設定時即使信心值夠高，也應優雅跳過而非拋出例外。"""
    result = notify_if_high_confidence(_signal(0.9))
    assert result is False


def test_send_telegram_message_handles_network_failure(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    with patch("core.notifier.requests.post", side_effect=requests.ConnectionError("down")):
        result = send_telegram_message("測試訊息")
    assert result is False


def test_format_signal_alert_includes_code_and_name():
    message = format_signal_alert(
        code="3036", name="文曄", price=205.5, z_score=2.082,
        action="Buy", confidence=0.72, take_profit=280.0, stop_loss=184.0,
        reasoning_summary="測試理由",
    )
    assert "文曄" in message
    assert "3036" in message
    assert "72%" in message
    assert "測試理由" in message
