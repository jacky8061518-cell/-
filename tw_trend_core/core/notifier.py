"""
core/notifier.py
Telegram 即時推播模組：當首席策略官對台股標的產生高信心分析時，
主動通知使用者的 Telegram。與 ai_trend_core/core/notifier.py 邏輯相同
（共用同一組 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，同一個 Telegram Bot），
獨立成一份檔案是為了避免兩個專案的 core 套件互相匯入造成命名衝突，
而非邏輯上的差異。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 10

# 信心評分超過此門檻時，主動透過 Telegram 推播警報
TELEGRAM_CONFIDENCE_THRESHOLD = 0.8


def _get_credentials() -> tuple[str, str] | None:
    """讀取 Telegram Bot 憑證，未設定時回傳 None（由呼叫端決定如何降級處理）。"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    return token, chat_id


def send_telegram_message(text: str) -> bool:
    """透過 Telegram Bot API 傳送訊息。未設定憑證或傳送失敗時記錄警告並回傳 False，不中斷主流程。"""
    credentials = _get_credentials()
    if credentials is None:
        logger.warning("尚未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，略過 Telegram 推播。")
        return False

    token, chat_id = credentials
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Telegram 推播失敗：%s", exc)
        return False
    return True


def format_signal_alert(
    code: str,
    name: str,
    price: float,
    z_score: float,
    action: str,
    confidence: float,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    reasoning_summary: str = "",
) -> str:
    """組成 Telegram 警報訊息內容：標的代碼／名稱、價格、Z-Score、建議動作、停損停利與 AI 理由摘要。"""
    lines = [
        "🚨 <b>TW Trend Core 高信心訊號</b>",
        f"標的：<b>{name}（{code}）</b>",
        f"目前價格：{price:.2f}",
        f"Z-Score：{z_score:+.2f}",
        f"建議動作：<b>{action}</b>",
        f"信心評分：{confidence:.0%}",
    ]
    if take_profit is not None:
        lines.append(f"停利：{take_profit:.2f}")
    if stop_loss is not None:
        lines.append(f"停損：{stop_loss:.2f}")
    if reasoning_summary:
        lines.append(f"理由：{reasoning_summary}")
    return "\n".join(lines)


def notify_if_high_confidence(
    signal: dict[str, Any], threshold: float = TELEGRAM_CONFIDENCE_THRESHOLD,
) -> bool:
    """若訊號信心評分超過門檻，組成警報並推播；否則不動作。回傳是否已送出推播。

    `signal` 為單一分析結果的 dict，需至少包含 code / name / price / z_score /
    action / confidence 欄位（與 daily_scan.py、dashboard_artifact.html 使用的
    signals 文件格式一致），take_profit / stop_loss / core_reason 為選填。
    """
    confidence = signal.get("confidence")
    if confidence is None or confidence <= threshold:
        return False

    message = format_signal_alert(
        code=signal["code"],
        name=signal["name"],
        price=signal["price"],
        z_score=signal["z_score"],
        action=signal.get("action") or "N/A",
        confidence=confidence,
        take_profit=signal.get("take_profit"),
        stop_loss=signal.get("stop_loss"),
        reasoning_summary=signal.get("core_reason") or "",
    )
    return send_telegram_message(message)
