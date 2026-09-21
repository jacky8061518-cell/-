"""
agents/parsing.py
純文字解析工具：從 CrewAI 首席策略官的輸出中萃取結構化欄位與精簡摘要。
刻意不依賴 crewai / langchain_anthropic，讓不需要真正呼叫 LLM 的呼叫端
（例如 Streamlit 儀表板）也能輕量匯入使用。
"""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s*")


def parse_strategist_output(text: str) -> dict[str, Any]:
    """從首席策略官的輸出文字中解析出結構化欄位。"""
    patterns = {
        "action": r"Action:\s*(\w+)",
        "entry": r"Entry:\s*([\d.,]+)",
        "take_profit": r"TP:\s*([\d.,]+)",
        "stop_loss": r"SL:\s*([\d.,]+)",
        "confidence": r"Confidence:\s*([\d.]+)",
        "reasoning": r"Reasoning:\s*(.+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            parsed[key] = None
            continue
        value = match.group(1).strip()
        if key in {"entry", "take_profit", "stop_loss", "confidence"}:
            try:
                parsed[key] = float(value.replace(",", ""))
            except ValueError:
                parsed[key] = None
        else:
            parsed[key] = value
    return parsed


def summarize_reasoning(text: str | None, max_sentences: int = 2) -> str:
    """從完整的 AI 輸出中萃取『Reasoning』之後最關鍵的前幾句話，供 UI 精簡顯示。"""
    if not text:
        return ""

    match = re.search(r"Reasoning:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    body = match.group(1).strip() if match else text.strip()

    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(body) if s.strip()]
    if not sentences:
        return body
    return "".join(sentences[:max_sentences])
