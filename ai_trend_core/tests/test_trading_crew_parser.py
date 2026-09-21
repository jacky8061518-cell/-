"""
tests/test_trading_crew_parser.py
針對 agents/trading_crew.py 中 parse_strategist_output 解析器的單元測試。
這是 CrewAI 真實對話輸出與資料庫結構化欄位之間唯一的橋樑，
若解析器在真實 LLM 輸出的各種變化格式下出錯，會導致訊號靜默寫入失敗的欄位。
"""

from __future__ import annotations

import pytest

from agents.trading_crew import parse_strategist_output


def test_parse_full_well_formed_output():
    text = (
        "Action: Sell\n"
        "Entry: 160.5\n"
        "TP: 150.0\n"
        "SL: 165.0\n"
        "Confidence: 0.82\n"
        "Reasoning: 價格顯著高於均值，且新聞情緒偏空，建議放空。"
    )
    parsed = parse_strategist_output(text)
    assert parsed["action"] == "Sell"
    assert parsed["entry"] == pytest.approx(160.5)
    assert parsed["take_profit"] == pytest.approx(150.0)
    assert parsed["stop_loss"] == pytest.approx(165.0)
    assert parsed["confidence"] == pytest.approx(0.82)
    assert "放空" in parsed["reasoning"]


def test_parse_handles_thousands_separator_in_prices():
    """BTC 等高價標的常見以千分位逗號表示，例如 Entry: 65,000.5。"""
    text = "Action: Buy\nEntry: 65,000.5\nTP: 68,000\nSL: 62,500\nConfidence: 0.9\nReasoning: 測試"
    parsed = parse_strategist_output(text)
    assert parsed["entry"] == pytest.approx(65000.5)
    assert parsed["take_profit"] == pytest.approx(68000)
    assert parsed["stop_loss"] == pytest.approx(62500)


def test_parse_is_case_insensitive_for_labels():
    text = "action: hold\nentry: 100\ntp: 110\nsl: 90\nconfidence: 0.5\nreasoning: 盤整格局"
    parsed = parse_strategist_output(text)
    assert parsed["action"] == "hold"
    assert parsed["confidence"] == pytest.approx(0.5)


def test_parse_missing_fields_returns_none_instead_of_raising():
    """真實 LLM 有時會漏掉部分欄位或格式跑掉，解析器不應拋出例外，而是回傳 None 讓上層決定如何處理。"""
    text = "這次分析比較模糊，Action: Hold，但沒有給出明確的進出場價位。"
    parsed = parse_strategist_output(text)
    assert parsed["action"] == "Hold" or parsed["action"] is None
    assert parsed["entry"] is None
    assert parsed["take_profit"] is None
    assert parsed["stop_loss"] is None
    assert parsed["confidence"] is None


def test_parse_non_numeric_confidence_value_is_none():
    """若 LLM 誤把信心評分寫成非數字（例如「高」），應優雅地回傳 None 而非拋錯。"""
    text = "Action: Buy\nEntry: 100\nTP: 110\nSL: 90\nConfidence: 很高\nReasoning: 測試"
    parsed = parse_strategist_output(text)
    assert parsed["confidence"] is None


def test_parse_empty_string_returns_all_none():
    parsed = parse_strategist_output("")
    assert all(value is None for value in parsed.values())


def test_parse_reasoning_captures_multiline_text():
    text = "Action: Buy\nEntry: 100\nTP: 110\nSL: 90\nConfidence: 0.7\nReasoning: 第一行理由\n第二行補充說明"
    parsed = parse_strategist_output(text)
    assert "第一行理由" in parsed["reasoning"]
    assert "第二行補充說明" in parsed["reasoning"]
