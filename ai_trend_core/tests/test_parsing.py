"""
tests/test_parsing.py
針對 agents/parsing.py 的單元測試：parse_strategist_output 是 CrewAI 真實對話輸出
與資料庫結構化欄位之間唯一的橋樑，summarize_reasoning 則是 UI／Telegram 警報
精簡摘要的來源。若這兩者在真實 LLM 輸出的各種變化格式下出錯，
會導致訊號靜默寫入失敗的欄位，或警報訊息顯示空白。
"""

from __future__ import annotations

import pytest

from agents.parsing import parse_strategist_output, summarize_reasoning


# ---------------------------------------------------------------------------
# parse_strategist_output
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# summarize_reasoning
# ---------------------------------------------------------------------------


def test_summarize_reasoning_extracts_text_after_label():
    text = (
        "Action: Sell\nEntry: 100\nTP: 90\nSL: 110\nConfidence: 0.9\n"
        "Reasoning: 價格顯著超出布林上軌，統計上處於極端值。新聞情緒同步轉空，兩者一致指向做空。"
    )
    summary = summarize_reasoning(text)
    assert "統計上處於極端值" in summary
    assert "Action:" not in summary


def test_summarize_reasoning_limits_to_max_sentences():
    text = "Reasoning: 第一句話。第二句話。第三句話不應出現。"
    summary = summarize_reasoning(text, max_sentences=2)
    assert "第一句話" in summary
    assert "第二句話" in summary
    assert "第三句話不應出現" not in summary


def test_summarize_reasoning_without_label_falls_back_to_whole_text():
    """若輸出沒有『Reasoning:』標籤（例如解析器攔不到格式跑掉的內容），仍應回傳可讀摘要而非空字串。"""
    summary = summarize_reasoning("這是一段沒有標籤的純文字說明。內容依然重要。")
    assert "這是一段沒有標籤的純文字說明" in summary


def test_summarize_reasoning_handles_none_and_empty():
    assert summarize_reasoning(None) == ""
    assert summarize_reasoning("") == ""


def test_summarize_reasoning_handles_text_without_sentence_terminator():
    """若文字沒有句號等結尾標點（例如被截斷），仍應原樣回傳而非拋錯或回傳空字串。"""
    summary = summarize_reasoning("Reasoning: 這段文字沒有結尾標點")
    assert summary == "這段文字沒有結尾標點"
