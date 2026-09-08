"""Tests for the semantic layer's guard rails.

The dedup and citation tests are the important ones: they are what stop wire
copy from being counted as independent confirmation, and what stops a language
model from inventing a fact the pipeline then sizes a position on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_desk.news import (
    DUPLICATE_THRESHOLD,
    LexiconAnnotator,
    LLMAnnotator,
    NewsItem,
    aggregate_sentiment,
    deduplicate,
    hamming,
    simhash,
)

NOW = datetime(2026, 7, 30, 1, 0, tzinfo=timezone.utc)


def item(event_id, headline, summary="", symbol="2330.TW", minutes=0) -> NewsItem:
    return NewsItem(
        event_id=event_id,
        symbol=symbol,
        headline=headline,
        summary=summary,
        publisher="測試",
        published_at=NOW + timedelta(minutes=minutes),
    )


def test_simhash_separates_rewrites_from_different_stories():
    original = item("a", "台積電法說會上修全年營收展望", "管理層表示 AI 需求超預期")
    rewrite = item("b", "台積電法說上修全年營收展望", "管理層表示AI需求超預期")
    unrelated = item("c", "台積電宣布美國新廠擴產計畫", "資本支出上調")
    assert hamming(simhash(original.text), simhash(rewrite.text)) <= DUPLICATE_THRESHOLD
    assert hamming(simhash(original.text), simhash(unrelated.text)) > DUPLICATE_THRESHOLD


def test_syndicated_copy_does_not_count_as_fresh_news():
    items = [
        item("a", "台積電法說會上修全年營收展望", "管理層表示 AI 需求超預期"),
        item("b", "台積電法說上修全年營收展望", "管理層表示AI需求超預期", minutes=5),
        item("c", "台積電宣布美國新廠擴產計畫", "資本支出上調", minutes=9),
    ]
    kept, novelty = deduplicate(items)
    assert novelty["a"] == 1.0
    assert novelty["b"] < 1.0, "the second outlet is not a second fact"
    assert novelty["c"] == 1.0
    assert {entry.event_id for entry in kept} == {"a", "c"}


def test_duplicates_from_different_symbols_are_not_merged():
    items = [
        item("a", "法說會上修全年營收展望", symbol="2330.TW"),
        item("b", "法說會上修全年營收展望", symbol="2454.TW", minutes=1),
    ]
    _, novelty = deduplicate(items)
    assert novelty["a"] == novelty["b"] == 1.0


def test_stale_duplicates_outside_the_window_stay_novel():
    items = [
        item("a", "台積電法說會上修全年營收展望"),
        item("b", "台積電法說會上修全年營收展望", minutes=60 * 48),
    ]
    _, novelty = deduplicate(items)
    assert novelty["b"] == 1.0


def test_lexicon_annotator_cites_a_verbatim_span():
    annotations = LexiconAnnotator().annotate([item("a", "台積電上修全年營收展望")], {"a": 1.0})
    assert annotations
    assert annotations[0].evidence_span in "台積電上修全年營收展望"
    assert annotations[0].polarity > 0


def test_lexicon_annotator_handles_negation():
    positive = LexiconAnnotator().annotate([item("a", "分析師調降目標價")], {"a": 1.0})[0]
    negated = LexiconAnnotator().annotate([item("b", "分析師不調降目標價")], {"b": 1.0})[0]
    assert positive.polarity < 0
    assert negated.polarity > positive.polarity


def test_annotations_without_any_signal_word_are_dropped():
    assert LexiconAnnotator().annotate([item("a", "公司今日召開股東常會")], {"a": 1.0}) == []


def test_an_llm_span_that_is_not_in_the_source_is_rejected():
    """The single most important guard: a model that cannot cite is discarded."""
    annotator = LLMAnnotator(
        call=lambda entry: {
            "evidence_span": "這句話原文裡並不存在",
            "polarity": 0.9,
            "horizon": "days",
        }
    )
    assert annotator.annotate([item("a", "台積電上修展望")], {"a": 1.0}) == []
    assert "引用片段" in annotator.rejected[0]


def test_an_llm_symbol_outside_the_security_master_is_rejected():
    annotator = LLMAnnotator(
        call=lambda entry: {
            "symbol": "9999.TW",
            "evidence_span": "上修",
            "polarity": 0.5,
            "horizon": "days",
        },
        known_symbols=frozenset({"2330.TW"}),
    )
    assert annotator.annotate([item("a", "台積電上修展望")], {"a": 1.0}) == []
    assert "security master" in annotator.rejected[0]


def test_a_valid_llm_annotation_passes_through():
    annotator = LLMAnnotator(
        call=lambda entry: {
            "evidence_span": "上修",
            "polarity": 0.8,
            "materiality": 0.9,
            "event_type": "earnings",
            "horizon": "days",
        },
        known_symbols=frozenset({"2330.TW"}),
    )
    result = annotator.annotate([item("a", "台積電上修展望")], {"a": 1.0})
    assert len(result) == 1
    assert result[0].annotator == "llm"
    assert not annotator.rejected


def test_an_llm_that_raises_does_not_break_the_batch():
    def explode(entry):
        raise TimeoutError("模型逾時")

    annotator = LLMAnnotator(call=explode)
    assert annotator.annotate([item("a", "台積電上修展望")], {"a": 1.0}) == []
    assert "呼叫失敗" in annotator.rejected[0]


def test_many_weak_articles_cannot_manufacture_a_strong_reading():
    """The +1 in the denominator is what stops volume from faking conviction."""
    annotations = LexiconAnnotator().annotate(
        [item(str(index), "公司獲得訂單", minutes=index * 90) for index in range(20)],
        {str(index): 0.2 for index in range(20)},
    )
    features = aggregate_sentiment(annotations, NOW)
    assert abs(features["2330.TW"]["sentiment_level"]) < 0.5


def test_dispersion_reports_disagreement_between_sources():
    annotations = LexiconAnnotator().annotate(
        [item("a", "公司獲利成長"), item("b", "公司轉虧", minutes=30)],
        {"a": 1.0, "b": 1.0},
    )
    features = aggregate_sentiment(annotations, NOW)
    assert features["2330.TW"]["sentiment_dispersion"] > 0.3
