"""實體解析：信心不足時進人工佇列，不猜。"""

from __future__ import annotations

from datetime import datetime

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.normalize.entities import (
    EntityResolver,
    MatchMethod,
    normalize_text,
    similarity,
)

NOW = datetime(2024, 3, 19, tzinfo=UTC)


@pytest.fixture
def resolver() -> EntityResolver:
    r = EntityResolver(market=Market.TW)
    r.register("2330", ["台積電", "台積", "TSMC", "台灣積體電路製造股份有限公司"])
    r.register("2317", ["鴻海", "鴻海精密", "Foxconn"])
    r.register("2454", ["聯發科", "MediaTek"])
    return r


def test_the_spec_example_all_resolve_to_one_entity(resolver: EntityResolver) -> None:
    """SPEC 3.2：「台積」「TSMC」「2330」「台積電」必須解析到同一個 entity_id。"""
    results = [resolver.resolve(text, now=NOW) for text in ("台積", "TSMC", "2330", "台積電")]
    assert all(item.resolved for item in results)
    assert len({item.entity_id for item in results}) == 1
    assert results[0].entity_id == "TW:2330"


def test_ticker_with_suffix_resolves(resolver: EntityResolver) -> None:
    assert resolver.resolve("2330.TW", now=NOW).entity_id == "TW:2330"
    assert resolver.resolve("2330.tw", now=NOW).method is MatchMethod.EXACT_CODE


def test_full_company_name_resolves(resolver: EntityResolver) -> None:
    result = resolver.resolve("台灣積體電路製造股份有限公司", now=NOW)
    assert result.entity_id == "TW:2330"


def test_company_suffix_is_stripped(resolver: EntityResolver) -> None:
    assert resolver.resolve("鴻海精密股份有限公司", now=NOW).entity_id == "TW:2317"


def test_case_and_whitespace_are_ignored(resolver: EntityResolver) -> None:
    assert resolver.resolve("  tsmc  ", now=NOW).entity_id == "TW:2330"
    assert resolver.resolve("Media Tek", now=NOW).entity_id == "TW:2454"


def test_unknown_text_is_unresolved(resolver: EntityResolver) -> None:
    result = resolver.resolve("完全不相干的公司名稱", now=NOW)
    assert not result.resolved
    assert result.method is MatchMethod.UNRESOLVED


def test_low_confidence_goes_to_the_review_queue(resolver: EntityResolver) -> None:
    """SPEC 3.2：模糊比對閾值以外的丟人工佇列，不要猜。"""
    resolver.resolve("台積化學", now=NOW, context="某則新聞")
    queue = resolver.review_queue
    assert len(queue) == 1
    assert queue[0].query == "台積化學"
    assert queue[0].context == "某則新聞"
    assert queue[0].candidates


def test_resolved_queries_do_not_enter_the_queue(resolver: EntityResolver) -> None:
    resolver.resolve("台積電", now=NOW)
    resolver.resolve("2330", now=NOW)
    assert resolver.review_queue == ()


def test_ambiguous_match_is_not_auto_decided() -> None:
    """兩個候選分數過於接近時視為歧義，交人工而非取第一名。"""
    r = EntityResolver(market=Market.TW)
    r.register("1101", ["台泥A"])
    r.register("1102", ["台泥B"])
    result = r.resolve("台泥X", now=NOW)
    assert not result.resolved
    assert result.needs_review
    assert len(r.review_queue) == 1


def test_manual_alias_resolves_afterwards(resolver: EntityResolver) -> None:
    """人工審核通過後回填別名，之後就是精確命中。"""
    unresolved = resolver.resolve("護國神山", now=NOW)
    assert not unresolved.resolved
    resolver.add_alias("護國神山", "TW:2330")  # type: ignore[arg-type]
    result = resolver.resolve("護國神山", now=NOW)
    assert result.entity_id == "TW:2330"
    assert result.method is MatchMethod.EXACT_ALIAS


def test_blank_alias_is_rejected(resolver: EntityResolver) -> None:
    with pytest.raises(ValueError, match="正規化後為空字串"):
        resolver.add_alias("   ", "TW:2330")  # type: ignore[arg-type]


def test_empty_query_is_unresolved(resolver: EntityResolver) -> None:
    result = resolver.resolve("   ", now=NOW)
    assert not result.resolved
    assert result.candidates == ()


def test_clear_review_queue(resolver: EntityResolver) -> None:
    resolver.resolve("台積化學", now=NOW)
    resolver.clear_review_queue()
    assert resolver.review_queue == ()


def test_normalize_handles_fullwidth_and_punctuation() -> None:
    assert normalize_text("台積電（２３３０）") == "台積電2330"
    assert normalize_text("T.S.M.C.") == "TSMC"


def test_similarity_bounds() -> None:
    assert similarity("TSMC", "TSMC") == 1.0
    assert similarity("", "TSMC") == 0.0
    assert 0.0 < similarity("TSMC", "TSMD") < 1.0


def test_candidates_are_ranked(resolver: EntityResolver) -> None:
    result = resolver.resolve("台積化學", now=NOW)
    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
