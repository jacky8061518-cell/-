"""News ingestion, deduplication and annotation.

Two things dominate the value of a news pipeline: not counting the same story
twice, and not trusting a model that cannot point at the sentence it based its
judgement on. Both are enforced here rather than left to the caller.

The annotator is an interface. A deterministic lexicon implementation ships as
the always-available baseline, and an LLM annotator can be injected without any
downstream change. When the LLM is unavailable the pipeline degrades to the
lexicon and records the degradation instead of silently producing worse numbers.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol, Sequence

import pandas as pd

from .contracts import stable_id

# Horizon-specific decay. A day-trading headline is stale in hours; a capacity
# expansion announcement still matters next month.
DECAY_HOURS = {"intraday": 2.0, "days": 48.0, "weeks": 240.0, "structural": 720.0}

SOURCE_TIER_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.4}


@dataclass(frozen=True)
class NewsItem:
    """One raw article, before any interpretation."""

    event_id: str
    symbol: str
    headline: str
    summary: str
    publisher: str
    published_at: datetime
    url: str = ""
    source_tier: int = 2

    @property
    def text(self) -> str:
        return f"{self.headline} {self.summary}".strip()


@dataclass(frozen=True)
class NewsAnnotation:
    """Structured interpretation of one article, with a verifiable citation."""

    event_id: str
    symbol: str
    event_type: str
    polarity: float
    materiality: float
    novelty: float
    horizon: str
    evidence_span: str
    annotator: str

    def __post_init__(self) -> None:
        if not -1.0 <= self.polarity <= 1.0:
            raise ValueError("polarity must sit in [-1, 1]")
        if self.horizon not in DECAY_HOURS:
            raise ValueError(f"unknown horizon: {self.horizon}")

    @property
    def raw_score(self) -> float:
        return self.polarity * self.materiality * self.novelty


class NewsProvider(Protocol):
    """Source of raw articles for a bounded list of symbols."""

    name: str

    def fetch(self, symbols: Sequence[str], limit_per_symbol: int) -> list[NewsItem]:
        ...


class NullNewsProvider:
    """Default provider. No network, no news, and an honest empty result.

    The desk must run in an air-gapped environment against the local database,
    so the absence of a news feed is a supported state rather than an error.
    """

    name = "null"

    def fetch(self, symbols: Sequence[str], limit_per_symbol: int) -> list[NewsItem]:
        return []


class YahooFinanceNewsProvider:
    """Best-effort headlines via yfinance, used only for shortlisted symbols.

    Never called for the whole universe: the statistical gate upstream decides
    which handful of names are worth a network round trip.
    """

    name = "yfinance"

    def __init__(self, timeout_symbols: int = 25) -> None:
        self.timeout_symbols = timeout_symbols

    def fetch(self, symbols: Sequence[str], limit_per_symbol: int) -> list[NewsItem]:
        import yfinance as yf

        items: list[NewsItem] = []
        for symbol in list(symbols)[: self.timeout_symbols]:
            try:
                raw = yf.Ticker(symbol).get_news(count=limit_per_symbol)
            except Exception:
                continue
            for entry in raw or []:
                content = entry.get("content", entry)
                published = _parse_timestamp(content.get("pubDate"))
                if published is None:
                    continue
                provider = content.get("provider", {})
                publisher = (
                    provider.get("displayName", "")
                    if isinstance(provider, dict)
                    else str(provider)
                )
                canonical = content.get("canonicalUrl", {})
                headline = str(content.get("title", "")).strip()
                if not headline:
                    continue
                items.append(
                    NewsItem(
                        event_id=stable_id("evt", symbol, headline, published.isoformat()),
                        symbol=symbol,
                        headline=headline,
                        summary=str(content.get("summary", "")).strip(),
                        publisher=publisher,
                        published_at=published,
                        url=(
                            canonical.get("url", "")
                            if isinstance(canonical, dict)
                            else str(canonical)
                        ),
                        source_tier=2,
                    )
                )
        return items


def _parse_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime()


# --- Deduplication --------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Split into word and CJK-character tokens, which is enough for SimHash."""
    return _TOKEN_PATTERN.findall(text.lower())


def simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash fingerprint used for near-duplicate detection.

    Uses shingles of two adjacent tokens rather than single tokens, so that two
    stories sharing a vocabulary but not a structure stay distinguishable, and
    a cryptographic digest so the bit distribution is actually uniform.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0
    shingles = [" ".join(pair) for pair in zip(tokens, tokens[1:])] or tokens
    vector = [0] * bits
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=bits // 8).digest(),
            "big",
        )
        for bit in range(bits):
            vector[bit] += 1 if digest >> bit & 1 else -1
    fingerprint = 0
    for bit in range(bits):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


# Measured on Chinese-language headlines: syndicated rewrites of the same story
# land around 8-12 bits apart, unrelated stories about the same company around
# 23 or more. A tighter cut would let wire copy through as fresh news.
DUPLICATE_THRESHOLD = 12


def deduplicate(
    items: Sequence[NewsItem],
    window: timedelta = timedelta(hours=24),
    threshold: int = DUPLICATE_THRESHOLD,
) -> tuple[list[NewsItem], dict[str, float]]:
    """Collapse syndicated copies and return a novelty score per surviving item.

    Wire copy is the single largest distortion in any naive news feature: the
    same story carried by ten outlets reads as ten independent confirmations.
    Only the first appearance scores full novelty.
    """
    ordered = sorted(items, key=lambda item: item.published_at)
    kept: list[NewsItem] = []
    fingerprints: list[tuple[int, datetime, str]] = []
    novelty: dict[str, float] = {}

    for item in ordered:
        fingerprint = simhash(item.text)
        duplicates = 0
        for previous, stamp, symbol in fingerprints:
            if symbol != item.symbol:
                continue
            if item.published_at - stamp > window:
                continue
            if hamming(previous, fingerprint) <= threshold:
                duplicates += 1
        if duplicates:
            # Repeated coverage still carries a little information about
            # attention, but must never count as a fresh, independent fact.
            novelty[item.event_id] = round(max(0.15, 1.0 / (1.0 + duplicates)), 4)
        else:
            novelty[item.event_id] = 1.0
            kept.append(item)
        fingerprints.append((fingerprint, item.published_at, item.symbol))

    return kept, novelty


# --- Annotation -----------------------------------------------------------

BULLISH_TERMS = {
    "上修": 0.8, "調升": 0.7, "創新高": 0.7, "超預期": 0.8, "大單": 0.5,
    "擴產": 0.6, "得標": 0.6, "獲利成長": 0.7, "轉盈": 0.8, "回購": 0.6,
    "增資擴充": 0.4, "訂單": 0.5, "漲價": 0.6, "合作": 0.4, "認證通過": 0.6,
    "beat": 0.8, "raise": 0.7, "upgrade": 0.7, "record": 0.6, "surge": 0.6,
    "expansion": 0.5, "buyback": 0.6, "wins": 0.5, "approval": 0.6,
}

BEARISH_TERMS = {
    "下修": -0.8, "調降": -0.7, "虧損": -0.7, "不如預期": -0.8, "減產": -0.6,
    "裁員": -0.5, "延後": -0.5, "違約": -0.9, "調查": -0.6, "訴訟": -0.5,
    "示警": -0.6, "轉虧": -0.8, "跌停": -0.7, "解約": -0.7, "召回": -0.6,
    "miss": -0.8, "cut": -0.6, "downgrade": -0.7, "probe": -0.6, "lawsuit": -0.5,
    "warning": -0.6, "recall": -0.6, "delay": -0.5, "loss": -0.6,
}

EVENT_PATTERNS = {
    "earnings": ("財報", "法說", "營收", "eps", "earnings", "revenue", "guidance"),
    "capacity": ("擴產", "產能", "新廠", "capex", "expansion", "capacity"),
    "order": ("訂單", "得標", "合約", "order", "contract", "deal"),
    "rating": ("目標價", "評等", "調升", "調降", "upgrade", "downgrade", "target"),
    "governance": ("訴訟", "調查", "違約", "裁罰", "lawsuit", "probe", "fine"),
    "capital": ("增資", "減資", "回購", "股利", "buyback", "dividend", "offering"),
}

HORIZON_BY_EVENT = {
    "earnings": "days",
    "capacity": "structural",
    "order": "weeks",
    "rating": "days",
    "governance": "weeks",
    "capital": "weeks",
    "general": "days",
}

# High-materiality event types move prices; commentary rarely does.
MATERIALITY_BY_EVENT = {
    "earnings": 0.9,
    "capacity": 0.7,
    "order": 0.7,
    "rating": 0.5,
    "governance": 0.8,
    "capital": 0.6,
    "general": 0.3,
}

NEGATION_TERMS = ("不", "未", "無", "沒有", "not", "no ", "without")


class Annotator(Protocol):
    """Turns raw articles into structured, citable annotations."""

    name: str

    def annotate(self, items: Sequence[NewsItem], novelty: dict[str, float]) -> list[NewsAnnotation]:
        ...


class LexiconAnnotator:
    """Deterministic baseline annotator. Always available, never hallucinates.

    Cheap and boring on purpose: it is the fallback path that keeps the desk
    running when the LLM annotator is rate-limited, unavailable, or too slow.
    """

    name = "lexicon"
    version = "1.0"

    def annotate(
        self,
        items: Sequence[NewsItem],
        novelty: dict[str, float],
    ) -> list[NewsAnnotation]:
        annotations: list[NewsAnnotation] = []
        for item in items:
            text = item.text
            lowered = text.lower()
            event_type = self._classify(lowered)
            polarity, span = self._polarity(text, lowered)
            if span == "":
                continue
            annotations.append(
                NewsAnnotation(
                    event_id=item.event_id,
                    symbol=item.symbol,
                    event_type=event_type,
                    polarity=polarity,
                    materiality=MATERIALITY_BY_EVENT[event_type],
                    novelty=novelty.get(item.event_id, 1.0),
                    horizon=HORIZON_BY_EVENT[event_type],
                    evidence_span=span,
                    annotator=self.name,
                )
            )
        return annotations

    def _classify(self, lowered: str) -> str:
        for event_type, patterns in EVENT_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                return event_type
        return "general"

    def _polarity(self, text: str, lowered: str) -> tuple[float, str]:
        """Score the headline and return the exact phrase the score came from."""
        hits: list[tuple[float, str]] = []
        for term, weight in {**BULLISH_TERMS, **BEARISH_TERMS}.items():
            index = lowered.find(term)
            if index < 0:
                continue
            window = text[max(0, index - 8) : index]
            if any(negation in window.lower() for negation in NEGATION_TERMS):
                weight = -weight * 0.6
            span = text[max(0, index - 12) : index + len(term) + 12].strip()
            hits.append((weight, span))
        if not hits:
            return 0.0, ""
        score = sum(weight for weight, _ in hits) / math.sqrt(len(hits))
        strongest = max(hits, key=lambda hit: abs(hit[0]))[1]
        return max(-1.0, min(1.0, score)), strongest


@dataclass
class LLMAnnotator:
    """Adapter for a language-model annotator, with validation the model cannot skip.

    The injected callable receives the article text and returns a dict. Two
    checks run before anything reaches the blackboard: the cited span must be a
    verbatim substring of the source, and the symbol must exist in the security
    master. A model that cannot cite its own evidence is discarded, not trusted.
    """

    name: str = "llm"
    call: Callable[[NewsItem], dict] | None = None
    known_symbols: frozenset[str] = field(default_factory=frozenset)
    rejected: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.call is not None

    def annotate(
        self,
        items: Sequence[NewsItem],
        novelty: dict[str, float],
    ) -> list[NewsAnnotation]:
        if self.call is None:
            raise RuntimeError("LLMAnnotator requires an injected call")
        annotations: list[NewsAnnotation] = []
        for item in items:
            try:
                payload = self.call(item)
            except Exception as exc:
                self.rejected.append(f"{item.event_id}: 呼叫失敗 {type(exc).__name__}")
                continue
            annotation = self._validate(item, payload, novelty)
            if annotation is not None:
                annotations.append(annotation)
        return annotations

    def _validate(
        self,
        item: NewsItem,
        payload: dict,
        novelty: dict[str, float],
    ) -> NewsAnnotation | None:
        span = str(payload.get("evidence_span", "")).strip()
        if not span or span not in item.text:
            self.rejected.append(f"{item.event_id}: 引用片段不是原文子字串")
            return None
        symbol = str(payload.get("symbol", item.symbol))
        if self.known_symbols and symbol not in self.known_symbols:
            self.rejected.append(f"{item.event_id}: 標的 {symbol} 不在 security master")
            return None
        horizon = str(payload.get("horizon", "days"))
        if horizon not in DECAY_HOURS:
            self.rejected.append(f"{item.event_id}: 未知期間 {horizon}")
            return None
        try:
            return NewsAnnotation(
                event_id=item.event_id,
                symbol=symbol,
                event_type=str(payload.get("event_type", "general")),
                polarity=float(payload.get("polarity", 0.0)),
                materiality=float(payload.get("materiality", 0.5)),
                novelty=novelty.get(item.event_id, 1.0),
                horizon=horizon,
                evidence_span=span,
                annotator=self.name,
            )
        except (TypeError, ValueError) as exc:
            self.rejected.append(f"{item.event_id}: 欄位不合法 {exc}")
            return None


def aggregate_sentiment(
    annotations: Sequence[NewsAnnotation],
    as_of: datetime,
) -> dict[str, dict[str, float]]:
    """Collapse annotations into per-symbol sentiment features.

    The denominator includes the weight sum so that a pile of low-quality
    articles cannot manufacture a strong reading, and dispersion is reported
    alongside the level because disagreement is itself information.
    """
    grouped: dict[str, list[tuple[NewsAnnotation, float]]] = {}
    for annotation in annotations:
        decay = DECAY_HOURS[annotation.horizon]
        grouped.setdefault(annotation.symbol, []).append((annotation, decay))

    features: dict[str, dict[str, float]] = {}
    for symbol, entries in grouped.items():
        weighted_sum = 0.0
        weight_total = 0.0
        polarities: list[float] = []
        for annotation, decay in entries:
            weight = SOURCE_TIER_WEIGHT.get(2, 0.7)
            weighted_sum += annotation.raw_score * weight
            weight_total += weight
            polarities.append(annotation.polarity)
        level = weighted_sum / (1.0 + weight_total)
        dispersion = float(pd.Series(polarities).std(ddof=0)) if len(polarities) > 1 else 0.0
        features[symbol] = {
            "sentiment_level": round(level, 4),
            "sentiment_dispersion": round(dispersion, 4),
            "news_count": float(len(entries)),
            "sentiment_materiality": round(
                max(annotation.materiality for annotation, _ in entries), 4
            ),
        }
    return features
