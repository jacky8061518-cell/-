"""Sentiment agent: the semantic layer, deliberately kept on a short leash.

This is the only agent that reads free text, and it runs last and narrowest.
It looks at the shortlist the statistical agents produced, not the universe,
because semantic work is the expensive part of the pipeline and spending it on
two thousand names that nobody flagged is how an AI system's cost curve gets
out of control.

The agent never computes a number the deterministic layer could have computed.
It classifies, extracts and cites; the arithmetic happens in ``news.py``.
"""

from __future__ import annotations

import pandas as pd

from ..blackboard import Blackboard
from ..contracts import CounterEvidence, Evidence, Quality
from ..news import (
    Annotator,
    LexiconAnnotator,
    NewsProvider,
    NullNewsProvider,
    aggregate_sentiment,
    deduplicate,
)
from .base import Agent, MarketContext

MAX_SYMBOLS = 30
ARTICLES_PER_SYMBOL = 5
LEVEL_TRIGGER = 0.15
DISPERSION_TRIGGER = 0.5


class SentimentAgent(Agent):
    name = "sentiment"
    version = "1.0"
    sources = ("news",)

    def __init__(
        self,
        provider: NewsProvider | None = None,
        annotator: Annotator | None = None,
        fallback: Annotator | None = None,
        max_symbols: int = MAX_SYMBOLS,
    ) -> None:
        self.provider = provider or NullNewsProvider()
        self.fallback = fallback or LexiconAnnotator()
        self.annotator = annotator or self.fallback
        self.max_symbols = max_symbols

    @property
    def uses_llm(self) -> bool:  # type: ignore[override]
        """True only when a real model is wired in, not for the lexicon fallback."""
        return self.annotator is not self.fallback

    def run(self, context: MarketContext, board: Blackboard) -> int:
        # A source that was never configured is a capability the desk does not
        # have, not a feed that has gone stale. Conflating the two would leave
        # every signal permanently discounted for a feed nobody subscribed to.
        configured = not isinstance(self.provider, NullNewsProvider) or (
            context.news is not None and not context.news.empty
        )
        board.market["news_enabled"] = configured
        if not configured:
            board.market["news_status"] = "未接新聞來源，語意層停用"
            return 0

        shortlist = self._shortlist(board)
        if not shortlist:
            board.market["news_status"] = "無候選標的，未觸發語意分析"
            return 0

        items = self._fetch(shortlist, context, board)
        if not items:
            board.record_degradation("sentiment: 已設定新聞來源但取回 0 則，語意證據停用")
            board.record_freshness("news", None, expected_lag_hours=1.0)
            return 0

        latest = max(item.published_at for item in items)
        board.record_freshness("news", latest, expected_lag_hours=24.0)

        kept, novelty = deduplicate(items)
        board.market["news_items"] = len(items)
        board.market["news_unique"] = len(kept)

        annotations = self._annotate(items, novelty, board)
        if not annotations:
            return 0

        features = aggregate_sentiment(annotations, context.as_of.to_pydatetime())
        board.market["news_annotations"] = annotations
        return self._emit(features, annotations, context, board)

    def _shortlist(self, board: Blackboard) -> list[str]:
        """Rank the flagged candidates and keep only the strongest handful.

        The cost gate: semantic analysis is reserved for names the cheap
        deterministic layer already found interesting.
        """
        candidates = board.candidates()
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda view: (
                len({item.kind for item in view.evidence}),
                sum(abs(item.weight * item.direction) for item in view.evidence),
            ),
            reverse=True,
        )
        return [view.symbol for view in ranked[: self.max_symbols]]

    def _fetch(self, shortlist: list[str], context: MarketContext, board: Blackboard):
        supplied = context.visible_news()
        if not supplied.empty:
            from ..news import NewsItem

            rows = supplied[supplied["symbol"].isin(shortlist)]
            return [
                NewsItem(
                    event_id=str(row["event_id"]),
                    symbol=str(row["symbol"]),
                    headline=str(row["headline"]),
                    summary=str(row.get("summary", "")),
                    publisher=str(row.get("publisher", "")),
                    published_at=pd.Timestamp(row["published_at"]).to_pydatetime(),
                    url=str(row.get("url", "")),
                    source_tier=int(row.get("source_tier", 2)),
                )
                for _, row in rows.iterrows()
            ]
        try:
            return self.provider.fetch(shortlist, ARTICLES_PER_SYMBOL)
        except Exception as exc:
            board.record_degradation(
                f"sentiment: 新聞來源 {self.provider.name} 失敗 {type(exc).__name__}"
            )
            return []

    def _annotate(self, items, novelty, board: Blackboard):
        """Try the configured annotator, fall back rather than fail the run."""
        try:
            annotations = self.annotator.annotate(items, novelty)
        except Exception as exc:
            board.record_degradation(
                f"sentiment: {self.annotator.name} 標註失敗，降級為規則式（{type(exc).__name__}）"
            )
            annotations = self.fallback.annotate(items, novelty)
        rejected = getattr(self.annotator, "rejected", [])
        if rejected:
            board.market["news_rejected"] = list(rejected)
            board.record_degradation(f"sentiment: {len(rejected)} 則標註未通過引用驗證，已丟棄")
        return annotations

    def _emit(self, features, annotations, context: MarketContext, board: Blackboard) -> int:
        by_symbol: dict[str, list] = {}
        for annotation in annotations:
            by_symbol.setdefault(annotation.symbol, []).append(annotation)

        frame = pd.DataFrame.from_dict(features, orient="index")
        board.write_features(frame, agent=self.name, quality=Quality.OK)

        findings = 0
        for symbol, values in features.items():
            level = values["sentiment_level"]
            dispersion = values["sentiment_dispersion"]
            strongest = max(by_symbol[symbol], key=lambda item: abs(item.raw_score))

            if abs(level) >= LEVEL_TRIGGER:
                direction = 1 if level > 0 else -1
                board.view(symbol, context.name_of(symbol))
                board.add_evidence(
                    symbol,
                    Evidence(
                        kind="sentiment",
                        detail=(
                            f"新聞情緒 {level:+.2f}（{int(values['news_count'])} 則，"
                            f"主要事件 {strongest.event_type}）"
                            f"；引用「{strongest.evidence_span}」"
                        ),
                        weight=0.20,
                        direction=direction,
                        agent=self.name,
                        value=float(level),
                    ),
                )
                board.tag(symbol, "news")
                findings += 1

            # Disagreement across sources means the event is not yet priced and
            # is equally likely to resolve against the position.
            if dispersion >= DISPERSION_TRIGGER:
                board.add_counter_evidence(
                    symbol,
                    CounterEvidence(
                        detail=f"各來源情緒分歧度 {dispersion:.2f}，事件解讀尚未收斂",
                        severity="medium",
                        agent=self.name,
                    ),
                )
        return findings
