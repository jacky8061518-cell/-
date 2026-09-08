"""Shared state that agents write to and the decision layer reads from.

Agents never call each other. Each one writes its own findings onto the
blackboard keyed by symbol, and the decision layer reads the whole picture.
That keeps the dependency graph acyclic, every agent independently testable,
and one slow agent from blocking the rest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .contracts import CounterEvidence, Evidence, Quality, RegimeState


@dataclass
class SymbolView:
    """Everything the agents have said about one symbol during this run."""

    symbol: str
    name: str = ""
    features: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Quality] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    counter_evidence: list[CounterEvidence] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    notes: dict[str, Any] = field(default_factory=dict)

    def feature(self, name: str, default: float = float("nan")) -> float:
        value = self.features.get(name, default)
        return default if value is None else value

    def is_usable(self, name: str) -> bool:
        """True when a feature exists and is not missing.

        Callers must ask this rather than treating a missing feature as zero,
        which would silently turn "no data" into "neutral reading".
        """
        if name not in self.features:
            return False
        if self.quality.get(name, Quality.OK) is Quality.MISSING:
            return False
        return pd.notna(self.features[name])


class Blackboard:
    """Per-run shared state. One instance per pipeline execution, never global."""

    def __init__(self, as_of: datetime) -> None:
        self.as_of = as_of
        self._symbols: dict[str, SymbolView] = {}
        self.regime: RegimeState | None = None
        # Initialised here, not by the narrative agent: an agent that returns
        # early must not leave a downstream consumer reading a missing attribute.
        self.narratives: list[dict[str, Any]] = []
        self.market: dict[str, Any] = {}
        self.source_freshness: dict[str, dict[str, Any]] = {}
        self.degradations: list[str] = []
        self._writes: dict[str, int] = defaultdict(int)

    def view(self, symbol: str, name: str = "") -> SymbolView:
        view = self._symbols.get(symbol)
        if view is None:
            view = SymbolView(symbol=symbol, name=name)
            self._symbols[symbol] = view
        elif name and not view.name:
            view.name = name
        return view

    def write_feature(
        self,
        symbol: str,
        feature: str,
        value: float,
        agent: str,
        quality: Quality = Quality.OK,
    ) -> None:
        view = self.view(symbol)
        view.features[feature] = float(value) if value is not None else float("nan")
        view.quality[feature] = quality
        self._writes[agent] += 1

    def write_features(
        self,
        frame: pd.DataFrame,
        agent: str,
        quality: Quality = Quality.OK,
    ) -> None:
        """Bulk write a symbol-indexed frame of features, one column per feature."""
        for feature in frame.columns:
            column = frame[feature]
            for symbol, value in column.items():
                if pd.isna(value):
                    continue
                self.write_feature(str(symbol), str(feature), float(value), agent, quality)

    def add_evidence(self, symbol: str, evidence: Evidence) -> None:
        self.view(symbol).evidence.append(evidence)
        self._writes[evidence.agent] += 1

    def add_counter_evidence(self, symbol: str, counter: CounterEvidence) -> None:
        self.view(symbol).counter_evidence.append(counter)
        self._writes[counter.agent] += 1

    def tag(self, symbol: str, tag: str) -> None:
        self.view(symbol).tags.add(tag)

    def record_freshness(
        self,
        source: str,
        last_event_time: datetime | pd.Timestamp | None,
        expected_lag_hours: float,
    ) -> None:
        """Track how stale each source is so signals can be discounted, not blocked."""
        stale = True
        lag_hours = float("nan")
        if last_event_time is not None:
            stamp = pd.Timestamp(last_event_time)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            reference = pd.Timestamp(self.as_of)
            if reference.tzinfo is None:
                reference = reference.tz_localize("UTC")
            lag_hours = max(0.0, (reference - stamp).total_seconds() / 3600.0)
            stale = lag_hours > expected_lag_hours
        self.source_freshness[source] = {
            "last_event_time": last_event_time,
            "lag_hours": lag_hours,
            "expected_lag_hours": expected_lag_hours,
            "stale": stale,
        }

    def record_degradation(self, reason: str) -> None:
        """A degraded run still produces signals, but they carry the discount."""
        if reason not in self.degradations:
            self.degradations.append(reason)

    @property
    def stale_sources(self) -> tuple[str, ...]:
        return tuple(
            source for source, meta in self.source_freshness.items() if meta["stale"]
        )

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def views(self) -> list[SymbolView]:
        return list(self._symbols.values())

    def candidates(self) -> list[SymbolView]:
        """Only symbols an agent actually flagged reach the decision layer.

        This is the cost gate: the statistical layer narrows thousands of names
        down to a handful before any expensive semantic work happens.
        """
        return [view for view in self._symbols.values() if view.evidence]

    def feature_frame(self) -> pd.DataFrame:
        """Symbol-indexed view of every feature written during this run."""
        if not self._symbols:
            return pd.DataFrame()
        rows = {symbol: view.features for symbol, view in self._symbols.items()}
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "Symbol"
        return frame.sort_index()

    def write_counts(self) -> dict[str, int]:
        return dict(self._writes)
