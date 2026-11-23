"""Narrative agent: what story is the market telling today.

A trader does not need five hundred flagged names; they need to know the two or
three themes driving the tape and which side of each theme their book sits on.
This agent clusters the day's findings by industry and by the kind of evidence
behind them, then names the theme.

Clustering is deterministic (industry plus evidence mix) rather than embedding
based, which keeps it replayable and free. An LLM could write nicer prose here,
but it would not change which clusters exist.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..blackboard import Blackboard
from .base import Agent, MarketContext

MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 7

EVIDENCE_LABELS = {
    "flow": "資金流",
    "momentum": "動能",
    "anomaly": "價格異常",
    "sentiment": "新聞情緒",
}


class NarrativeAgent(Agent):
    name = "narrative"
    version = "1.0"
    sources = ("blackboard",)

    def run(self, context: MarketContext, board: Blackboard) -> int:
        candidates = board.candidates()
        if not candidates:
            return 0

        industries = self._industry_map(context)
        buckets: dict[str, list] = defaultdict(list)
        for view in candidates:
            buckets[industries.get(view.symbol, "未分類")].append(view)

        clusters = []
        for industry, members in buckets.items():
            if len(members) < MIN_CLUSTER_SIZE or industry == "未分類":
                continue
            clusters.append(self._summarise(industry, members, context))

        clusters.sort(key=lambda cluster: -abs(cluster["net_direction"] * cluster["size"]))
        board.narratives = clusters[:MAX_CLUSTERS]
        return len(board.narratives)

    def _industry_map(self, context: MarketContext) -> dict[str, str]:
        if context.metadata.empty or "Industry" not in context.metadata.columns:
            return {}
        return context.metadata["Industry"].dropna().astype(str).to_dict()

    def _summarise(self, industry: str, members: list, context: MarketContext) -> dict:
        directions = [
            item.direction for view in members for item in view.evidence if item.direction
        ]
        net_direction = float(np.mean(directions)) if directions else 0.0
        kinds = Counter(
            item.kind for view in members for item in view.evidence
        )
        drivers = "、".join(
            EVIDENCE_LABELS.get(kind, kind) for kind, _ in kinds.most_common(3)
        )
        leaders = sorted(
            members,
            key=lambda view: -sum(abs(item.weight) for item in view.evidence),
        )[:5]

        tone = "偏多" if net_direction > 0.2 else "偏空" if net_direction < -0.2 else "分歧"
        return {
            "theme": f"{industry}：{drivers}{tone}",
            "industry": industry,
            "size": len(members),
            "net_direction": net_direction,
            "tone": tone,
            "drivers": drivers,
            "leaders": [
                {"symbol": view.symbol, "name": view.name or context.name_of(view.symbol)}
                for view in leaders
            ],
            "summary": (
                f"{len(members)} 檔 {industry} 標的同時被觸發，主要來自{drivers}；"
                f"整體方向{tone}（淨方向 {net_direction:+.2f}）。"
            ),
        }
