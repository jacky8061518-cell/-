"""Corroborator agent: does the evidence actually agree with itself.

This is the one place allowed to form a cross-source judgement, and its powers
are deliberately narrow: it may only adjust a confidence score inside fixed
bounds and write an explanation. It cannot create a signal, change a direction,
or override the risk engine.

Independent sources agreeing is worth far more than the same source repeating
itself, so agreement is measured across evidence *kinds* rather than count.
"""

from __future__ import annotations

import numpy as np

from ..blackboard import Blackboard
from ..contracts import CounterEvidence
from .base import Agent, MarketContext

# Bounds on what corroboration may do to a signal's confidence. Without a clamp
# a single agreement heuristic could quietly become the whole model.
MIN_MULTIPLIER = 0.60
MAX_MULTIPLIER = 1.30

DIVERGENCE_PENALTY = 0.75
SINGLE_SOURCE_PENALTY = 0.85


class CorroboratorAgent(Agent):
    name = "corroborator"
    version = "1.0"
    uses_llm = False
    sources = ("blackboard",)

    def run(self, context: MarketContext, board: Blackboard) -> int:
        findings = 0
        for view in board.candidates():
            score, explanation = self._score(view)
            view.notes["corroboration_score"] = score
            view.notes["corroboration_explanation"] = explanation
            if score < 1.0 and len({item.kind for item in view.evidence}) > 1:
                board.add_counter_evidence(
                    view.symbol,
                    CounterEvidence(
                        detail=explanation,
                        severity="high" if score <= 0.7 else "medium",
                        agent=self.name,
                    ),
                )
            findings += 1
        return findings

    def _score(self, view) -> tuple[float, str]:
        kinds = {item.kind for item in view.evidence}
        directions = {item.kind: item.direction for item in view.evidence}
        weights = {item.kind: item.weight for item in view.evidence}

        signed = sum(weights[kind] * directions[kind] for kind in kinds)
        total = sum(weights[kind] for kind in kinds)
        agreement = signed / total if total else 0.0

        if len(kinds) == 1:
            only = next(iter(kinds))
            return (
                SINGLE_SOURCE_PENALTY,
                f"僅有單一來源（{only}）支持，缺乏交叉驗證",
            )

        positive = [kind for kind in kinds if directions[kind] > 0]
        negative = [kind for kind in kinds if directions[kind] < 0]

        if positive and negative:
            return (
                DIVERGENCE_PENALTY,
                f"證據互相矛盾：{'、'.join(positive)} 偏多，{'、'.join(negative)} 偏空",
            )

        # Independent sources pointing the same way: reward, but with a ceiling.
        bonus = 1.0 + 0.10 * (len(kinds) - 1)
        multiplier = float(np.clip(bonus * abs(agreement), MIN_MULTIPLIER, MAX_MULTIPLIER))
        return (
            multiplier,
            f"{len(kinds)} 類獨立證據方向一致（{'、'.join(sorted(kinds))}）",
        )
