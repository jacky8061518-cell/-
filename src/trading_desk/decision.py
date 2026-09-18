"""Decision layer: turn blackboard evidence into signal cards.

Deterministic by construction. Every number here can be recomputed from the
blackboard, which is what makes a decision replayable months later when someone
asks why the desk was long a name that subsequently halved.

Conviction is built from four separable factors rather than one opaque score,
so a trader can see which factor a weak signal is weak on:

    conviction = base(agreement, breadth)
               × corroboration      (do independent sources agree)
               × regime multiplier  (is the tape hostile to this direction)
               × data-quality       (how stale or conflicted are the inputs)
               − counter-evidence penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .blackboard import Blackboard, SymbolView
from .contracts import (
    DataQuality,
    RegimeState,
    Severity,
    SignalCard,
    stable_id,
    utc_now,
)
from .risk import RiskEngine

COUNTER_PENALTY = {"low": 0.04, "medium": 0.09, "high": 0.16}
MAX_COUNTER_PENALTY = 0.30

# Base conviction band. A single-source finding with weak agreement starts near
# 0.45 and cannot reach the action threshold on its own; corroboration across
# independent evidence families is what earns a tradeable score.
BASE_FLOOR = 0.45
AGREEMENT_SPAN = 0.30
BREADTH_BONUS = 0.06

# Hard ceiling on conviction. An ensemble of four heuristics agreeing is
# encouraging, not certain, and a card reading "100%" would misrepresent what
# the system actually knows. The multipliers can stack past 1.0; this is where
# that gets truncated back to something honest.
MAX_CONVICTION = 0.90

HORIZON_BY_KIND = {
    "flow": "5-15 個交易日",
    "momentum": "20-60 個交易日",
    "anomaly": "1-5 個交易日",
    "sentiment": "2-10 個交易日",
}


@dataclass(frozen=True)
class DecisionConfig:
    """Knobs the decision layer is allowed to have. Risk limits live elsewhere."""

    min_conviction: float = 0.55
    signal_budget: int = 10
    min_evidence_kinds: int = 2
    require_flow_or_momentum: bool = True

    def __post_init__(self) -> None:
        if self.signal_budget < 1:
            raise ValueError("signal budget must allow at least one signal")
        if self.min_evidence_kinds < 1:
            raise ValueError("a signal needs at least one kind of evidence")


@dataclass
class DecisionLayer:
    """Assembles signal cards and hands each one to the risk engine for sizing."""

    config: DecisionConfig = field(default_factory=DecisionConfig)
    risk_engine: RiskEngine = field(default_factory=RiskEngine)

    def build(
        self,
        board: Blackboard,
        run_id: str,
        breaker=None,
    ) -> tuple[list[SignalCard], int]:
        """Return the accepted signals and how many were suppressed by the budget."""
        regime = board.regime
        candidates: list[SignalCard] = []

        for view in board.candidates():
            card = self._build_one(view, board, regime, run_id, breaker)
            if card is not None:
                candidates.append(card)

        # Convictions tie often, because many names match the same evidence
        # pattern on the same day. Break ties on how strong that evidence
        # actually was, then on symbol, so the ordering is reproducible.
        candidates.sort(
            key=lambda card: (
                -card.conviction,
                -sum(abs(item.value or 0.0) * item.weight for item in card.evidence),
                card.symbol,
            )
        )
        accepted = candidates[: self.config.signal_budget]
        suppressed = len(candidates) - len(accepted)

        for card in accepted:
            card.severity = self._severity(card)
        return accepted, suppressed

    # --- Single-signal assembly -------------------------------------------

    def _build_one(
        self,
        view: SymbolView,
        board: Blackboard,
        regime: RegimeState | None,
        run_id: str,
        breaker,
    ) -> SignalCard | None:
        kinds = {item.kind for item in view.evidence}
        if len(kinds) < self.config.min_evidence_kinds:
            return None
        if self.config.require_flow_or_momentum and not (kinds & {"flow", "momentum"}):
            # Anomaly plus news with no confirmation from money or trend is a
            # story, not a position.
            return None

        direction, strength = self._direction(view)
        if direction == "flat":
            return None

        conviction = self._conviction(view, kinds, strength, direction, regime)
        if conviction < self.config.min_conviction:
            return None

        quality = self._data_quality(board, view)
        budget = self._size(view, conviction, direction, regime, breaker)
        if not budget.is_tradeable:
            # Keep the reason: a rejected size is still worth showing as intel.
            view.notes["risk_rejection"] = budget.binding_constraint

        decision_id = stable_id("dec", run_id, view.symbol, direction)
        created = utc_now()

        return SignalCard(
            signal_id=stable_id("sig", decision_id, f"{conviction:.4f}"),
            decision_id=decision_id,
            created_at=created,
            symbol=view.symbol,
            name=view.name or view.symbol,
            direction=direction,
            conviction=round(conviction, 4),
            horizon=self._horizon(kinds),
            thesis=self._thesis(view, direction, regime),
            evidence=list(view.evidence),
            counter_evidence=list(view.counter_evidence),
            regime_context=regime.describe() if regime else "未判定",
            risk=budget,
            invalidation=self._invalidation(view, direction, budget),
            data_quality=quality,
            model_provenance={
                "run_id": run_id,
                "agents": sorted({item.agent for item in view.evidence}),
                "corroboration_score": round(
                    float(view.notes.get("corroboration_score", 1.0)), 3
                ),
                "regime_multiplier": self._regime_multiplier(regime, direction),
                "feature_count": len(view.features),
                "decision_layer_version": "1.0",
            },
        )

    def _direction(self, view: SymbolView) -> tuple[str, float]:
        """Weighted vote across evidence, with the margin of victory."""
        support = {1: 0.0, -1: 0.0}
        for item in view.evidence:
            if item.direction:
                support[item.direction] += item.weight
        total = support[1] + support[-1]
        if total <= 0:
            return "flat", 0.0
        if support[1] == support[-1]:
            return "flat", 0.0
        direction = "long" if support[1] > support[-1] else "short"
        winner = max(support[1], support[-1])
        loser = min(support[1], support[-1])
        return direction, float((winner - loser) / total)

    def _conviction(
        self,
        view: SymbolView,
        kinds: set[str],
        strength: float,
        direction: str,
        regime: RegimeState | None,
    ) -> float:
        sign = 1 if direction == "long" else -1
        agreeing = {item.kind for item in view.evidence if item.direction == sign}

        base = BASE_FLOOR + AGREEMENT_SPAN * strength + BREADTH_BONUS * (len(agreeing) - 1)
        corroboration = float(view.notes.get("corroboration_score", 1.0))
        regime_multiplier = self._regime_multiplier(regime, direction)

        penalty = sum(
            COUNTER_PENALTY.get(item.severity, 0.05) for item in view.counter_evidence
        )
        penalty = min(penalty, MAX_COUNTER_PENALTY)

        conviction = base * corroboration * regime_multiplier - penalty
        capped = float(np.clip(conviction, 0.0, MAX_CONVICTION))
        view.notes["conviction_breakdown"] = {
            "base": round(base, 4),
            "corroboration": round(corroboration, 4),
            "regime_multiplier": round(regime_multiplier, 4),
            "counter_penalty": round(penalty, 4),
            "raw": round(conviction, 4),
            "capped_at": MAX_CONVICTION if conviction > MAX_CONVICTION else None,
        }
        return capped

    def _regime_multiplier(self, regime: RegimeState | None, direction: str) -> float:
        if regime is None:
            return 1.0
        from .agents.regime import REGIME_MULTIPLIERS

        return REGIME_MULTIPLIERS.get(regime.label, {}).get(direction, 1.0)

    def _data_quality(self, board: Blackboard, view: SymbolView) -> DataQuality:
        stale = board.stale_sources
        conflicts = tuple(
            item.detail
            for item in view.counter_evidence
            if item.agent == "corroborator"
        )
        return DataQuality(
            all_sources_fresh=not stale and not conflicts,
            stale_sources=stale,
            conflicts=conflicts,
        )

    def _size(self, view: SymbolView, conviction: float, direction: str, regime, breaker):
        volatility = view.feature("vol_ewma")
        if not np.isfinite(volatility):
            volatility = view.feature("vol_20d")
        price = view.feature("last_close")
        # ATR is approximated from EWMA volatility; the real ATR needs intraday
        # highs and lows, which the daily panel does not carry.
        atr = price * volatility / np.sqrt(252) if np.isfinite(volatility) and np.isfinite(price) else None
        return self.risk_engine.size(
            conviction=conviction,
            annual_volatility=volatility,
            last_price=price,
            direction=direction,
            atr=atr,
            regime_multiplier=1.0,
            breaker=breaker,
        )

    def _horizon(self, kinds: set[str]) -> str:
        for kind in ("momentum", "flow", "sentiment", "anomaly"):
            if kind in kinds:
                return HORIZON_BY_KIND[kind]
        return "5-15 個交易日"

    def _thesis(self, view: SymbolView, direction: str, regime: RegimeState | None) -> str:
        """Plain-language summary assembled from the evidence, not generated prose."""
        sign = 1 if direction == "long" else -1
        agreeing = [item for item in view.evidence if item.direction == sign]
        agreeing.sort(key=lambda item: -item.weight)
        lead = "；".join(item.detail for item in agreeing[:3])
        stance = "做多" if direction == "long" else "做空"
        regime_note = f"當前 {regime.label} 環境" if regime else "環境未判定"
        corroboration = view.notes.get("corroboration_explanation", "")
        text = f"{stance}理由：{lead}。{corroboration}。{regime_note}。"
        if view.counter_evidence:
            text += f"需同時注意 {len(view.counter_evidence)} 項反面證據。"
        return text

    def _invalidation(self, view: SymbolView, direction: str, budget) -> str:
        """What has to happen for this idea to be wrong. Non-negotiable field."""
        parts: list[str] = []
        if budget.stop_level is not None:
            comparison = "跌破" if direction == "long" else "站上"
            parts.append(f"價格{comparison} {budget.stop_level:.2f}（{budget.stop_basis}）")
        kinds = {item.kind for item in view.evidence}
        if "flow" in kinds:
            parts.append("法人由買超轉為連 2 日賣超" if direction == "long" else "法人轉為連 2 日買超")
        if "momentum" in kinds:
            parts.append("多週期動能分數轉為反向")
        if "sentiment" in kinds:
            parts.append("出現方向相反的重大新聞")
        return "；或".join(parts) if parts else "動能與資金流同時轉向"

    def _severity(self, card: SignalCard) -> Severity:
        """Alert priority. The budget for P1 is small on purpose."""
        if (
            card.conviction >= 0.72
            and card.risk.is_tradeable
            and not card.data_quality.stale_sources
        ):
            return Severity.P1_ACTION
        if card.conviction >= 0.62:
            return Severity.P2_ATTENTION
        return Severity.P3_INTEL
