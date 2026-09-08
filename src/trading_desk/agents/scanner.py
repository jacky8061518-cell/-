"""Scanner agent: the price-structure feature family.

Deterministic. Reads the point-in-time price panel once, computes every
price-derived feature in vectorised form, and writes them to the blackboard.
It emits evidence only for names whose relative strength is genuinely extreme
in the cross-section, because "everything is a finding" is the same as no
findings at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..blackboard import Blackboard
from ..contracts import Evidence, Quality
from ..statistics import (
    downside_volatility_frame,
    ewma_volatility_frame,
    max_drawdown_frame,
    rank_to_normal,
    realised_volatility_frame,
)
from .base import Agent, MarketContext

RETURN_HORIZONS = {"ret_5d": 5, "ret_20d": 20, "ret_60d": 60, "ret_120d": 120}
MOVING_AVERAGES = {"dist_ma20": 20, "dist_ma60": 60, "dist_ma200": 200}

# Weights blend short, medium and long horizons so no single window dominates.
MOMENTUM_WEIGHTS = {"ret_20d": 0.20, "ret_60d": 0.35, "ret_120d": 0.45}

MIN_HISTORY = 130
EVIDENCE_THRESHOLD = 1.5


class ScannerAgent(Agent):
    name = "scanner"
    version = "1.0"
    sources = ("prices",)

    def __init__(self, min_history: int = MIN_HISTORY) -> None:
        self.min_history = min_history

    def run(self, context: MarketContext, board: Blackboard) -> int:
        prices = context.visible_prices()
        if prices.empty:
            board.record_degradation("scanner: 無可見價格")
            return 0

        board.record_freshness("prices", prices.index.max(), expected_lag_hours=30.0)

        # Drop names without enough history rather than imputing one. A short
        # series produces a confident-looking but meaningless z-score.
        usable = prices.loc[:, prices.notna().sum() >= self.min_history]
        if usable.empty:
            board.record_degradation("scanner: 所有標的歷史長度不足")
            return 0

        filled = usable.ffill()
        returns = filled.pct_change(fill_method=None)
        latest = filled.index.max()

        features = pd.DataFrame(index=usable.columns)
        features["last_close"] = filled.iloc[-1]
        features["ret_1d"] = returns.iloc[-1]

        for label, window in RETURN_HORIZONS.items():
            if len(filled) > window:
                features[label] = filled.pct_change(window, fill_method=None).iloc[-1]

        for label, window in MOVING_AVERAGES.items():
            if len(filled) >= window:
                average = filled.rolling(window, min_periods=window // 2).mean().iloc[-1]
                features[label] = filled.iloc[-1] / average - 1.0

        tail = returns.tail(260)
        features["vol_ewma"] = ewma_volatility_frame(tail)
        features["vol_20d"] = realised_volatility_frame(tail, 20)
        features["vol_252d"] = realised_volatility_frame(tail, 252)
        features["vol_downside"] = downside_volatility_frame(tail, 60)
        features["vol_regime_ratio"] = features["vol_20d"] / features["vol_252d"].replace(0.0, np.nan)
        features["drawdown_120d"] = max_drawdown_frame(filled.tail(120))

        self._add_relative_strength(features, filled, context.benchmark, board)
        self._add_composite(features)

        board.write_features(features, agent=self.name)
        self._mark_short_history(prices, usable, board)

        return self._emit_evidence(features, context, board, latest)

    def _add_relative_strength(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark: str,
        board: Blackboard,
    ) -> None:
        """Excess return over the benchmark, which is what alpha actually means."""
        if benchmark not in prices.columns:
            board.record_degradation(f"scanner: 基準 {benchmark} 不在價格資料中，跳過相對強弱")
            return
        board.market["benchmark"] = benchmark
        for label, window in (("rs_20d", 20), ("rs_60d", 60)):
            if len(prices) <= window:
                continue
            asset = prices.pct_change(window, fill_method=None).iloc[-1]
            reference = float(asset.get(benchmark, np.nan))
            if not np.isfinite(reference):
                continue
            features[label] = asset - reference
            board.market[f"benchmark_{label}"] = reference

    def _add_composite(self, features: pd.DataFrame) -> None:
        """Rank-normalised momentum, then volatility-adjusted.

        Ranks are used instead of raw returns so a single gap cannot dominate,
        and the volatility adjustment stops the score from simply picking the
        most volatile names in the universe.
        """
        available = {
            label: weight
            for label, weight in MOMENTUM_WEIGHTS.items()
            if label in features.columns
        }
        if not available:
            return
        total = sum(available.values())
        composite = pd.Series(0.0, index=features.index)
        coverage = pd.Series(True, index=features.index)
        for label, weight in available.items():
            scores = rank_to_normal(features[label])
            composite += scores.fillna(0.0) * (weight / total)
            coverage &= features[label].notna()
        features["momentum_score"] = composite.where(coverage)

        volatility = features.get("vol_ewma")
        if volatility is not None:
            adjusted = features["momentum_score"] / volatility.replace(0.0, np.nan)
            features["momentum_risk_adjusted"] = rank_to_normal(adjusted)

        if "rs_60d" in features.columns:
            features["rs_rank"] = rank_to_normal(features["rs_60d"])

    def _mark_short_history(
        self,
        prices: pd.DataFrame,
        usable: pd.DataFrame,
        board: Blackboard,
    ) -> None:
        """Record excluded names explicitly instead of letting them vanish."""
        excluded = [column for column in prices.columns if column not in usable.columns]
        board.market["short_history_excluded"] = excluded
        for symbol in excluded:
            board.write_feature(symbol, "momentum_score", float("nan"), self.name, Quality.MISSING)

    def _emit_evidence(
        self,
        features: pd.DataFrame,
        context: MarketContext,
        board: Blackboard,
        latest: pd.Timestamp,
    ) -> int:
        if "momentum_score" not in features.columns:
            return 0
        findings = 0
        scores = features["momentum_score"].dropna()
        for symbol, score in scores.items():
            if abs(score) < EVIDENCE_THRESHOLD:
                continue
            direction = 1 if score > 0 else -1
            rs = features.at[symbol, "rs_60d"] if "rs_60d" in features.columns else np.nan
            detail = f"多週期動能分數 {score:+.2f}"
            if np.isfinite(rs):
                detail += f"，60 日相對基準 {rs:+.1%}"
            board.view(str(symbol), context.name_of(str(symbol)))
            board.add_evidence(
                str(symbol),
                Evidence(
                    kind="momentum",
                    detail=detail,
                    weight=0.25,
                    direction=direction,
                    agent=self.name,
                    value=float(score),
                ),
            )
            board.tag(str(symbol), "momentum")
            findings += 1
        board.market["scan_date"] = latest
        board.market["scanned_symbols"] = int(len(features))
        return findings
