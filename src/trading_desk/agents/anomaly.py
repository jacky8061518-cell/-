"""Anomaly agent: how unusual is today, relative to this name's own history.

Deterministic. An anomaly is not a signal — it is a cheap trigger that decides
which of thousands of names deserve expensive downstream attention. Keeping
this gate strict is the main cost-control lever in the whole system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..blackboard import Blackboard
from ..contracts import CounterEvidence, Evidence
from ..statistics import market_neutral_returns, robust_zscore_frame
from .base import Agent, MarketContext

RETURN_Z_WINDOW = 60
RETURN_Z_TRIGGER = 3.0
GAP_TRIGGER = 0.05

# A statistically extreme move that is economically trivial is not tradeable.
# Both gates must pass: unusual for this name, and large enough to act on.
MIN_EXCESS_MOVE = 0.03
# Daily residual volatility floor, roughly the quietest liquid name on the board.
MIN_DAILY_SIGMA = 0.006
VOL_REGIME_TRIGGER = 1.5
DRAWDOWN_TRIGGER = -0.20


class AnomalyAgent(Agent):
    name = "anomaly"
    version = "1.0"
    sources = ("prices",)

    def __init__(
        self,
        z_trigger: float = RETURN_Z_TRIGGER,
        window: int = RETURN_Z_WINDOW,
    ) -> None:
        self.z_trigger = z_trigger
        self.window = window

    def run(self, context: MarketContext, board: Blackboard) -> int:
        prices = context.visible_prices()
        if prices.empty:
            return 0
        usable = prices.loc[:, prices.notna().sum() >= self.window + 5]
        if usable.empty:
            board.record_degradation("anomaly: 歷史長度不足以計算穩健 z 分數")
            return 0

        filled = usable.ffill()
        returns = filled.pct_change(fill_method=None)

        # Neutralise the market factor first. Without this, a broad selloff
        # flags the entire universe as anomalous and the gate stops gating.
        residual = market_neutral_returns(returns.tail(self.window + 60))
        z_scores = robust_zscore_frame(
            residual, self.window, min_sigma=MIN_DAILY_SIGMA
        ).iloc[-1]
        latest_return = returns.iloc[-1]
        market_move = float(returns.iloc[-1].median())
        board.market["market_move"] = market_move
        board.market["residual_basis"] = "橫斷面中位數"

        findings = 0
        findings += self._flag_return_shocks(
            z_scores, latest_return, market_move, context, board
        )
        findings += self._flag_volatility_regime(context, board)
        findings += self._flag_deep_drawdowns(context, board)

        board.write_features(
            pd.DataFrame({"return_zscore": z_scores}).dropna(),
            agent=self.name,
        )
        board.market["anomaly_candidates"] = findings
        return findings

    def _flag_return_shocks(
        self,
        z_scores: pd.Series,
        latest_return: pd.Series,
        market_move: float,
        context: MarketContext,
        board: Blackboard,
    ) -> int:
        """A move the name's own history cannot explain, once the market is removed."""
        flagged = z_scores.dropna()
        flagged = flagged[flagged.abs() >= self.z_trigger]
        findings = 0
        for symbol, score in flagged.items():
            move = float(latest_return.get(symbol, np.nan))
            if not np.isfinite(move):
                continue
            excess = move - market_move
            if abs(excess) < MIN_EXCESS_MOVE:
                continue
            direction = 1 if score > 0 else -1
            board.view(str(symbol), context.name_of(str(symbol)))
            board.add_evidence(
                str(symbol),
                Evidence(
                    kind="anomaly",
                    detail=(
                        f"單日報酬 {move:+.2%}（大盤 {market_move:+.2%}，"
                        f"個股超額 {excess:+.2%}），市場中性化後 z 分數 {score:+.1f}"
                    ),
                    weight=0.20,
                    direction=direction,
                    agent=self.name,
                    value=float(score),
                ),
            )
            board.tag(str(symbol), "anomaly")
            if abs(excess) >= GAP_TRIGGER:
                # A gap this size usually means news the price agent cannot see.
                board.add_counter_evidence(
                    str(symbol),
                    CounterEvidence(
                        detail=f"單日超額跳空 {excess:+.2%}，可能已反映未知消息，追價風險高",
                        severity="medium",
                        agent=self.name,
                    ),
                )
            findings += 1
        return findings

    def _flag_volatility_regime(self, context: MarketContext, board: Blackboard) -> int:
        """Short-vol over long-vol above the trigger means the model's world changed."""
        findings = 0
        for view in board.views():
            if not view.is_usable("vol_regime_ratio"):
                continue
            ratio = view.feature("vol_regime_ratio")
            if ratio < VOL_REGIME_TRIGGER:
                continue
            board.tag(view.symbol, "vol_regime_shift")
            if view.evidence:
                board.add_counter_evidence(
                    view.symbol,
                    CounterEvidence(
                        detail=f"短期波動為長期的 {ratio:.1f} 倍，已進入高波動狀態，部位需縮小",
                        severity="high" if ratio >= 2.0 else "medium",
                        agent=self.name,
                    ),
                )
                findings += 1
        return findings

    def _flag_deep_drawdowns(self, context: MarketContext, board: Blackboard) -> int:
        """A name still deep in drawdown needs that stated next to any long thesis."""
        findings = 0
        for view in board.views():
            if not view.evidence or not view.is_usable("drawdown_120d"):
                continue
            drawdown = view.feature("drawdown_120d")
            if drawdown > DRAWDOWN_TRIGGER:
                continue
            board.add_counter_evidence(
                view.symbol,
                CounterEvidence(
                    detail=f"120 日內最大回撤 {drawdown:.1%}，趨勢尚未修復",
                    severity="medium",
                    agent=self.name,
                ),
            )
            findings += 1
        return findings
