"""Top-level orchestration: one function that runs the whole desk.

Backtest, paper and live all call this with a different ``MarketContext``. There
is no second code path, because a branch here is exactly where a backtest starts
quietly disagreeing with production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .agents import Agent, AgentRunner, MarketContext, default_pipeline
from .alerts import AlertRouter
from .blackboard import Blackboard
from .contracts import DeskState, Position, stable_id, utc_now
from .decision import DecisionConfig, DecisionLayer
from .risk import BreakerStatus, RiskEngine, RiskLimits, build_clusters


@dataclass
class DeskConfig:
    """Everything a run needs that is not market data."""

    decision: DecisionConfig = field(default_factory=DecisionConfig)
    limits: RiskLimits = field(default_factory=RiskLimits)
    agents: list[Agent] | None = None
    daily_pnl_pct: float = 0.0
    rolling_5d_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    data_delay_seconds: float = 0.0


def run_desk(
    context: MarketContext,
    config: DeskConfig | None = None,
    positions: list[Position] | None = None,
) -> DeskState:
    """Execute one full pipeline pass and return everything the console renders."""
    config = config or DeskConfig()
    started = time.perf_counter()
    as_of = pd.Timestamp(context.as_of)
    run_id = stable_id("run", context.mode, as_of.isoformat())

    board = Blackboard(as_of=as_of.to_pydatetime())
    agents = config.agents if config.agents is not None else default_pipeline()
    agent_runs = AgentRunner(agents).run_all(context, board)

    risk_engine = RiskEngine(limits=config.limits)
    breaker = risk_engine.evaluate_breakers(
        daily_pnl_pct=config.daily_pnl_pct,
        rolling_5d_pnl_pct=config.rolling_5d_pnl_pct,
        drawdown_pct=config.drawdown_pct,
        data_delay_seconds=config.data_delay_seconds,
    )

    decision = DecisionLayer(config=config.decision, risk_engine=risk_engine)
    signals, suppressed = decision.build(board, run_id=run_id, breaker=breaker)

    signals = _apply_portfolio_limits(signals, context, risk_engine, positions)

    alerts = AlertRouter().route(
        signals,
        breaker=breaker,
        degradations=board.degradations,
        stale_sources=board.stale_sources,
    )

    state = DeskState(
        run_id=run_id,
        as_of=as_of.to_pydatetime(),
        universe_size=int(board.market.get("scanned_symbols", len(board.symbols))),
        regime=board.regime,
        signals=signals,
        alerts=alerts,
        agent_runs=agent_runs,
        narratives=board.narratives,
        data_freshness=board.source_freshness,
        suppressed_signals=suppressed,
        breaker=breaker,
        board=board,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        market={
            key: value
            for key, value in board.market.items()
            if not isinstance(value, pd.DataFrame)
        },
    )
    return state


def _apply_portfolio_limits(
    signals,
    context: MarketContext,
    risk_engine: RiskEngine,
    positions: list[Position] | None,
):
    """Trim sizes so correlated ideas cannot add up to one oversized bet."""
    tradeable = [signal for signal in signals if signal.risk.is_tradeable]
    if not tradeable:
        return signals

    symbols = [signal.symbol for signal in tradeable]
    prices = context.visible_prices()
    available = [symbol for symbol in symbols if symbol in prices.columns]
    returns = prices[available].ffill().pct_change(fill_method=None).tail(120)
    clusters = build_clusters(returns)

    proposals = [(signal.symbol, signal.direction, signal.risk) for signal in tradeable]
    approved, notes = risk_engine.apply_portfolio_limits(proposals, clusters, positions)

    from .contracts import RiskBudget

    for signal in tradeable:
        allowed = approved.get(signal.symbol, 0.0)
        if allowed == signal.risk.suggested_weight_pct:
            continue
        reason = next(
            (note for note in notes if note.startswith(f"{signal.symbol}：")),
            "組合層限制",
        )
        signal.risk = RiskBudget(
            suggested_weight_pct=allowed,
            stop_level=signal.risk.stop_level,
            stop_basis=signal.risk.stop_basis,
            max_loss_pct_nav=round(
                signal.risk.max_loss_pct_nav
                * (allowed / signal.risk.suggested_weight_pct)
                if signal.risk.suggested_weight_pct
                else 0.0,
                3,
            ),
            target_level=signal.risk.target_level,
            reward_risk=signal.risk.reward_risk,
            binding_constraint=reason.split("：", 1)[-1],
        )
        signal.model_provenance["cluster"] = clusters.get(signal.symbol, signal.symbol)
    return signals
