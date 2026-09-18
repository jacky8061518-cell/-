"""Agent protocol and the guarded runner that executes them.

Every agent declares a name, a version, and the sources it depends on. The
runner times each execution, catches failures, and records a health record.
An agent that raises does not take the pipeline down: it is marked failed, the
run continues with the remaining evidence, and the affected signals inherit a
degraded data-quality penalty.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..blackboard import Blackboard
from ..contracts import AgentRun


@dataclass(frozen=True)
class MarketContext:
    """Immutable inputs handed to every agent. Agents never fetch data themselves.

    Keeping data access out of the agents is what allows backtest and live to
    walk the identical code path: only the construction of this context differs.
    """

    as_of: pd.Timestamp
    prices: pd.DataFrame
    metadata: pd.DataFrame
    benchmark: str
    flows: pd.DataFrame | None = None
    news: pd.DataFrame | None = None
    mode: str = "paper"
    extras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.prices.empty:
            raise ValueError("MarketContext requires price history")
        if self.mode not in {"backtest", "paper", "live"}:
            raise ValueError(f"unknown mode: {self.mode}")

    def visible_prices(self) -> pd.DataFrame:
        """Prices up to and including as_of.

        The point-in-time cut lives here, in one place, so no agent can
        accidentally read a bar that had not printed at decision time.
        """
        index = pd.to_datetime(self.prices.index)
        return self.prices.loc[index <= self.as_of]

    def visible_flows(self) -> pd.DataFrame:
        if self.flows is None or self.flows.empty:
            return pd.DataFrame()
        dates = pd.to_datetime(self.flows["Date"])
        return self.flows.loc[dates <= self.as_of]

    def visible_news(self) -> pd.DataFrame:
        if self.news is None or self.news.empty:
            return pd.DataFrame()
        published = pd.to_datetime(self.news["published_at"], errors="coerce")
        return self.news.loc[published <= self.as_of]

    def name_of(self, symbol: str) -> str:
        if self.metadata.empty or symbol not in self.metadata.index:
            return symbol
        return str(self.metadata.at[symbol, "Name"])


class Agent(ABC):
    """Base class for every agent. Single responsibility, declared dependencies."""

    name: str = "agent"
    version: str = "0.0"
    uses_llm: bool = False
    sources: tuple[str, ...] = ()

    @abstractmethod
    def run(self, context: MarketContext, board: Blackboard) -> int:
        """Write findings onto the blackboard and return how many were produced."""

    def describe(self) -> str:
        kind = "語意" if self.uses_llm else "確定性"
        return f"{self.name} v{self.version}（{kind}）"


class AgentRunner:
    """Executes agents in dependency order, isolating failures from the pipeline."""

    def __init__(self, agents: list[Agent], timeout_ms: float = 15_000) -> None:
        self.agents = agents
        self.timeout_ms = timeout_ms

    def run_all(self, context: MarketContext, board: Blackboard) -> list[AgentRun]:
        records: list[AgentRun] = []
        for agent in self.agents:
            started = datetime.now(timezone.utc)
            clock = time.perf_counter()
            status, findings, error, degraded = "ok", 0, None, None
            try:
                findings = int(agent.run(context, board))
            except Exception as exc:  # Isolate: one broken agent must not stop the desk.
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                board.record_degradation(f"{agent.name} 失效：{error}")
            duration_ms = (time.perf_counter() - clock) * 1000.0
            if status == "ok" and duration_ms > self.timeout_ms:
                status = "slow"
                degraded = f"耗時 {duration_ms:.0f}ms 超過 {self.timeout_ms:.0f}ms 上限"
                board.record_degradation(f"{agent.name} {degraded}")
            records.append(
                AgentRun(
                    agent=agent.name,
                    version=agent.version,
                    started_at=started,
                    duration_ms=duration_ms,
                    status=status,
                    symbols_processed=len(board.symbols),
                    findings=findings,
                    degraded_reason=degraded,
                    error=error,
                )
            )
        return records
