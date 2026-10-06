"""Data contracts shared by every layer of the trading desk.

These dataclasses are the only structures that cross layer boundaries. Agents
write evidence, the decision layer assembles signal cards, the risk engine
attaches a budget, and the console renders them. Nothing downstream is allowed
to invent fields that are not declared here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    """Single source of wall-clock time so tests can monkeypatch one function."""
    return datetime.now(timezone.utc)


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic identifier so replaying the same inputs yields the same id."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


class Severity(IntEnum):
    """Alert priority. Lower is more urgent, matching the P0..P4 convention."""

    P0_CRISIS = 0
    P1_ACTION = 1
    P2_ATTENTION = 2
    P3_INTEL = 3
    P4_LOG = 4

    @property
    def label(self) -> str:
        return {
            Severity.P0_CRISIS: "P0 危機",
            Severity.P1_ACTION: "P1 行動",
            Severity.P2_ATTENTION: "P2 注意",
            Severity.P3_INTEL: "P3 情報",
            Severity.P4_LOG: "P4 日誌",
        }[self]

    @property
    def color(self) -> str:
        return {
            Severity.P0_CRISIS: "#EF4444",
            Severity.P1_ACTION: "#F97316",
            Severity.P2_ATTENTION: "#FACC15",
            Severity.P3_INTEL: "#38BDF8",
            Severity.P4_LOG: "#64748B",
        }[self]


class Quality(IntEnum):
    """Per-feature data quality. Never silently treat missing data as zero."""

    OK = 0
    STALE = 1
    IMPUTED = 2
    MISSING = 3

    @property
    def label(self) -> str:
        return {
            Quality.OK: "正常",
            Quality.STALE: "過期",
            Quality.IMPUTED: "推估",
            Quality.MISSING: "缺漏",
        }[self]


@dataclass(frozen=True)
class Evidence:
    """One piece of support for a signal, always traceable to its producer."""

    kind: str
    detail: str
    weight: float
    direction: int
    agent: str
    value: float | None = None

    def __post_init__(self) -> None:
        if not -1 <= self.direction <= 1:
            raise ValueError("direction must be -1, 0 or 1")
        if self.weight < 0:
            raise ValueError("evidence weight cannot be negative")


@dataclass(frozen=True)
class CounterEvidence:
    """Evidence pointing against the signal. Never hide this from the trader."""

    detail: str
    severity: str
    agent: str

    def __post_init__(self) -> None:
        if self.severity not in {"low", "medium", "high"}:
            raise ValueError(f"unknown counter-evidence severity: {self.severity}")


@dataclass(frozen=True)
class DataQuality:
    """Freshness and consistency of everything the signal depends on."""

    all_sources_fresh: bool
    stale_sources: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def penalty(self) -> float:
        """Conviction multiplier applied when inputs are degraded."""
        factor = 1.0
        factor *= 0.85 ** len(self.stale_sources)
        factor *= 0.75 ** len(self.conflicts)
        return round(factor, 4)


@dataclass(frozen=True)
class RiskBudget:
    """What the risk engine actually authorised, and why it was capped."""

    suggested_weight_pct: float
    stop_level: float | None
    stop_basis: str
    max_loss_pct_nav: float
    target_level: float | None
    reward_risk: float | None
    binding_constraint: str

    @property
    def is_tradeable(self) -> bool:
        return self.suggested_weight_pct > 0


@dataclass(frozen=True)
class RegimeState:
    """Market-wide context. The same signal is worth different amounts by regime."""

    label: str
    trend: str
    volatility: str
    breadth: float
    risk_appetite: str
    detail: str
    as_of: datetime

    def describe(self) -> str:
        return f"{self.label}（{self.trend}／{self.volatility}／廣度 {self.breadth:.0%}）"


@dataclass
class SignalCard:
    """The unit of work handed to a trader. Every field is mandatory by design."""

    signal_id: str
    decision_id: str
    created_at: datetime
    symbol: str
    name: str
    direction: str
    conviction: float
    horizon: str
    thesis: str
    evidence: list[Evidence]
    counter_evidence: list[CounterEvidence]
    regime_context: str
    risk: RiskBudget
    invalidation: str
    data_quality: DataQuality
    model_provenance: dict[str, Any]
    severity: Severity = Severity.P2_ATTENTION
    status: str = "proposed"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.direction not in {"long", "short", "flat"}:
            raise ValueError(f"unknown direction: {self.direction}")
        if not 0 <= self.conviction <= 1:
            raise ValueError("conviction must sit in [0, 1]")
        if not self.evidence:
            raise ValueError("a signal without evidence must never be emitted")
        if not self.invalidation:
            raise ValueError("a signal without an invalidation condition cannot be managed")

    @property
    def direction_label(self) -> str:
        return {"long": "做多", "short": "做空", "flat": "觀望"}[self.direction]

    @property
    def evidence_balance(self) -> float:
        """Weighted agreement across evidence, in [-1, 1]."""
        total = sum(item.weight for item in self.evidence)
        if total <= 0:
            return 0.0
        return sum(item.weight * item.direction for item in self.evidence) / total

    @property
    def replay_command(self) -> str:
        return f"replay --decision-id {self.decision_id}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["severity"] = int(self.severity)
        payload["replay_command"] = self.replay_command
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


@dataclass
class Alert:
    """Anything that competes for trader attention, including system health."""

    alert_id: str
    severity: Severity
    channel: str
    title: str
    body: str
    created_at: datetime
    source: str
    symbol: str | None = None
    signal_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["severity"] = int(self.severity)
        payload["severity_label"] = self.severity.label
        return payload


@dataclass
class AgentRun:
    """Observability record for one agent execution. No metrics, no production."""

    agent: str
    version: str
    started_at: datetime
    duration_ms: float
    status: str
    symbols_processed: int
    findings: int
    degraded_reason: str | None = None
    error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.status == "ok"


@dataclass
class Position:
    """A held position tracked for risk, independent of any broker connection."""

    symbol: str
    name: str
    direction: str
    weight_pct: float
    entry_price: float
    entry_date: datetime
    stop_level: float | None
    target_level: float | None
    signal_id: str | None = None
    cluster: str = "未分類"

    def unrealised_pct(self, last_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        raw = (last_price - self.entry_price) / self.entry_price
        return raw if self.direction == "long" else -raw

    def stop_distance_pct(self, last_price: float) -> float | None:
        if self.stop_level is None or last_price <= 0:
            return None
        return abs(last_price - self.stop_level) / last_price


@dataclass
class DeskState:
    """Everything one pipeline run produced. This is what the console renders."""

    run_id: str
    as_of: datetime
    universe_size: int
    regime: RegimeState
    signals: list[SignalCard] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    agent_runs: list[AgentRun] = field(default_factory=list)
    narratives: list[dict[str, Any]] = field(default_factory=list)
    data_freshness: dict[str, Any] = field(default_factory=dict)
    suppressed_signals: int = 0
    breaker: Any = None
    board: Any = None
    elapsed_seconds: float = 0.0
    market: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return all(run.healthy for run in self.agent_runs)

    def signals_by_severity(self, severity: Severity) -> list[SignalCard]:
        return [signal for signal in self.signals if signal.severity == severity]
