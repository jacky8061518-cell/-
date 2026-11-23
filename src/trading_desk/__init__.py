"""Autonomous multi-agent market intelligence desk.

Layering, from the bottom up:

``loaders``      point-in-time data access, the only module that touches paths
``statistics``   robust estimators shared by every deterministic agent
``news``         article ingestion, deduplication and citable annotation
``blackboard``   shared per-run state agents write to
``agents``       single-responsibility scanners, each independently testable
``decision``     evidence to signal cards, deterministic and replayable
``risk``         the independent gate that decides size, or refuses
``alerts``       attention budgeting
``desk``         one entry point used identically by backtest, paper and live
"""

from .contracts import Alert, DeskState, Position, Severity, SignalCard
from .desk import DeskConfig, run_desk

__all__ = [
    "Alert",
    "DeskConfig",
    "DeskState",
    "Position",
    "Severity",
    "SignalCard",
    "run_desk",
]
