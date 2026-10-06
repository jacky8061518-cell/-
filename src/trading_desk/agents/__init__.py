"""Agent implementations. Each agent has one job and declares its dependencies."""

from .anomaly import AnomalyAgent
from .base import Agent, AgentRunner, MarketContext
from .corroborator import CorroboratorAgent
from .flow import FlowAgent
from .narrative import NarrativeAgent
from .regime import RegimeAgent
from .scanner import ScannerAgent
from .sentiment import SentimentAgent

__all__ = [
    "Agent",
    "AgentRunner",
    "AnomalyAgent",
    "CorroboratorAgent",
    "FlowAgent",
    "MarketContext",
    "NarrativeAgent",
    "RegimeAgent",
    "ScannerAgent",
    "SentimentAgent",
]


def default_pipeline() -> list[Agent]:
    """The standard execution order.

    Strictly a DAG: cheap deterministic agents populate the blackboard first,
    the expensive semantic agent only sees their shortlist, and the two
    synthesis agents run last over everything that was written.
    """
    return [
        ScannerAgent(),
        RegimeAgent(),
        AnomalyAgent(),
        FlowAgent(),
        SentimentAgent(),
        CorroboratorAgent(),
        NarrativeAgent(),
    ]
