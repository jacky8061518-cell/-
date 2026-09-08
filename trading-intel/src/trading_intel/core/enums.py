"""Closed vocabularies shared across the whole system.

Values equal their names so that serialisation, database storage, and log
output all read the same.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Market(StrEnum):
    TW = "TW"
    US = "US"


class TradingState(StrEnum):
    TRADABLE = "TRADABLE"
    NO_TRADE = "NO_TRADE"
    HALTED = "HALTED"
    RESTRICTED = "RESTRICTED"


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Horizon(StrEnum):
    INTRADAY = "INTRADAY"
    DAYS = "DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"


class Severity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class DecisionLevel(IntEnum):
    """Ordered on purpose: downgrades are expressed as ``max(...)`` comparisons."""

    AUTO = 1
    CONFIRM = 2
    RESEARCH_ONLY = 3


class QualityCheck(StrEnum):
    FRESHNESS = "FRESHNESS"
    COMPLETENESS = "COMPLETENESS"
    SANITY = "SANITY"
    CONSISTENCY = "CONSISTENCY"
    DRIFT = "DRIFT"


class DocType(StrEnum):
    NEWS = "NEWS"
    ANNOUNCEMENT = "ANNOUNCEMENT"
    FINANCIAL_REPORT = "FINANCIAL_REPORT"
    EARNINGS_CALL = "EARNINGS_CALL"
    BROKER_REPORT = "BROKER_REPORT"
