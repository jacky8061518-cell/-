"""Exception hierarchy for trading-intel.

Every exception carries a ``context`` mapping so structured logging can emit the
failure with machine-readable fields instead of a formatted string.
"""

from __future__ import annotations

from typing import Any


class TradingIntelError(Exception):
    """Base class for every error raised by this project."""

    def __init__(self, message: str = "", /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})" if self.message else rendered

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class ConfigError(TradingIntelError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ClockError(TradingIntelError):
    """Base class for clock-related failures."""


class NaiveDatetimeError(ClockError):
    """A datetime without tzinfo reached code that requires an aware value."""


class ClockRewindError(ClockError):
    """An attempt was made to move a simulated clock backwards."""


class TemporalIntegrityError(TradingIntelError):
    """Timestamps violate the required ordering (e.g. ingest before event)."""


class LookaheadError(TradingIntelError):
    """Data dated after the current ``asof`` boundary was read."""


class NetworkAccessDenied(TradingIntelError):  # noqa: N818  (name fixed by spec)
    """Network access was attempted inside a no-network sandbox."""


class DataQualityError(TradingIntelError):
    """A data quality check failed hard enough to stop processing."""


class SchemaValidationError(TradingIntelError):
    """Payload did not satisfy its declared schema."""


class BudgetExceededError(TradingIntelError):
    """A token, call-rate, or time budget was exhausted."""
