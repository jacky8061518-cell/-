"""structlog wiring.

Every log line carries a UTC timestamp taken from the active clock, so lines
emitted inside a backtest are stamped with simulated time and line up with the
events they describe.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from trading_intel.core.clock import utc_now
from trading_intel.core.ids import CorrelationId

_CORRELATION_KEY = "correlation_id"


def _add_timestamp(
    logger: Any,  # noqa: ARG001  (structlog processor signature)
    method_name: str,  # noqa: ARG001
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Stamp with the active clock, never with wall time directly."""
    event_dict["timestamp"] = utc_now().isoformat()
    return event_dict


def configure_logging(env: str, level: str = "INFO") -> None:
    """Human-readable output in dev, JSON everywhere else."""
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer(colors=True)
        if env == "dev"
        else structlog.processors.JSONRenderer(sort_keys=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_timestamp,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_correlation(cid: CorrelationId) -> None:
    """Attach ``cid`` to every log line emitted in this context."""
    structlog.contextvars.bind_contextvars(**{_CORRELATION_KEY: str(cid)})


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Bind the module name explicitly; the print-based factory has no name of its own."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name).bind(logger=name)
    return logger
