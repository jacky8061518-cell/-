from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pytest
import structlog

from trading_intel.core.clock import UTC, SimulatedClock, use_clock
from trading_intel.core.ids import CorrelationId
from trading_intel.core.logging import bind_correlation, configure_logging, get_logger

FIXED = datetime(2020, 3, 19, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def test_dev_configuration_uses_the_console_renderer() -> None:
    configure_logging("dev")
    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.dev.ConsoleRenderer)


def test_prod_configuration_uses_the_json_renderer() -> None:
    configure_logging("prod")
    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.processors.JSONRenderer)


def test_level_is_respected(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("prod", level="ERROR")
    logger = get_logger("test")
    logger.info("suppressed")
    logger.error("emitted")
    out = capsys.readouterr().out
    assert "suppressed" not in out
    assert "emitted" in out


def test_timestamp_comes_from_the_active_clock(capsys: pytest.CaptureFixture[str]) -> None:
    """A backtest's logs must be stamped with simulated time, not wall time."""
    configure_logging("prod")
    with use_clock(SimulatedClock(FIXED)):
        get_logger("test").warning("inside the backtest")
    out = capsys.readouterr().out
    assert FIXED.isoformat() in out


def test_bind_correlation_attaches_to_every_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("prod")
    bind_correlation(CorrelationId("cid-123"))
    get_logger("test").warning("first")
    get_logger("test").warning("second")
    out = capsys.readouterr().out
    assert out.count("cid-123") == 2


def test_get_logger_returns_a_bound_logger() -> None:
    configure_logging("dev")
    logger = get_logger("trading_intel.test")
    assert hasattr(logger, "bind")
