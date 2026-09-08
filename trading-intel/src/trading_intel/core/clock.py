"""The only module in this project allowed to read the current time.

Everything else calls :func:`utc_now`. The active clock lives in a
:class:`~contextvars.ContextVar` rather than a module global so that async tasks
and parallel tests each see their own clock.

``ruff`` bans ``datetime.now`` / ``datetime.utcnow`` / ``datetime.today`` /
``time.time`` project-wide and exempts this file only; ``tests/guard`` enforces
the same rule with an AST walk.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Final, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from trading_intel.core.enums import Market
from trading_intel.core.errors import ClockRewindError, NaiveDatetimeError

UTC: Final = _dt.UTC

#: How far ``ingest_time`` may precede ``event_time`` before we call it a bug
#: rather than clock drift between machines.
MAX_CLOCK_SKEW: Final = timedelta(seconds=5)

#: Local close time per market. The trading-day boundary is the close, not UTC
#: midnight: a US close at 16:00 New York is 20:00/21:00 UTC, so slicing on the
#: UTC date would push the whole afternoon session into the next day.
MARKET_CLOSE: Final[dict[Market, tuple[str, time]]] = {
    Market.TW: ("Asia/Taipei", time(13, 30)),
    Market.US: ("America/New_York", time(16, 0)),
}


@runtime_checkable
class Clock(Protocol):
    """Anything that can tell the time in tz-aware UTC."""

    def now(self) -> datetime: ...


@dataclass(frozen=True)
class SystemClock:
    """Wall-clock time. The single place a real ``now()`` call is made."""

    def now(self) -> datetime:
        return _dt.datetime.now(tz=UTC)


@dataclass
class SimulatedClock:
    """A clock under test/backtest control. Monotonic by construction."""

    current: datetime

    def __post_init__(self) -> None:
        self.current = ensure_utc(self.current)

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ClockRewindError(
                "simulated clock cannot advance by a negative delta",
                current=self.current.isoformat(),
                delta=str(delta),
            )
        self.current = self.current + delta

    def set_to(self, ts: datetime) -> None:
        target = ensure_utc(ts)
        if target < self.current:
            raise ClockRewindError(
                "simulated clock cannot be moved backwards",
                current=self.current.isoformat(),
                target=target.isoformat(),
            )
        self.current = target


# The default is a frozen, stateless dataclass, so sharing one instance is safe.
_DEFAULT_CLOCK: Final[Clock] = SystemClock()
_active_clock: ContextVar[Clock] = ContextVar(
    "_active_clock",
    default=_DEFAULT_CLOCK,
)


def get_clock() -> Clock:
    """Return the clock in effect for the current context."""
    return _active_clock.get()


@contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """Install ``clock`` for the duration of the block, then restore the previous one."""
    token = _active_clock.set(clock)
    try:
        yield clock
    finally:
        _active_clock.reset(token)


def utc_now() -> datetime:
    """The only sanctioned way to ask what time it is."""
    return ensure_utc(get_clock().now())


def ensure_utc(ts: datetime) -> datetime:
    """Normalise an aware datetime to UTC; reject naive ones outright."""
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise NaiveDatetimeError(
            "naive datetime is not accepted; attach a timezone",
            value=ts.isoformat(),
        )
    return ts.astimezone(UTC)


def to_display_tz(ts: datetime, tz: str = "Asia/Taipei") -> datetime:
    """Convert to a human-facing timezone. For the presentation layer only."""
    return ensure_utc(ts).astimezone(ZoneInfo(tz))


def trading_day(ts: datetime, market: Market) -> date:
    """Return the trading day ``ts`` belongs to for ``market``.

    The boundary is the market close in local time: anything at or before the
    close belongs to that local date, anything after it belongs to the next
    calendar day (holiday calendars are out of Phase 0 scope).
    """
    tz_name, close = MARKET_CLOSE[market]
    local = ensure_utc(ts).astimezone(ZoneInfo(tz_name))
    day = local.date()
    if local.timetz().replace(tzinfo=None) > close:
        day = day + timedelta(days=1)
    return day
