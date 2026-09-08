from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_intel.core.clock import (
    UTC,
    SimulatedClock,
    SystemClock,
    ensure_utc,
    get_clock,
    to_display_tz,
    trading_day,
    use_clock,
    utc_now,
)
from trading_intel.core.enums import Market
from trading_intel.core.errors import ClockRewindError, NaiveDatetimeError

FIXED = datetime(2020, 3, 19, 12, 0, tzinfo=UTC)


def test_system_clock_returns_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_default_clock_is_the_system_clock() -> None:
    assert isinstance(get_clock(), SystemClock)


def test_use_clock_restores_the_previous_clock() -> None:
    before = get_clock()
    with use_clock(SimulatedClock(FIXED)) as clock:
        assert get_clock() is clock
    assert get_clock() is before


def test_use_clock_restores_even_when_the_block_raises() -> None:
    before = get_clock()
    with pytest.raises(RuntimeError), use_clock(SimulatedClock(FIXED)):
        raise RuntimeError("boom")
    assert get_clock() is before


def test_use_clock_nests() -> None:
    outer = SimulatedClock(FIXED)
    inner = SimulatedClock(FIXED + timedelta(days=1))
    with use_clock(outer):
        assert utc_now() == FIXED
        with use_clock(inner):
            assert utc_now() == FIXED + timedelta(days=1)
        assert utc_now() == FIXED


def test_utc_now_uses_the_simulated_clock_not_wall_time() -> None:
    with use_clock(SimulatedClock(FIXED)):
        assert utc_now() == FIXED
        assert utc_now().year == 2020


def test_simulated_clock_requires_an_aware_datetime() -> None:
    with pytest.raises(NaiveDatetimeError):
        SimulatedClock(datetime(2020, 3, 19, 12, 0))  # noqa: DTZ001


def test_simulated_clock_normalises_to_utc() -> None:
    taipei = datetime(2020, 3, 19, 20, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    assert SimulatedClock(taipei).now() == FIXED


def test_advance_moves_forward() -> None:
    clock = SimulatedClock(FIXED)
    clock.advance(timedelta(hours=3))
    assert clock.now() == FIXED + timedelta(hours=3)


def test_advance_rejects_a_negative_delta() -> None:
    clock = SimulatedClock(FIXED)
    with pytest.raises(ClockRewindError):
        clock.advance(timedelta(seconds=-1))
    assert clock.now() == FIXED


def test_set_to_forward_is_allowed() -> None:
    clock = SimulatedClock(FIXED)
    target = FIXED + timedelta(days=2)
    clock.set_to(target)
    assert clock.now() == target


def test_set_to_backwards_raises_clock_rewind_error() -> None:
    clock = SimulatedClock(FIXED)
    with pytest.raises(ClockRewindError) as excinfo:
        clock.set_to(FIXED - timedelta(seconds=1))
    assert clock.now() == FIXED
    assert "backwards" in str(excinfo.value)


def test_set_to_the_same_instant_is_allowed() -> None:
    clock = SimulatedClock(FIXED)
    clock.set_to(FIXED)
    assert clock.now() == FIXED


def test_ensure_utc_rejects_naive_datetimes() -> None:
    with pytest.raises(NaiveDatetimeError) as excinfo:
        ensure_utc(datetime(2020, 3, 19, 12, 0))  # noqa: DTZ001
    assert "naive" in str(excinfo.value)


def test_ensure_utc_converts_other_zones() -> None:
    ny = datetime(2020, 3, 19, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert ensure_utc(ny) == FIXED
    assert ensure_utc(ny).utcoffset() == timedelta(0)


def test_to_display_tz_defaults_to_taipei() -> None:
    shown = to_display_tz(FIXED)
    assert shown.hour == 20
    assert shown.utcoffset() == timedelta(hours=8)


def test_to_display_tz_accepts_another_zone() -> None:
    assert to_display_tz(FIXED, "America/New_York").hour == 8


# --- trading_day -----------------------------------------------------------
# The boundary is the local close, not UTC midnight. These cases are exactly the
# ones a naive `ts.date()` gets wrong.


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        # 21:00 Taipei on the 19th is 13:00 UTC — same UTC day, after the 13:30
        # close in local terms? No: 13:00 UTC is 21:00 Taipei, i.e. after close,
        # so it belongs to the next trading day.
        (datetime(2020, 3, 19, 13, 0, tzinfo=UTC), date(2020, 3, 20)),
        # 11:00 Taipei on the 19th (03:00 UTC) is mid-session: trading day 19th.
        (datetime(2020, 3, 19, 3, 0, tzinfo=UTC), date(2020, 3, 19)),
    ],
)
def test_trading_day_tw(ts: datetime, expected: date) -> None:
    assert trading_day(ts, Market.TW) == expected


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        # 15:00 New York on the 19th is 19:00 UTC: still the 19th session, and a
        # UTC-date split would already agree here.
        (datetime(2020, 3, 19, 19, 0, tzinfo=UTC), date(2020, 3, 19)),
        # 17:00 New York on the 19th is 21:00 UTC: after the 16:00 close, so the
        # 20th. A UTC-date split would wrongly say the 19th.
        (datetime(2020, 3, 19, 21, 0, tzinfo=UTC), date(2020, 3, 20)),
    ],
)
def test_trading_day_us(ts: datetime, expected: date) -> None:
    assert trading_day(ts, Market.US) == expected


def test_trading_day_crosses_the_utc_date_boundary_for_us() -> None:
    # 20:00 New York on the 19th is 00:00 UTC on the 20th. Slicing on the UTC
    # date would say the 20th... which is right here only by accident; the point
    # is that the answer comes from the local close, not from UTC.
    ts = datetime(2020, 3, 20, 0, 0, tzinfo=UTC)
    assert trading_day(ts, Market.US) == date(2020, 3, 20)
    # Meanwhile in Taipei that same instant is 08:00 on the 20th: pre-close.
    assert trading_day(ts, Market.TW) == date(2020, 3, 20)


def test_trading_day_rejects_naive_input() -> None:
    with pytest.raises(NaiveDatetimeError):
        trading_day(datetime(2020, 3, 19, 12, 0), Market.TW)  # noqa: DTZ001


def test_trading_day_exactly_at_the_close_stays_on_the_same_day() -> None:
    close = datetime(2020, 3, 19, 13, 30, tzinfo=ZoneInfo("Asia/Taipei"))
    assert trading_day(close, Market.TW) == date(2020, 3, 19)
