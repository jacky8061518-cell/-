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
    assert "往回撥" in str(excinfo.value)


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
# 邊界是當地收盤時間而非 UTC 午夜。以下案例正是天真的 `ts.date()` 會算錯的那些。


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        # 13:00 UTC 是台北 21:00，已過 13:30 收盤，因此屬於次一個交易日。
        (datetime(2020, 3, 19, 13, 0, tzinfo=UTC), date(2020, 3, 20)),
        # 03:00 UTC 是台北 11:00，盤中，交易日為 19 日。
        (datetime(2020, 3, 19, 3, 0, tzinfo=UTC), date(2020, 3, 19)),
    ],
)
def test_trading_day_tw(ts: datetime, expected: date) -> None:
    assert trading_day(ts, Market.TW) == expected


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        # 19:00 UTC 是紐約 15:00，仍屬 19 日盤中；此例即使用 UTC 日期切也會一致。
        (datetime(2020, 3, 19, 19, 0, tzinfo=UTC), date(2020, 3, 19)),
        # 21:00 UTC 是紐約 17:00，已過 16:00 收盤，因此屬於 20 日。
        # 若以 UTC 日期切日會誤判為 19 日，這正是本函式存在的理由。
        (datetime(2020, 3, 19, 21, 0, tzinfo=UTC), date(2020, 3, 20)),
    ],
)
def test_trading_day_us(ts: datetime, expected: date) -> None:
    assert trading_day(ts, Market.US) == expected


def test_trading_day_crosses_the_utc_date_boundary_for_us() -> None:
    # 紐約 19 日 20:00 是 UTC 20 日 00:00。用 UTC 日期切也會得到 20 日，
    # 但那只是碰巧對；重點在於答案來自當地收盤時間，不是來自 UTC。
    ts = datetime(2020, 3, 20, 0, 0, tzinfo=UTC)
    assert trading_day(ts, Market.US) == date(2020, 3, 20)
    # 同一瞬間在台北是 20 日 08:00，尚未收盤。
    assert trading_day(ts, Market.TW) == date(2020, 3, 20)


def test_trading_day_rejects_naive_input() -> None:
    with pytest.raises(NaiveDatetimeError):
        trading_day(datetime(2020, 3, 19, 12, 0), Market.TW)  # noqa: DTZ001


def test_trading_day_exactly_at_the_close_stays_on_the_same_day() -> None:
    close = datetime(2020, 3, 19, 13, 30, tzinfo=ZoneInfo("Asia/Taipei"))
    assert trading_day(close, Market.TW) == date(2020, 3, 19)
