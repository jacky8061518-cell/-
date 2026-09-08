"""全專案唯一允許讀取現在時間的模組。

其他所有程式碼一律呼叫 :func:`utc_now`。作用中的時鐘存放在
:class:`~contextvars.ContextVar` 而非模組全域變數，如此 async 任務與平行測試
各自看到自己的時鐘，不會互相污染。

``ruff`` 在全專案禁用 ``datetime.now`` / ``datetime.utcnow`` / ``datetime.today``
與 ``time.time``，只豁免本檔；``tests/guard`` 另以 AST 走訪執行同一條規則。

對應 CLAUDE.md 第 4 條：系統內部時間一律 UTC，只在展示層轉換為台北時間。
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

#: ingest_time 允許早於 event_time 的最大幅度。超過此值視為管線有 bug，
#: 而不是機器之間的時鐘漂移。
MAX_CLOCK_SKEW: Final = timedelta(seconds=5)

#: 各市場的當地收盤時間。交易日邊界是收盤而非 UTC 午夜：美股 16:00 紐約時間
#: 換算為 20:00 或 21:00 UTC，若以 UTC 日期切日，整個下午盤會被推到隔天。
MARKET_CLOSE: Final[dict[Market, tuple[str, time]]] = {
    Market.TW: ("Asia/Taipei", time(13, 30)),
    Market.US: ("America/New_York", time(16, 0)),
}


@runtime_checkable
class Clock(Protocol):
    """任何能以 tz-aware UTC 回報時間的物件。"""

    def now(self) -> datetime: ...


@dataclass(frozen=True)
class SystemClock:
    """牆上時鐘。全專案唯一真正呼叫系統時間的地方。"""

    def now(self) -> datetime:
        return _dt.datetime.now(tz=UTC)


@dataclass
class SimulatedClock:
    """受測試或回測控制的時鐘，結構上保證單調不倒退。"""

    current: datetime

    def __post_init__(self) -> None:
        self.current = ensure_utc(self.current)

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ClockRewindError(
                "模擬時鐘不得以負值前進",
                current=self.current.isoformat(),
                delta=str(delta),
            )
        self.current = self.current + delta

    def set_to(self, ts: datetime) -> None:
        target = ensure_utc(ts)
        if target < self.current:
            raise ClockRewindError(
                "模擬時鐘不得往回撥",
                current=self.current.isoformat(),
                target=target.isoformat(),
            )
        self.current = target


# 預設值是 frozen 且無狀態的 dataclass，因此共用單一實例是安全的。
_DEFAULT_CLOCK: Final[Clock] = SystemClock()
_active_clock: ContextVar[Clock] = ContextVar(
    "_active_clock",
    default=_DEFAULT_CLOCK,
)


def get_clock() -> Clock:
    """取得目前 context 生效中的時鐘。"""
    return _active_clock.get()


@contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """在區塊期間安裝 ``clock``，離開時還原前一個時鐘。"""
    token = _active_clock.set(clock)
    try:
        yield clock
    finally:
        _active_clock.reset(token)


def utc_now() -> datetime:
    """全專案取得現在時間的唯一入口。"""
    return ensure_utc(get_clock().now())


def ensure_utc(ts: datetime) -> datetime:
    """把 aware datetime 正規化為 UTC；naive datetime 直接拒收。

    刻意不假設 naive 值就是 UTC：猜錯就是一個差 8 小時、而且不會報錯的 bug。
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise NaiveDatetimeError(
            "不接受 naive datetime，請附上時區",
            value=ts.isoformat(),
        )
    return ts.astimezone(UTC)


def to_display_tz(ts: datetime, tz: str = "Asia/Taipei") -> datetime:
    """轉換為人類可讀的時區。僅供 L5 展示層使用。"""
    return ensure_utc(ts).astimezone(ZoneInfo(tz))


def trading_day(ts: datetime, market: Market) -> date:
    """回傳 ``ts`` 在 ``market`` 所屬的交易日。

    邊界為當地收盤時間：收盤當下與之前屬於當地日期，之後屬於次一個日曆日。
    假日行事曆不在 Phase 0 範圍內，因此「次日」是次一個日曆日，
    不保證是次一個真正的交易日。
    """
    tz_name, close = MARKET_CLOSE[market]
    local = ensure_utc(ts).astimezone(ZoneInfo(tz_name))
    day = local.date()
    if local.timetz().replace(tzinfo=None) > close:
        day = day + timedelta(days=1)
    return day
