"""把一整類 bug 變成不可能發生，而不只是不太可能發生的防護。

能連上網路的回測，就能把未來洩漏進過去。與其要求每一個資料轉接器自律，
不如在回測期間直接把 socket 層抽掉。對應 CLAUDE.md 第 3 條。
"""

from __future__ import annotations

import socket
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from trading_intel.core.clock import SimulatedClock, use_clock
from trading_intel.core.errors import NetworkAccessDenied

_PATCH_TARGETS: tuple[tuple[object, str], ...] = (
    (socket, "socket"),
    (socket, "create_connection"),
    (socket, "socketpair"),
    (ssl, "wrap_socket"),
)


def _denied(target: str) -> Any:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessDenied(
            f"沙箱內禁止透過 {target} 存取網路",
            target=target,
            call=f"args={args!r} kwargs={kwargs!r}",
        )

    return _raise


@contextmanager
def no_network() -> Iterator[None]:
    """在區塊期間阻斷對外連線。

    巢狀使用是安全的：每一層各自記下進入時看到的屬性，離開時只還原自己那一份，
    因此內層區塊不會替外層提前解除封鎖。
    """
    saved: list[tuple[object, str, Any]] = []
    for module, attr in _PATCH_TARGETS:
        if not hasattr(module, attr):
            continue
        saved.append((module, attr, getattr(module, attr)))
        qualified = f"{getattr(module, '__name__', module)}.{attr}"
        setattr(module, attr, _denied(qualified))
    try:
        yield
    finally:
        for module, attr, original in reversed(saved):
            setattr(module, attr, original)


@contextmanager
def backtest_mode(asof: datetime) -> Iterator[SimulatedClock]:
    """把時鐘凍結在 ``asof`` 並切斷網路。

    Phase 2 的回測引擎一律在這個 context 內執行，如此「這次有沒有記得包沙箱」
    就不再是任何人需要記住的事。
    """
    clock = SimulatedClock(asof)
    with no_network(), use_clock(clock):
        yield clock
