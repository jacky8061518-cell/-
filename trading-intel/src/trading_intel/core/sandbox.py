"""Guard rails that make a whole class of bugs impossible rather than unlikely.

A backtest that can reach the network can leak the future into the past. So
instead of asking every data adapter to behave, we cut the socket layer out from
under them for the duration of the run.
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
            f"network access via {target} is disabled inside this sandbox",
            target=target,
            call=f"args={args!r} kwargs={kwargs!r}",
        )

    return _raise


@contextmanager
def no_network() -> Iterator[None]:
    """Block outbound sockets for the duration of the block.

    Nesting is safe: each level saves the attributes it found on entry and puts
    exactly those back on exit, so an inner block cannot restore networking for
    an outer one.
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
    """Freeze the clock at ``asof`` and cut the network.

    The Phase 2 backtest engine runs inside this and nowhere else, so "did we
    remember to sandbox this run" stops being a question anyone has to ask.
    """
    clock = SimulatedClock(asof)
    with no_network(), use_clock(clock):
        yield clock
