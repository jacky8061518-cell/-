"""沙箱必須真的切斷網路，而且離開後必須完整還原。"""

from __future__ import annotations

import socket
import ssl
from datetime import datetime

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.errors import NetworkAccessDenied
from trading_intel.core.sandbox import backtest_mode, no_network


def test_create_connection_is_blocked() -> None:
    """對應 CLAUDE.md 第 3 條：回測期間禁止任何網路呼叫。"""
    with no_network(), pytest.raises(NetworkAccessDenied) as excinfo:
        socket.create_connection(("example.com", 80), timeout=0.1)
    assert "create_connection" in str(excinfo.value)
    assert excinfo.value.context["target"].endswith("create_connection")


def test_socket_constructor_is_blocked() -> None:
    with no_network(), pytest.raises(NetworkAccessDenied):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_originals_are_restored_on_exit() -> None:
    original_socket = socket.socket
    original_create = socket.create_connection
    original_wrap = getattr(ssl, "wrap_socket", None)

    with no_network():
        assert socket.socket is not original_socket

    assert socket.socket is original_socket
    assert socket.create_connection is original_create
    assert getattr(ssl, "wrap_socket", None) is original_wrap
    # 而且可以再次建立真正的 socket。
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()


def test_originals_are_restored_even_when_the_block_raises() -> None:
    original_socket = socket.socket
    with pytest.raises(RuntimeError), no_network():
        raise RuntimeError("boom")
    assert socket.socket is original_socket


def test_nested_use_does_not_restore_early() -> None:
    original_socket = socket.socket

    with no_network():
        with no_network():  # noqa: SIM117  (the nesting is the thing under test)
            with pytest.raises(NetworkAccessDenied):
                socket.socket()
        # 內層區塊已離開，外層必須仍在封鎖狀態。
        with pytest.raises(NetworkAccessDenied):
            socket.socket()

    assert socket.socket is original_socket


def test_backtest_mode_blocks_the_network_and_freezes_the_clock() -> None:
    from trading_intel.core.clock import utc_now

    asof = datetime(2020, 3, 19, 13, 30, tzinfo=UTC)
    with backtest_mode(asof) as clock:
        assert utc_now() == asof
        assert clock.now() == asof
        with pytest.raises(NetworkAccessDenied) as excinfo:
            socket.create_connection(("example.com", 443), timeout=0.1)
    assert "沙箱內禁止" in str(excinfo.value)
