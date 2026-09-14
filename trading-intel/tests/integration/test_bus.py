"""事件匯流排：consumer group、at-least-once、可重放。

同一組測試同時跑 ``InMemoryBus`` 與 ``RedisStreamBus``，強制兩者語意一致。
Redis 不在時該組自動跳過，因此離線開發不會卡住。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from trading_intel.core.bus import DeliveredEvent, EventBus, InMemoryBus, RedisStreamBus
from trading_intel.core.clock import UTC, SimulatedClock, use_clock
from trading_intel.core.enums import Market
from trading_intel.core.events import wrap
from trading_intel.core.ids import make_entity_id
from trading_intel.core.types import Bar

REDIS_PORT = int(os.environ.get("TI_TEST_REDIS_PORT", "6399"))
EVENT_TIME = datetime(2024, 3, 19, 12, 0, tzinfo=UTC)
ENTITY = make_entity_id(Market.TW, "2330")


def make_bar(close: str = "105") -> Bar:
    return Bar(
        event_time=EVENT_TIME,
        ingest_time=EVENT_TIME + timedelta(minutes=1),
        entity_id=ENTITY,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal(close),
        volume=1000,
    )


def _redis_client() -> Any:
    redis = pytest.importorskip("redis")
    client = redis.Redis(port=REDIS_PORT, decode_responses=True, socket_connect_timeout=1)
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - 取決於環境
        pytest.skip(f"Redis 未就緒（port {REDIS_PORT}）：{exc}")
    return client


@pytest.fixture(params=["memory", "redis"])
def bus(request: pytest.FixtureRequest) -> Iterator[EventBus]:
    if request.param == "memory":
        yield InMemoryBus()
        return
    client = _redis_client()
    # node.name 對參數化測試含中括號，而 Redis KEYS 的 glob 會把 [..] 當字元集，
    # 導致清理靜默失效、測試之間互相污染。改用不含特殊字元的鍵名。
    prefix = _safe_key(request.node.name)
    _purge(client, prefix)
    yield RedisStreamBus(client)
    _purge(client, prefix)


def _safe_key(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name)


def _purge(client: Any, prefix: str) -> None:
    for key in client.scan_iter(match=f"{prefix}*", count=1000):
        client.delete(key)


@pytest.fixture
def stream(request: pytest.FixtureRequest) -> str:
    return f"{_safe_key(request.node.name)}_bars"


def test_publish_then_read(bus: EventBus, stream: str) -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        envelope = wrap(make_bar(), event_type="market.bar")
    bus.publish(stream, envelope)
    delivered = bus.read(stream, group="g1", consumer="c1")
    assert len(delivered) == 1
    restored = delivered[0].parse(Bar)
    assert restored.payload == make_bar()
    assert restored.idempotency_key == envelope.idempotency_key


def test_consumer_group_does_not_redeliver_unacked_reads(bus: EventBus, stream: str) -> None:
    """XREADGROUP 的 ``>`` 只給未曾投遞過的訊息。"""
    with use_clock(SimulatedClock(EVENT_TIME)):
        for close in ("101", "102", "103"):
            bus.publish(stream, wrap(make_bar(close), event_type="market.bar"))
    first = bus.read(stream, group="g1", consumer="c1", count=2)
    second = bus.read(stream, group="g1", consumer="c1", count=10)
    assert len(first) == 2
    assert len(second) == 1
    assert {item.message_id for item in first}.isdisjoint({item.message_id for item in second})


def test_two_groups_each_see_everything(bus: EventBus, stream: str) -> None:
    """不同 consumer group 互不影響——這是 fan-out 的基礎。"""
    with use_clock(SimulatedClock(EVENT_TIME)):
        for close in ("101", "102"):
            bus.publish(stream, wrap(make_bar(close), event_type="market.bar"))
    assert len(bus.read(stream, group="features", consumer="c1")) == 2
    assert len(bus.read(stream, group="quality", consumer="c1")) == 2


def test_ack_reduces_pending(bus: EventBus, stream: str) -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        bus.publish(stream, wrap(make_bar(), event_type="market.bar"))
    delivered = bus.read(stream, group="g1", consumer="c1")
    assert bus.ack(stream, "g1", [item.message_id for item in delivered]) == 1
    assert bus.ack(stream, "g1", []) == 0


def test_replay_is_independent_of_consumption(bus: EventBus, stream: str) -> None:
    """重放不受消費進度影響：它的用途是用新程式碼重跑舊事件。"""
    with use_clock(SimulatedClock(EVENT_TIME)):
        for close in ("101", "102", "103"):
            bus.publish(stream, wrap(make_bar(close), event_type="market.bar"))
    bus.read(stream, group="g1", consumer="c1", count=10)
    assert len(list(bus.replay(stream))) == 3


def test_replaying_twice_yields_identical_results(bus: EventBus, stream: str) -> None:
    """SPEC 第 11 節 Phase 1 驗收：事件流重放兩次產生相同結果。"""
    with use_clock(SimulatedClock(EVENT_TIME)):
        for close in ("101", "102", "103"):
            bus.publish(stream, wrap(make_bar(close), event_type="market.bar"))

    def digest(events: list[DeliveredEvent]) -> list[tuple[str, str, str]]:
        parsed = [item.parse(Bar) for item in events]
        return [(item.event_id, item.idempotency_key, str(item.payload.close)) for item in parsed]

    first = digest(list(bus.replay(stream)))
    second = digest(list(bus.replay(stream)))
    assert first == second
    assert len(first) == 3
    # 內容相同的事件其 idempotency_key 也必須相同，重放才能去重。
    assert len({key for _, key, _ in first}) == 3


def test_identical_payloads_share_an_idempotency_key(bus: EventBus, stream: str) -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        first = wrap(make_bar(), event_type="market.bar")
        second = wrap(make_bar(), event_type="market.bar")
    bus.publish(stream, first)
    bus.publish(stream, second)
    replayed = [item.parse(Bar) for item in bus.replay(stream)]
    assert replayed[0].idempotency_key == replayed[1].idempotency_key
    # 去重之後只剩一個事實。
    assert len({item.idempotency_key for item in replayed}) == 1


def test_replay_of_an_unknown_stream_is_empty(bus: EventBus) -> None:
    assert list(bus.replay("ti_test_absent_stream")) == []


def test_ensure_group_is_idempotent(bus: EventBus, stream: str) -> None:
    bus.ensure_group(stream, "g1")
    bus.ensure_group(stream, "g1")
    assert bus.read(stream, group="g1", consumer="c1") == []


def test_produced_at_uses_the_active_clock(bus: EventBus, stream: str) -> None:
    with use_clock(SimulatedClock(EVENT_TIME)):
        bus.publish(stream, wrap(make_bar(), event_type="market.bar"))
    replayed = next(iter(bus.replay(stream))).parse(Bar)
    assert replayed.produced_at == EVENT_TIME
