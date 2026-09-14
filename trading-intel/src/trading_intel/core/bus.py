"""事件匯流排：at-least-once 投遞、consumer group、可重放。

SPEC 第 2 節選定 Redis Streams。這裡把它抽象成 ``EventBus`` 協定，
並提供兩個實作：

- ``RedisStreamBus`` 正式使用，consumer group 由 Redis 提供；
- ``InMemoryBus`` 供測試與離線開發，語意刻意做成與 Redis Streams 一致。

抽象出協定不是為了「以後也許會換掉 Redis」這種空泛理由，而是因為
ADR 0001 已經決定信封契約必須與傳輸層無關：重放的正確性應該由事件本身的
``idempotency_key`` 保證，而不是由某個 broker 的特性保證。有兩個實作，
這件事就會被測試持續檢查。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from trading_intel.core.events import EventEnvelope


@dataclass(frozen=True)
class DeliveredEvent:
    """一則從匯流排取出的事件，以及確認它所需的 id。"""

    #: broker 指派的訊息 id。確認（ack）時要回傳它。
    message_id: str
    stream: str
    #: 信封的 JSON 原文。呼叫端自行以正確的 payload 型別反序列化。
    raw: str

    def parse(self, payload_type: type[BaseModel]) -> EventEnvelope[Any]:
        return EventEnvelope[payload_type].model_validate_json(self.raw)  # type: ignore[valid-type]


@runtime_checkable
class EventBus(Protocol):
    """匯流排的最小介面。"""

    def publish(self, stream: str, envelope: EventEnvelope[Any]) -> str: ...

    def ensure_group(self, stream: str, group: str) -> None: ...

    def read(
        self,
        stream: str,
        *,
        group: str,
        consumer: str,
        count: int = 100,
    ) -> list[DeliveredEvent]: ...

    def ack(self, stream: str, group: str, message_ids: Sequence[str]) -> int: ...

    def replay(self, stream: str) -> Iterator[DeliveredEvent]: ...


@dataclass
class InMemoryBus:
    """行程內的 Redis Streams 模擬，語意刻意對齊。

    重放（``replay``）不受 consumer group 的消費進度影響——這是刻意的：
    重放的用途是「用新程式碼重跑舊事件」，而不是「處理沒消化完的積壓」。
    """

    _streams: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    _groups: dict[tuple[str, str], int] = field(default_factory=dict)
    _pending: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    _sequence: int = 0

    def publish(self, stream: str, envelope: EventEnvelope[Any]) -> str:
        self._sequence += 1
        message_id = f"{self._sequence}-0"
        self._streams.setdefault(stream, []).append((message_id, envelope.model_dump_json()))
        return message_id

    def ensure_group(self, stream: str, group: str) -> None:
        self._streams.setdefault(stream, [])
        self._groups.setdefault((stream, group), 0)
        self._pending.setdefault((stream, group), set())

    def read(
        self,
        stream: str,
        *,
        group: str,
        consumer: str,  # noqa: ARG002  (單行程模擬不需要區分 consumer)
        count: int = 100,
    ) -> list[DeliveredEvent]:
        self.ensure_group(stream, group)
        key = (stream, group)
        offset = self._groups[key]
        entries = self._streams[stream][offset : offset + count]
        self._groups[key] = offset + len(entries)
        self._pending[key].update(message_id for message_id, _ in entries)
        return [
            DeliveredEvent(message_id=message_id, stream=stream, raw=raw)
            for message_id, raw in entries
        ]

    def ack(self, stream: str, group: str, message_ids: Sequence[str]) -> int:
        key = (stream, group)
        pending = self._pending.setdefault(key, set())
        acknowledged = 0
        for message_id in message_ids:
            if message_id in pending:
                pending.discard(message_id)
                acknowledged += 1
        return acknowledged

    def pending_count(self, stream: str, group: str) -> int:
        return len(self._pending.get((stream, group), set()))

    def replay(self, stream: str) -> Iterator[DeliveredEvent]:
        """從頭重放整條串流，不動任何 consumer group 的進度。"""
        for message_id, raw in self._streams.get(stream, []):
            yield DeliveredEvent(message_id=message_id, stream=stream, raw=raw)


class RedisStreamBus:
    """以 Redis Streams 為底的實作。

    ``XADD`` / ``XREADGROUP`` / ``XACK`` 提供 at-least-once 與消費進度追蹤；
    ``XRANGE`` 提供不影響消費進度的重放。
    """

    #: 信封 JSON 存放在這個欄位下。單欄位即可，因為信封本身已是完整訊息。
    FIELD = "envelope"

    def __init__(self, client: Any) -> None:
        self._client = client

    def publish(self, stream: str, envelope: EventEnvelope[Any]) -> str:
        message_id = self._client.xadd(stream, {self.FIELD: envelope.model_dump_json()})
        return str(_as_text(message_id))

    def ensure_group(self, stream: str, group: str) -> None:
        try:
            self._client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as exc:
            # Redis 對「群組已存在」回 BUSYGROUP。這不是錯誤，是冪等。
            if "BUSYGROUP" not in str(exc):
                raise

    def read(
        self,
        stream: str,
        *,
        group: str,
        consumer: str,
        count: int = 100,
    ) -> list[DeliveredEvent]:
        self.ensure_group(stream, group)
        response = self._client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
        )
        delivered: list[DeliveredEvent] = []
        for _stream_name, entries in response or []:
            for message_id, fields in entries:
                delivered.append(
                    DeliveredEvent(
                        message_id=str(_as_text(message_id)),
                        stream=stream,
                        raw=_field_value(fields, self.FIELD),
                    )
                )
        return delivered

    def ack(self, stream: str, group: str, message_ids: Sequence[str]) -> int:
        if not message_ids:
            return 0
        return int(self._client.xack(stream, group, *message_ids))

    def pending_count(self, stream: str, group: str) -> int:
        summary = self._client.xpending(stream, group)
        if isinstance(summary, dict):
            return int(summary.get("pending", 0))
        return int(summary[0]) if summary else 0

    def replay(self, stream: str) -> Iterator[DeliveredEvent]:
        for message_id, fields in self._client.xrange(stream, min="-", max="+"):
            yield DeliveredEvent(
                message_id=str(_as_text(message_id)),
                stream=stream,
                raw=_field_value(fields, self.FIELD),
            )


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _field_value(fields: Any, key: str) -> str:
    for raw_key, raw_value in fields.items():
        if _as_text(raw_key) == key:
            return _as_text(raw_value)
    msg = f"訊息缺少 {key} 欄位"
    raise KeyError(msg)
