"""Event envelopes.

Every message on the bus is wrapped so that replay, deduplication, and tracing
work without the payload models knowing anything about transport.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from trading_intel.core.clock import utc_now
from trading_intel.core.ids import CorrelationId, new_correlation_id

_EVENT_ID_WIDTH = 32


class EventEnvelope[T: BaseModel](BaseModel):
    """Transport metadata around a payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    event_type: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    produced_at: AwareDatetime
    correlation_id: CorrelationId
    #: Content hash of the payload: replaying the same fact is a no-op downstream.
    idempotency_key: str
    payload: T


def payload_hash(payload: BaseModel) -> str:
    """sha256 over canonical JSON, so field ordering cannot change the result."""
    data = payload.model_dump(mode="json", by_alias=True)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def wrap[T: BaseModel](
    payload: T,
    *,
    event_type: str,
    schema_version: int = 1,
    correlation_id: CorrelationId | None = None,
) -> EventEnvelope[T]:
    """Put ``payload`` in an envelope, deriving its idempotency key from content."""
    key = payload_hash(payload)
    return EventEnvelope[T](
        event_id=hashlib.sha256(f"{event_type}\x1f{schema_version}\x1f{key}".encode()).hexdigest()[
            :_EVENT_ID_WIDTH
        ],
        event_type=event_type,
        schema_version=schema_version,
        produced_at=utc_now(),
        correlation_id=correlation_id if correlation_id is not None else new_correlation_id(),
        idempotency_key=key,
        payload=payload,
    )
