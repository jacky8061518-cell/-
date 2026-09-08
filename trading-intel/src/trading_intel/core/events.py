"""事件信封。

匯流排上的每一則訊息都經過包裝，讓重放、去重與追蹤三件事得以運作，
而 payload 模型本身完全不需要知道傳輸層的存在。
"""

from __future__ import annotations

import hashlib
import json

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from trading_intel.core.clock import utc_now
from trading_intel.core.ids import CorrelationId, new_correlation_id

_EVENT_ID_WIDTH = 32


class EventEnvelope[T: BaseModel](BaseModel):
    """包在 payload 外層的傳輸中繼資料。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    event_type: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    produced_at: AwareDatetime
    correlation_id: CorrelationId
    #: payload 的內容雜湊：同一個事實被重送時，下游可據此視為無動作。
    idempotency_key: str
    payload: T


def payload_hash(payload: BaseModel) -> str:
    """對正規化 JSON 取 sha256，因此欄位順序不會影響結果。"""
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
    """把 ``payload`` 裝進信封，其 idempotency key 由內容推導而來。"""
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
