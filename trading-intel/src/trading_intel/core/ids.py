"""決定性識別碼。

每個 id 都是輸入的純函數。若 id 帶隨機性，同一份事件流重放兩次就會產生不同的
訊號 id，雙時間戳設計換來的可重現性也就沒了（SPEC 1）。因此 ``uuid4`` 只允許
用於 correlation id：那是純粹的追蹤中繼資料，不會進入訊號內容。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import NewType

from trading_intel.core.clock import ensure_utc
from trading_intel.core.enums import Market

EntityId = NewType("EntityId", str)
EvidenceId = NewType("EvidenceId", str)
SignalId = NewType("SignalId", str)
CorrelationId = NewType("CorrelationId", str)

_HASH_WIDTH = 16
_SEP = b"\x1f"  # ASCII unit separator：不會出現在被串接的欄位內容中


def _digest(*parts: bytes, width: int = _HASH_WIDTH) -> str:
    return hashlib.sha256(_SEP.join(parts)).hexdigest()[:width]


def make_entity_id(market: Market, local_symbol: str) -> EntityId:
    """格式為 ``TW:2330`` 或 ``US:NVDA``，代號先做 strip 與 upper。"""
    symbol = local_symbol.strip().upper()
    if not symbol:
        msg = "local_symbol 不得為空白"
        raise ValueError(msg)
    return EntityId(f"{market.value}:{symbol}")


def make_evidence_id(source: str, payload: bytes) -> EvidenceId:
    """文件的內容位址，同一份文件重複收到會收斂到同一個 id。"""
    return EvidenceId(_digest(source.encode("utf-8"), payload))


def make_signal_id(entity_id: EntityId, model_version: str, asof: datetime) -> SignalId:
    """決定性雜湊：同一組（標的、模型版本、asof）永遠對應同一個 id。"""
    return SignalId(
        _digest(
            str(entity_id).encode("utf-8"),
            model_version.encode("utf-8"),
            ensure_utc(asof).isoformat().encode("utf-8"),
        )
    )


def new_correlation_id() -> CorrelationId:
    """刻意採用隨機值：純追蹤用途，不構成訊號身分的一部分。"""
    return CorrelationId(uuid.uuid4().hex)
