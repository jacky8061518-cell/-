"""Deterministic identifiers.

Every id is a pure function of its inputs. If ids were random, replaying the
same event stream would produce different signal ids and reproducibility — the
whole point of the bitemporal design — would be gone. ``uuid4`` is therefore
allowed only for correlation ids, which are pure tracing metadata and never
enter signal content.
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
_SEP = b"\x1f"  # ASCII unit separator: cannot appear in the fields we join


def _digest(*parts: bytes, width: int = _HASH_WIDTH) -> str:
    return hashlib.sha256(_SEP.join(parts)).hexdigest()[:width]


def make_entity_id(market: Market, local_symbol: str) -> EntityId:
    """``TW:2330`` / ``US:NVDA``. The symbol is stripped and upper-cased first."""
    symbol = local_symbol.strip().upper()
    if not symbol:
        msg = "local_symbol must not be blank"
        raise ValueError(msg)
    return EntityId(f"{market.value}:{symbol}")


def make_evidence_id(source: str, payload: bytes) -> EvidenceId:
    """Content address for a document, so re-ingesting it collapses to one id."""
    return EvidenceId(_digest(source.encode("utf-8"), payload))


def make_signal_id(entity_id: EntityId, model_version: str, asof: datetime) -> SignalId:
    """Deterministic: the same (entity, model version, asof) always maps to one id."""
    return SignalId(
        _digest(
            str(entity_id).encode("utf-8"),
            model_version.encode("utf-8"),
            ensure_utc(asof).isoformat().encode("utf-8"),
        )
    )


def new_correlation_id() -> CorrelationId:
    """Random by design — tracing only, never part of a signal's identity."""
    return CorrelationId(uuid.uuid4().hex)
