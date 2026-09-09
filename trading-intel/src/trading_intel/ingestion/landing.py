"""Raw landing zone：原始 payload 的唯讀存放區。

SPEC 3.2 與第 11 節 Phase 1 第 2 點要求原始 payload 永不覆寫，且路徑含抓取時間戳。

理由是「解析器會改，事實不會」。今天的解析器可能漏抓一個欄位、或誤判一個編碼；
半年後修好時，如果當初只存了解析結果，那段歷史就永遠是錯的。存下原始 bytes，
就能用新解析器重跑舊資料。這是整個 L0 唯一真正不可回復的決定。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from trading_intel.core.clock import ensure_utc, utc_now
from trading_intel.core.errors import DataQualityError
from trading_intel.core.ids import EvidenceId, make_evidence_id

#: 落地檔名中時間戳的格式。用底線與短横避免作業系統的路徑限制。
_TIMESTAMP_FORMAT: Final = "%Y%m%dT%H%M%S%fZ"

#: 中繼資料的副檔名。與 payload 同目錄同檔名，方便一起搬移。
_META_SUFFIX: Final = ".meta.json"


@dataclass(frozen=True)
class RawRecord:
    """一次抓取的完整紀錄：內容、來源、抓取時間、內容位址。"""

    evidence_id: EvidenceId
    source: str
    dataset: str
    #: 這批資料所描述的業務日期或鍵值，例如 "2024-03-19"。用於分區與查詢。
    partition: str
    fetch_time: datetime
    payload_path: Path
    content_sha256: str
    byte_size: int
    #: 抓取當下的請求細節（URL、參數），供日後重現與稽核。
    request: dict[str, Any]

    def read_payload(self) -> bytes:
        return self.payload_path.read_bytes()


def _partition_dir(root: Path, source: str, dataset: str, partition: str) -> Path:
    return root / source / dataset / partition


def _fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_raw(
    root: Path,
    *,
    source: str,
    dataset: str,
    partition: str,
    payload: bytes,
    request: dict[str, Any] | None = None,
    fetch_time: datetime | None = None,
    suffix: str = ".json",
) -> RawRecord:
    """把一次抓取的原始 payload 落地，回傳其紀錄。

    同一個 (source, dataset, partition) 可以有多筆紀錄——那正是重點：
    第二次抓取不覆寫第一次，兩者以 ``fetch_time`` 區分。這讓「資料在什麼時候
    變成什麼樣子」可以被查出來，也是 ``ingest_time`` 的來源。
    """
    if not payload:
        raise DataQualityError(
            "拒絕落地空的 payload",
            source=source,
            dataset=dataset,
            partition=partition,
        )
    stamped = ensure_utc(fetch_time) if fetch_time is not None else utc_now()
    target_dir = _partition_dir(root, source, dataset, partition)
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = stamped.strftime(_TIMESTAMP_FORMAT)
    payload_path = target_dir / f"{stem}{suffix}"
    if payload_path.exists():
        # 同一微秒內的第二次抓取。加上內容指紋以免覆寫。
        payload_path = target_dir / f"{stem}-{_fingerprint(payload)[:8]}{suffix}"
    if payload_path.exists():
        raise DataQualityError(
            "落地路徑已存在，拒絕覆寫原始 payload",
            path=str(payload_path),
        )

    record = RawRecord(
        evidence_id=make_evidence_id(f"{source}/{dataset}", payload),
        source=source,
        dataset=dataset,
        partition=partition,
        fetch_time=stamped,
        payload_path=payload_path,
        content_sha256=_fingerprint(payload),
        byte_size=len(payload),
        request=dict(request or {}),
    )

    payload_path.write_bytes(payload)
    _meta_path(payload_path).write_text(
        json.dumps(
            {
                "evidence_id": str(record.evidence_id),
                "source": record.source,
                "dataset": record.dataset,
                "partition": record.partition,
                "fetch_time": record.fetch_time.isoformat(),
                "content_sha256": record.content_sha256,
                "byte_size": record.byte_size,
                "request": record.request,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return record


def _meta_path(payload_path: Path) -> Path:
    return payload_path.with_suffix(payload_path.suffix + _META_SUFFIX)


def _load_record(meta_path: Path) -> RawRecord:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload_path = Path(str(meta_path).removesuffix(_META_SUFFIX))
    return RawRecord(
        evidence_id=EvidenceId(meta["evidence_id"]),
        source=meta["source"],
        dataset=meta["dataset"],
        partition=meta["partition"],
        fetch_time=datetime.fromisoformat(meta["fetch_time"]),
        payload_path=payload_path,
        content_sha256=meta["content_sha256"],
        byte_size=meta["byte_size"],
        request=meta.get("request", {}),
    )


def list_raw(
    root: Path,
    *,
    source: str,
    dataset: str,
    partition: str | None = None,
    asof: datetime | None = None,
) -> list[RawRecord]:
    """列出符合條件的原始紀錄，依抓取時間排序。

    ``asof`` 是本專案所有讀取路徑的通用參數（CLAUDE.md 第 2 條）：
    只回傳在該時點之前已經抓到的紀錄，讓回測看不到未來才落地的資料。
    """
    base = root / source / dataset
    if not base.exists():
        return []
    boundary = ensure_utc(asof) if asof is not None else None
    search_root = base / partition if partition is not None else base
    if not search_root.exists():
        return []

    records = [_load_record(meta) for meta in sorted(search_root.rglob(f"*{_META_SUFFIX}"))]
    if boundary is not None:
        records = [record for record in records if record.fetch_time <= boundary]
    return sorted(records, key=lambda record: (record.partition, record.fetch_time))


def latest_raw(
    root: Path,
    *,
    source: str,
    dataset: str,
    partition: str,
    asof: datetime | None = None,
) -> RawRecord | None:
    """取某個分區在 ``asof`` 之前最後一次抓到的紀錄。"""
    records = list_raw(root, source=source, dataset=dataset, partition=partition, asof=asof)
    return records[-1] if records else None
