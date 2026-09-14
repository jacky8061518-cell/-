"""Raw landing zone：原始 payload 永不覆寫。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.errors import DataQualityError
from trading_intel.ingestion.landing import latest_raw, list_raw, write_raw

T0 = datetime(2024, 3, 19, 8, 0, tzinfo=UTC)


def test_write_returns_a_record_with_content_address(tmp_path: Path) -> None:
    record = write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="2024-03-19",
        payload=b'{"stat":"OK"}',
        fetch_time=T0,
    )
    assert record.byte_size == 13
    assert record.payload_path.exists()
    assert record.read_payload() == b'{"stat":"OK"}'
    assert len(record.content_sha256) == 64


def test_second_fetch_does_not_overwrite_the_first(tmp_path: Path) -> None:
    """同一分區抓兩次，兩份 payload 都要留著——這是 SPEC 3.2 的核心要求。"""
    first = write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="2024-03-19",
        payload=b'{"v":1}',
        fetch_time=T0,
    )
    second = write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="2024-03-19",
        payload=b'{"v":2}',
        fetch_time=T0 + timedelta(hours=1),
    )
    assert first.payload_path != second.payload_path
    assert first.read_payload() == b'{"v":1}'
    assert second.read_payload() == b'{"v":2}'
    assert len(list_raw(tmp_path, source="twse", dataset="t86")) == 2


def test_same_microsecond_fetches_both_survive(tmp_path: Path) -> None:
    kwargs = {"source": "twse", "dataset": "t86", "partition": "2024-03-19", "fetch_time": T0}
    first = write_raw(tmp_path, payload=b'{"v":1}', **kwargs)  # type: ignore[arg-type]
    second = write_raw(tmp_path, payload=b'{"v":2}', **kwargs)  # type: ignore[arg-type]
    assert first.payload_path != second.payload_path
    assert first.read_payload() != second.read_payload()


def test_empty_payload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataQualityError, match="空的 payload"):
        write_raw(tmp_path, source="twse", dataset="t86", partition="2024-03-19", payload=b"")


def test_identical_payload_yields_identical_evidence_id(tmp_path: Path) -> None:
    """同一份文件重複收到會收斂到同一個 evidence id。"""
    first = write_raw(
        tmp_path, source="twse", dataset="t86", partition="d1", payload=b"same", fetch_time=T0
    )
    second = write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="d2",
        payload=b"same",
        fetch_time=T0 + timedelta(days=1),
    )
    assert first.evidence_id == second.evidence_id


def test_asof_hides_later_fetches(tmp_path: Path) -> None:
    """CLAUDE.md 第 2 條：讀取路徑必須以 asof 過濾。"""
    write_raw(
        tmp_path, source="twse", dataset="t86", partition="p", payload=b"early", fetch_time=T0
    )
    write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="p",
        payload=b"late",
        fetch_time=T0 + timedelta(days=5),
    )
    visible = list_raw(tmp_path, source="twse", dataset="t86", asof=T0 + timedelta(days=1))
    assert len(visible) == 1
    assert visible[0].read_payload() == b"early"
    assert len(list_raw(tmp_path, source="twse", dataset="t86")) == 2


def test_latest_raw_respects_asof(tmp_path: Path) -> None:
    for offset, payload in enumerate([b"v1", b"v2", b"v3"]):
        write_raw(
            tmp_path,
            source="twse",
            dataset="t86",
            partition="p",
            payload=payload,
            fetch_time=T0 + timedelta(days=offset),
        )
    latest = latest_raw(
        tmp_path, source="twse", dataset="t86", partition="p", asof=T0 + timedelta(days=1)
    )
    assert latest is not None
    assert latest.read_payload() == b"v2"


def test_missing_dataset_returns_empty(tmp_path: Path) -> None:
    assert list_raw(tmp_path, source="nope", dataset="none") == []
    assert latest_raw(tmp_path, source="nope", dataset="none", partition="p") is None


def test_request_metadata_is_persisted(tmp_path: Path) -> None:
    write_raw(
        tmp_path,
        source="twse",
        dataset="t86",
        partition="p",
        payload=b"x",
        request={"url": "https://example.invalid", "params": {"date": "20240319"}},
        fetch_time=T0,
    )
    restored = list_raw(tmp_path, source="twse", dataset="t86")[0]
    assert restored.request["url"] == "https://example.invalid"
    assert restored.fetch_time == T0
