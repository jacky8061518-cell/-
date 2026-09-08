"""id 必須是輸入的純函數。

以下期望值刻意硬編碼而非重算。若未來雜湊方式被無聲改動，
所有已存放的訊號 id 都會對不起來——所以這裡故意設計成會壞掉。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.core.errors import NaiveDatetimeError
from trading_intel.core.ids import (
    make_entity_id,
    make_evidence_id,
    make_signal_id,
    new_correlation_id,
)

ASOF = datetime(2020, 3, 19, 12, 0, tzinfo=UTC)

EXPECTED_ENTITY_ID = "TW:2330"
EXPECTED_EVIDENCE_ID = "0361101ba4934e81"
EXPECTED_SIGNAL_ID = "c7e518d87a27b29f"


def test_entity_id_format() -> None:
    assert make_entity_id(Market.TW, "2330") == EXPECTED_ENTITY_ID
    assert make_entity_id(Market.US, "NVDA") == "US:NVDA"


def test_entity_id_strips_and_upcases() -> None:
    assert make_entity_id(Market.TW, "  2330  ") == EXPECTED_ENTITY_ID
    assert make_entity_id(Market.US, " nvda ") == "US:NVDA"


def test_entity_id_rejects_a_blank_symbol() -> None:
    with pytest.raises(ValueError, match="不得為空白"):
        make_entity_id(Market.TW, "   ")


def test_evidence_id_is_content_addressed() -> None:
    first = make_evidence_id("twse", b"hello world")
    second = make_evidence_id("twse", b"hello world")
    assert first == second == EXPECTED_EVIDENCE_ID
    assert len(first) == 16


def test_evidence_id_separates_source_from_payload() -> None:
    # 若沒有分隔字元，("ab", b"c") 與 ("a", b"bc") 會碰撞。
    assert make_evidence_id("ab", b"c") != make_evidence_id("a", b"bc")


def test_evidence_id_changes_with_content() -> None:
    assert make_evidence_id("twse", b"hello world") != make_evidence_id("twse", b"hello world!")


def test_signal_id_is_deterministic() -> None:
    entity = make_entity_id(Market.TW, "2330")
    assert make_signal_id(entity, "v1.2.3", ASOF) == EXPECTED_SIGNAL_ID
    assert make_signal_id(entity, "v1.2.3", ASOF) == make_signal_id(entity, "v1.2.3", ASOF)


def test_signal_id_varies_with_every_input() -> None:
    entity = make_entity_id(Market.TW, "2330")
    other = make_entity_id(Market.US, "NVDA")
    assert make_signal_id(other, "v1.2.3", ASOF) != EXPECTED_SIGNAL_ID
    assert make_signal_id(entity, "v1.2.4", ASOF) != EXPECTED_SIGNAL_ID
    assert make_signal_id(entity, "v1.2.3", ASOF + timedelta(seconds=1)) != EXPECTED_SIGNAL_ID


def test_signal_id_is_timezone_normalised() -> None:
    entity = make_entity_id(Market.TW, "2330")
    same_instant = ASOF.astimezone(ZoneInfo("Asia/Taipei"))
    assert make_signal_id(entity, "v1.2.3", same_instant) == EXPECTED_SIGNAL_ID


def test_signal_id_rejects_naive_asof() -> None:
    with pytest.raises(NaiveDatetimeError):
        make_signal_id(make_entity_id(Market.TW, "2330"), "v1", datetime(2020, 3, 19))  # noqa: DTZ001


def test_ids_are_stable_across_separate_processes() -> None:
    """PYTHONHASHSEED 的隨機化不得洩漏到任何 id 裡。"""
    script = (
        "from datetime import datetime, UTC;"
        "from trading_intel.core.enums import Market;"
        "from trading_intel.core.ids import make_entity_id, make_evidence_id, make_signal_id;"
        "e=make_entity_id(Market.TW,'2330');"
        "print(e, make_evidence_id('twse', b'hello world'),"
        " make_signal_id(e,'v1.2.3',datetime(2020,3,19,12,0,tzinfo=UTC)))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    }
    assert outputs == {f"{EXPECTED_ENTITY_ID} {EXPECTED_EVIDENCE_ID} {EXPECTED_SIGNAL_ID}"}


def test_correlation_ids_are_unique() -> None:
    assert len({new_correlation_id() for _ in range(100)}) == 100
