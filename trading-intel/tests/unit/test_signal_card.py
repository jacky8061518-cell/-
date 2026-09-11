"""訊號卡（SPEC 8.1）：失效條件與反方論點是硬性要求，不是文件建議。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading_intel.agents.schemas import AttackFinding, AttackSeverity, AttackVector, RedTeamReport
from trading_intel.core.clock import UTC
from trading_intel.core.enums import DecisionLevel, Direction, Horizon, Market
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import EvidenceId, SignalId, make_entity_id
from trading_intel.core.types import Signal
from trading_intel.surface.signal_card import build_signal_card

TSMC = make_entity_id(Market.TW, "2330")
T0 = datetime(2024, 3, 19, 12, 0, tzinfo=UTC)


def make_signal(**overrides: object) -> Signal:
    defaults: dict[str, object] = {
        "event_time": T0,
        "ingest_time": T0 + timedelta(minutes=1),
        "signal_id": SignalId("sig1"),
        "entity_id": TSMC,
        "direction": Direction.LONG,
        "score": 0.6,
        "confidence": 0.75,
        "half_life_days": 10.0,
        "invalidation_condition": "月營收年增率轉為負成長",
        "horizon": Horizon.WEEKS,
        "evidence_ids": (EvidenceId("ev1"),),
        "model_version": "v1.0.0",
        "decision_level": DecisionLevel.CONFIRM,
    }
    defaults.update(overrides)
    return Signal(**defaults)


def make_redteam_report(**overrides: object) -> RedTeamReport:
    findings = tuple(
        AttackFinding(vector=vector, severity=AttackSeverity.LOW, detail="檢查通過")
        for vector in AttackVector
    )
    defaults: dict[str, object] = {
        "findings": findings,
        "confidence": 0.8,
        "reasoning_digest": "八項均已檢視",
    }
    defaults.update(overrides)
    return RedTeamReport(**defaults)


def test_card_carries_the_invalidation_condition() -> None:
    """SPEC 8.1：失效條件是最重要的欄位。"""
    signal = make_signal(invalidation_condition="法人連續三日賣超逾五千張")
    card = build_signal_card(signal, make_redteam_report(), generated_at=T0)
    assert card.invalidation_condition == "法人連續三日賣超逾五千張"


def test_card_without_redteam_report_is_rejected() -> None:
    """SPEC 8.1：反方論點是必填。沒有 RedTeamAgent 報告不得產卡。"""
    empty_report = RedTeamReport.model_construct(findings=(), confidence=0.0, reasoning_digest="")
    with pytest.raises(SchemaValidationError, match="拒絕在沒有 RedTeamAgent 報告"):
        build_signal_card(make_signal(), empty_report, generated_at=T0)


def test_counter_arguments_are_populated_from_redteam_findings() -> None:
    report = make_redteam_report(
        findings=(
            AttackFinding(
                vector=AttackVector.CROWDING, severity=AttackSeverity.HIGH, detail="因子擁擠"
            ),
            *[
                AttackFinding(vector=v, severity=AttackSeverity.NOT_APPLICABLE, detail="不適用")
                for v in list(AttackVector)[1:]
            ],
        )
    )
    card = build_signal_card(make_signal(), report, generated_at=T0)
    assert len(card.counter_arguments) == 1
    assert card.counter_arguments[0].text == "因子擁擠"


def test_not_applicable_findings_are_excluded_from_counter_arguments() -> None:
    """不適用的檢查項不該以「反方論點」的姿態出現在卡片上混淆視聽。"""
    report = make_redteam_report()  # 全部 LOW，皆非 NOT_APPLICABLE
    card = build_signal_card(make_signal(), report, generated_at=T0)
    assert len(card.counter_arguments) == 8


def test_high_severity_counter_argument_is_flagged_as_blocking() -> None:
    report = make_redteam_report(
        findings=(
            AttackFinding(
                vector=AttackVector.DATA_LEAKAGE,
                severity=AttackSeverity.CRITICAL,
                detail="用到未來資料",
            ),
            *[
                AttackFinding(vector=v, severity=AttackSeverity.LOW, detail="通過")
                for v in list(AttackVector)[1:]
            ],
        )
    )
    card = build_signal_card(make_signal(), report, generated_at=T0)
    assert card.has_blocking_counter_argument


def test_low_severity_findings_do_not_block() -> None:
    card = build_signal_card(make_signal(), make_redteam_report(), generated_at=T0)
    assert not card.has_blocking_counter_argument


def test_display_text_includes_all_required_fields() -> None:
    """SPEC 8.1 清單：方向、標的、進場區間、失效條件、持有期、信心度、證據、反方論點。"""
    card = build_signal_card(
        make_signal(), make_redteam_report(), generated_at=T0, entry_range=(95.0, 100.0)
    )
    text = card.to_display_text()
    assert "LONG" in text
    assert "TW:2330" in text
    assert "95.00" in text and "100.00" in text
    assert "月營收年增率轉為負成長" in text
    assert "WEEKS" in text
    assert "75%" in text


def test_card_preserves_evidence_ids() -> None:
    card = build_signal_card(make_signal(), make_redteam_report(), generated_at=T0)
    assert card.evidence_ids == (EvidenceId("ev1"),)


def test_card_without_entry_range_omits_it_from_display() -> None:
    card = build_signal_card(make_signal(), make_redteam_report(), generated_at=T0)
    assert "進場區間" not in card.to_display_text()
