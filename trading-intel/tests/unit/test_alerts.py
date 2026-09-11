"""分級告警（SPEC 8.2）：去重、速率上限、novelty 檢查三道防線。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market, Severity
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import EntityId, make_entity_id
from trading_intel.core.settings import AlertingSettings
from trading_intel.surface.alerts import Alert, AlertRouter, Digest

TSMC = make_entity_id(Market.TW, "2330")
UMC = make_entity_id(Market.TW, "2303")
T0 = datetime(2024, 3, 19, 9, 0, tzinfo=UTC)

SETTINGS = AlertingSettings(
    max_alerts_per_hour=3,
    novelty_window_hours=24,
    dedup_similarity_threshold=0.85,
)


def alert(
    summary: str = "外資買超放大",
    severity: Severity = Severity.P1,
    entity_id: EntityId | None = TSMC,
    at: datetime = T0,
) -> Alert:
    return Alert(
        entity_id=entity_id,
        severity=severity,
        summary=summary,
        action="留意是否持續，考慮加碼觀察名單",
        created_at=at,
    )


# --- 「所以呢」是硬性要求 ---------------------------------------------------


def test_alert_without_action_is_rejected() -> None:
    """SPEC 8.2：每則告警必須包含「所以呢」，不得只丟數字。"""
    with pytest.raises(SchemaValidationError, match="所以呢"):
        Alert(
            entity_id=TSMC, severity=Severity.P1, summary="外資買超放大", action="", created_at=T0
        )


def test_alert_with_whitespace_only_action_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="所以呢"):
        Alert(entity_id=TSMC, severity=Severity.P1, summary="s", action="   ", created_at=T0)


# --- P0 一律直接送出 ---------------------------------------------------------


def test_p0_bypasses_all_throttling() -> None:
    """P0 之下沒有更嚴重的層級，一律直接送出。"""
    router = AlertRouter(settings=SETTINGS)
    for _ in range(10):
        result = router.submit(alert(severity=Severity.P0, summary="風控觸發全部平倉"))
        assert isinstance(result, Alert)


# --- 去重 --------------------------------------------------------------------


def test_identical_alert_is_deduplicated() -> None:
    """SPEC 8.2：同一事件不重複發。"""
    router = AlertRouter(settings=SETTINGS)
    first = router.submit(alert(at=T0))
    second = router.submit(alert(at=T0 + timedelta(minutes=1)))
    assert isinstance(first, Alert)
    assert second is None


def test_similar_wording_is_also_deduplicated() -> None:
    """措辭略有不同但本質相同的告警，同樣視為重複。"""
    router = AlertRouter(settings=SETTINGS)
    router.submit(alert(summary="台積電外資買超放大", at=T0))
    result = router.submit(alert(summary="台積電外資買超略放大", at=T0 + timedelta(minutes=1)))
    assert result is None


def test_different_entity_is_not_deduplicated() -> None:
    router = AlertRouter(settings=SETTINGS)
    router.submit(alert(entity_id=TSMC, at=T0))
    result = router.submit(alert(entity_id=UMC, at=T0 + timedelta(minutes=1)))
    assert isinstance(result, Alert)


def test_genuinely_different_content_is_not_deduplicated() -> None:
    router = AlertRouter(settings=SETTINGS)
    router.submit(alert(summary="外資買超放大", at=T0))
    result = router.submit(alert(summary="投信轉為賣超", at=T0 + timedelta(minutes=1)))
    assert isinstance(result, Alert)


# --- 速率上限：超過改為彙整 --------------------------------------------------


def test_rate_limit_switches_to_digest() -> None:
    """SPEC 8.2：每小時上限 N 則，超過改為彙整。"""
    router = AlertRouter(settings=SETTINGS)
    results = [
        router.submit(alert(summary=f"事件 {i}", at=T0 + timedelta(minutes=i))) for i in range(5)
    ]
    assert all(isinstance(item, Alert) for item in results[:3])
    assert isinstance(results[3], Digest)
    assert isinstance(results[4], Digest)


def test_digest_contains_the_alerts_from_the_past_hour() -> None:
    router = AlertRouter(settings=SETTINGS)
    for i in range(4):
        result = router.submit(alert(summary=f"事件 {i}", at=T0 + timedelta(minutes=i)))
    assert isinstance(result, Digest)
    assert len(result.alerts) == 4


def test_rate_limit_is_per_entity() -> None:
    """速率上限是逐標的計算，不同標的互不影響。"""
    router = AlertRouter(settings=SETTINGS)
    for i in range(3):
        router.submit(alert(entity_id=TSMC, summary=f"a{i}", at=T0 + timedelta(minutes=i)))
    result = router.submit(alert(entity_id=UMC, summary="b", at=T0 + timedelta(minutes=5)))
    assert isinstance(result, Alert)


def test_rate_limit_window_slides() -> None:
    router = AlertRouter(settings=SETTINGS)
    for i in range(3):
        router.submit(alert(summary=f"a{i}", at=T0 + timedelta(minutes=i)))
    result = router.submit(alert(summary="b", at=T0 + timedelta(hours=2)))
    assert isinstance(result, Alert)


# --- novelty 檢查 -------------------------------------------------------------


def test_stale_novelty_is_suppressed() -> None:
    """SPEC 8.2：過去 24 小時已告知過的內容降級。"""
    router = AlertRouter(settings=SETTINGS)
    router.submit(alert(summary="外資買超放大", at=T0))
    result = router.submit(alert(summary="外資買超放大", at=T0 + timedelta(hours=12)))
    assert result is None


def test_content_outside_the_novelty_window_is_fresh_again() -> None:
    router = AlertRouter(settings=SETTINGS)
    router.submit(alert(summary="外資買超放大", at=T0))
    result = router.submit(alert(summary="外資買超放大", at=T0 + timedelta(hours=25)))
    assert isinstance(result, Alert)


# --- 顯示格式 ------------------------------------------------------------------


def test_display_text_includes_the_action() -> None:
    text = alert().to_display_text()
    assert "所以呢" not in text  # 這是概念名稱，不是要出現在畫面上的字
    assert "留意是否持續" in text


def test_digest_display_text_lists_all_alerts() -> None:
    digest = Digest(
        entity_id=TSMC,
        severity=Severity.P2,
        alerts=(alert(summary="事件一"), alert(summary="事件二")),
        created_at=T0,
    )
    text = digest.to_display_text()
    assert "事件一" in text
    assert "事件二" in text
    assert "2 則" in text
