"""分級告警（SPEC 8.2）。

SPEC 開門見山：「告警疲勞是這類系統最常見的死因。交易員關掉通知，
系統就等於不存在。」這一句話決定了本模組的優先順序——**抗疲勞機制
跟分級本身一樣重要，不是錦上添花**。

三道防線缺一不可：
1. 去重——同一事件不重複發；
2. 速率上限——超過就改成彙整，不是繼續轟炸；
3. novelty 檢查——過去 24 小時已經講過的事，降級處理。

每則告警還有一條硬性規定：**必須包含「所以呢」那一句話**。
一串數字不是告警，是資料傾印；告警要說清楚「所以你該做什麼」。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from trading_intel.core.clock import ensure_utc
from trading_intel.core.enums import Severity
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import EntityId
from trading_intel.core.settings import AlertingSettings

#: P0 之下沒有更嚴重的層級了，一律直接送出，不受速率上限與去重以外的節流。
_IMMEDIATE: frozenset[Severity] = frozenset({Severity.P0})


@dataclass(frozen=True)
class Alert:
    """一則告警。"""

    entity_id: EntityId | None
    severity: Severity
    summary: str
    #: 「所以呢」那句話。SPEC 8.2：「不得只丟數字」。
    action: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.action.strip():
            raise SchemaValidationError(
                "告警必須包含「所以呢」那一句話，不得只丟數字",
                entity_id=str(self.entity_id) if self.entity_id else None,
                summary=self.summary,
            )

    @property
    def content_key(self) -> str:
        """用於去重與 novelty 比對的內容指紋。"""
        return f"{self.entity_id}:{self.severity.value}:{_normalize(self.summary)}"

    def to_display_text(self) -> str:
        prefix = f"[{self.severity.value}]"
        target = f"（{self.entity_id}）" if self.entity_id else ""
        return f"{prefix}{target} {self.summary}\n→ {self.action}"


def _normalize(text: str) -> str:
    return "".join(text.split()).lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


@dataclass(frozen=True)
class Digest:
    """速率上限觸發後的彙整結果。"""

    entity_id: EntityId | None
    severity: Severity
    alerts: tuple[Alert, ...]
    created_at: datetime

    def to_display_text(self) -> str:
        target = f"（{self.entity_id}）" if self.entity_id else ""
        header = f"[彙整{target}] 過去一小時內 {len(self.alerts)} 則 {self.severity.value} 告警"
        body = "\n".join(f"  • {alert.summary}" for alert in self.alerts)
        return f"{header}\n{body}"


@dataclass
class AlertRouter:
    """告警路由器：去重、速率上限、novelty 檢查三道防線的執行者。"""

    settings: AlertingSettings
    _recent: deque[Alert] = field(default_factory=deque)
    #: 每個標的過去一小時內已送出的告警數，用於速率上限。
    _hourly_count: dict[EntityId | None, deque[datetime]] = field(default_factory=dict)

    def submit(self, alert: Alert) -> Alert | Digest | None:
        """提交一則告警。回傳實際要送出的東西：

        - ``Alert``：正常送出；
        - ``Digest``：速率超限，改為彙整；
        - ``None``：被去重或 novelty 檢查擋下，不送。
        """
        now = ensure_utc(alert.created_at)
        self._prune(now)

        if alert.severity not in _IMMEDIATE:
            if self._is_duplicate(alert):
                return None
            if self._is_stale_novelty(alert, now):
                return None

        self._recent.append(alert)

        if alert.severity in _IMMEDIATE:
            return alert

        bucket = self._hourly_count.setdefault(alert.entity_id, deque())
        bucket.append(now)
        if len(bucket) > self.settings.max_alerts_per_hour:
            same_bucket_alerts = tuple(
                item
                for item in self._recent
                if item.entity_id == alert.entity_id
                and ensure_utc(item.created_at) > now - timedelta(hours=1)
            )
            return Digest(
                entity_id=alert.entity_id,
                severity=alert.severity,
                alerts=same_bucket_alerts,
                created_at=now,
            )
        return alert

    def _is_duplicate(self, alert: Alert) -> bool:
        """同一事件不重複發：內容指紋完全相同，或高度相似。"""
        for item in self._recent:
            if item.content_key == alert.content_key:
                return True
            same_bucket = item.entity_id == alert.entity_id and item.severity == alert.severity
            if same_bucket and _similarity(item.summary, alert.summary) >= (
                self.settings.dedup_similarity_threshold
            ):
                return True
        return False

    def _is_stale_novelty(self, alert: Alert, now: datetime) -> bool:
        """過去 novelty_window_hours 內已經講過的內容，降級（不送）。"""
        window_start = now - timedelta(hours=self.settings.novelty_window_hours)
        for item in self._recent:
            if ensure_utc(item.created_at) < window_start:
                continue
            if item.entity_id != alert.entity_id:
                continue
            if _similarity(item.summary, alert.summary) >= self.settings.dedup_similarity_threshold:
                return True
        return False

    def _prune(self, now: datetime) -> None:
        """清掉超出 novelty 視窗的舊紀錄，避免無限成長。"""
        window_start = now - timedelta(hours=self.settings.novelty_window_hours)
        while self._recent and ensure_utc(self._recent[0].created_at) < window_start:
            self._recent.popleft()
        for bucket in self._hourly_count.values():
            while bucket and bucket[0] < now - timedelta(hours=1):
                bucket.popleft()
