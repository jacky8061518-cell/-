"""事中監控：分層停損、波動率控制、對帳、心跳（SPEC 7.2）。

這一層的設計原則是**失效時往安全的方向倒**。心跳中斷、對帳不符、
資料中斷——這些情況下系統不知道自己的真實狀態，此時唯一安全的動作是
停止建立新部位。

這就是 dead man's switch 的意義：不是「偵測到危險才停」，
而是「無法確認安全就停」。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import IntEnum, StrEnum

from trading_intel.core.clock import ensure_utc, utc_now
from trading_intel.core.enums import Severity
from trading_intel.core.ids import EntityId
from trading_intel.core.settings import MonitoringSettings, RiskLimits


class TradingMode(IntEnum):
    """交易模式。用 IntEnum 因為需要比大小取最嚴格者。"""

    NORMAL = 0
    #: 只降不增：可以減碼與平倉，不得建立或加碼。
    REDUCE_ONLY = 1
    #: 只平倉：唯一允許的動作是把部位歸零。
    CLOSE_ONLY = 2
    #: 完全凍結，等待人工處理。
    HALTED = 3


class TriggerReason(StrEnum):
    DRAWDOWN_DERISK = "DRAWDOWN_DERISK"
    DRAWDOWN_FLATTEN = "DRAWDOWN_FLATTEN"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    HEARTBEAT_LOST = "HEARTBEAT_LOST"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    ORDER_RATE_LIMIT = "ORDER_RATE_LIMIT"
    PRICE_DEVIATION = "PRICE_DEVIATION"


@dataclass(frozen=True)
class MonitoringAlert:
    """一次監控觸發。"""

    reason: TriggerReason
    severity: Severity
    mode: TradingMode
    detail: str
    triggered_at: datetime
    #: 建議的曝險乘數。1.0 維持、0.5 減半、0.0 全平。
    exposure_multiplier: float = 1.0


def check_drawdown(
    equity_curve: Mapping[datetime, Decimal],
    limits: RiskLimits,
    *,
    now: datetime | None = None,
) -> MonitoringAlert | None:
    """分層停損（SPEC 7.2）。

    回撤達第一層自動減半曝險，達第二層全部平倉並進入人工審核。
    兩層刻意不同動作：減碼是機械的，平倉之後則需要人來判斷要不要重啟。
    """
    if not equity_curve:
        return None
    stamp = utc_now() if now is None else ensure_utc(now)
    ordered = [equity_curve[key] for key in sorted(equity_curve)]
    peak = max(ordered)
    if peak <= 0:
        return None
    current = ordered[-1]
    drawdown = float((peak - current) / peak)

    if drawdown >= limits.drawdown_flatten:
        return MonitoringAlert(
            reason=TriggerReason.DRAWDOWN_FLATTEN,
            severity=Severity.P0,
            mode=TradingMode.CLOSE_ONLY,
            detail=(
                f"組合回撤 {drawdown:.2%} 達到全部平倉門檻 "
                f"{limits.drawdown_flatten:.2%}，進入人工審核"
            ),
            triggered_at=stamp,
            exposure_multiplier=0.0,
        )
    if drawdown >= limits.drawdown_derisk:
        return MonitoringAlert(
            reason=TriggerReason.DRAWDOWN_DERISK,
            severity=Severity.P0,
            mode=TradingMode.REDUCE_ONLY,
            detail=(f"組合回撤 {drawdown:.2%} 達到減碼門檻 {limits.drawdown_derisk:.2%}，曝險減半"),
            triggered_at=stamp,
            exposure_multiplier=0.5,
        )
    return None


def check_realized_volatility(
    realized_annual_volatility: float,
    target_annual_volatility: float,
    settings: MonitoringSettings,
    *,
    now: datetime | None = None,
) -> MonitoringAlert | None:
    """實現波動超過目標的設定倍數時自動降槓桿（SPEC 7.2）。"""
    stamp = utc_now() if now is None else ensure_utc(now)
    if target_annual_volatility <= 0:
        return None
    ratio = realized_annual_volatility / target_annual_volatility
    if ratio <= settings.realized_vol_multiple_for_derisk:
        return None
    # 降到讓實現波動回到目標所需的比例。
    multiplier = max(0.0, min(1.0, 1.0 / ratio))
    return MonitoringAlert(
        reason=TriggerReason.VOLATILITY_SPIKE,
        severity=Severity.P1,
        mode=TradingMode.REDUCE_ONLY,
        detail=(
            f"實現波動 {realized_annual_volatility:.2%} 為目標的 {ratio:.2f} 倍，"
            f"曝險調整為 {multiplier:.2f}"
        ),
        triggered_at=stamp,
        exposure_multiplier=multiplier,
    )


def check_heartbeat(
    last_heartbeat: datetime,
    settings: MonitoringSettings,
    *,
    now: datetime | None = None,
) -> MonitoringAlert | None:
    """Dead man's switch（SPEC 7.2）。

    心跳中斷代表系統無法確認自己的狀態。此時的正確動作不是繼續交易，
    也不是完全凍結（既有部位可能正在虧損），而是**只允許平倉**。
    """
    stamp = utc_now() if now is None else ensure_utc(now)
    elapsed = (stamp - ensure_utc(last_heartbeat)).total_seconds()
    if elapsed <= settings.heartbeat_timeout_seconds:
        return None
    return MonitoringAlert(
        reason=TriggerReason.HEARTBEAT_LOST,
        severity=Severity.P0,
        mode=TradingMode.CLOSE_ONLY,
        detail=(
            f"心跳中斷 {elapsed:.0f} 秒，超過 "
            f"{settings.heartbeat_timeout_seconds} 秒門檻，進入只平倉模式"
        ),
        triggered_at=stamp,
        exposure_multiplier=0.0,
    )


def check_reconciliation(
    internal: Mapping[EntityId, Decimal],
    broker: Mapping[EntityId, Decimal],
    *,
    tolerance: Decimal = Decimal("0.0001"),
    now: datetime | None = None,
) -> MonitoringAlert | None:
    """部位對帳。不符即凍結（SPEC 7.2）。

    對帳不符代表我們對自己的部位認知是錯的。在錯誤的部位認知下計算風險，
    比不計算更危險，因此直接凍結等人處理。
    """
    stamp = utc_now() if now is None else ensure_utc(now)
    mismatches: list[str] = []
    for entity_id in set(internal) | set(broker):
        ours = internal.get(entity_id, Decimal("0"))
        theirs = broker.get(entity_id, Decimal("0"))
        if abs(ours - theirs) > tolerance:
            mismatches.append(f"{entity_id}：內部 {ours} vs 券商 {theirs}")
    if not mismatches:
        return None
    return MonitoringAlert(
        reason=TriggerReason.RECONCILIATION_MISMATCH,
        severity=Severity.P0,
        mode=TradingMode.HALTED,
        detail="部位對帳不符，已凍結：" + "；".join(mismatches[:5]),
        triggered_at=stamp,
        exposure_multiplier=0.0,
    )


@dataclass
class OrderGuard:
    """下單速率上限、重複單防護、價格偏離防護（SPEC 7.2）。"""

    settings: MonitoringSettings
    _recent_orders: list[datetime] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set)

    def check_rate(self, now: datetime) -> MonitoringAlert | None:
        stamp = ensure_utc(now)
        window_start = stamp - timedelta(minutes=1)
        self._recent_orders = [item for item in self._recent_orders if item > window_start]
        if len(self._recent_orders) >= self.settings.max_orders_per_minute:
            return MonitoringAlert(
                reason=TriggerReason.ORDER_RATE_LIMIT,
                severity=Severity.P0,
                mode=TradingMode.HALTED,
                detail=(
                    f"一分鐘內下單 {len(self._recent_orders)} 筆，"
                    f"超過上限 {self.settings.max_orders_per_minute}"
                ),
                triggered_at=stamp,
                exposure_multiplier=0.0,
            )
        return None

    def record_order(self, now: datetime, idempotency_key: str) -> bool:
        """記錄一筆下單。回傳 False 代表這是重複單，應被丟棄。"""
        if idempotency_key in self._seen_keys:
            return False
        self._seen_keys.add(idempotency_key)
        self._recent_orders.append(ensure_utc(now))
        return True

    def check_price(
        self,
        order_price: Decimal,
        reference_price: Decimal,
        *,
        now: datetime,
    ) -> MonitoringAlert | None:
        """價格偏離防護。防的是打錯價格與行情跳動。"""
        if reference_price <= 0:
            return None
        deviation = abs(float(order_price / reference_price - Decimal("1")))
        if deviation <= self.settings.max_price_deviation:
            return None
        return MonitoringAlert(
            reason=TriggerReason.PRICE_DEVIATION,
            severity=Severity.P0,
            mode=TradingMode.REDUCE_ONLY,
            detail=(
                f"下單價 {order_price} 偏離參考價 {reference_price} 達 {deviation:.2%}，"
                f"超過 {self.settings.max_price_deviation:.2%} 門檻"
            ),
            triggered_at=ensure_utc(now),
            exposure_multiplier=1.0,
        )


def combine_alerts(alerts: list[MonitoringAlert | None]) -> tuple[TradingMode, float]:
    """彙整多個監控結果，取**最嚴格**者。

    刻意不做平均、不做投票：兩個警告加起來不會互相抵消，
    而任一個要求停止就該停止。
    """
    active = [alert for alert in alerts if alert is not None]
    if not active:
        return TradingMode.NORMAL, 1.0
    mode = max(alert.mode for alert in active)
    multiplier = min(alert.exposure_multiplier for alert in active)
    return mode, multiplier
