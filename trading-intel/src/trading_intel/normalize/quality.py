"""資料品質閘門（SPEC 3.3）。

五類檢查，任一失敗則該標的當日狀態設為 ``NO_TRADE``。

這裡的核心紀律寫在 CLAUDE.md 第 6 條：**資料壞掉時停止交易，不是輸出垃圾。**
所以本模組沒有任何插值、前值填補、或「盡量修一修再繼續」的路徑。
偵測到問題就產生告警並讓標的停止交易，因為在資料有問題時交易，
比不交易的期望損失大得多。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

from trading_intel.core.clock import ensure_utc
from trading_intel.core.enums import Market, QualityCheck, Severity, TradingState
from trading_intel.core.ids import EntityId
from trading_intel.core.settings import QualityLimits
from trading_intel.core.types import DataQualityAlert
from trading_intel.normalize.corporate_actions import RawPrice


def _alert(
    *,
    entity_id: EntityId | None,
    check: QualityCheck,
    severity: Severity,
    detail: str,
    event_time: datetime,
    ingest_time: datetime,
    state: TradingState = TradingState.NO_TRADE,
) -> DataQualityAlert:
    return DataQualityAlert(
        event_time=ensure_utc(event_time),
        ingest_time=ensure_utc(ingest_time),
        entity_id=entity_id,
        check=check,
        severity=severity,
        detail=detail,
        resulting_state=state,
    )


def check_freshness(
    prices: Sequence[RawPrice],
    *,
    entity_id: EntityId,
    asof: datetime,
    limits: QualityLimits,
) -> list[DataQualityAlert]:
    """時效性：最新一根 bar 距 ``asof`` 是否過久。"""
    boundary = ensure_utc(asof)
    visible = [price for price in prices if price.ingest_time <= boundary]
    if not visible:
        return [
            _alert(
                entity_id=entity_id,
                check=QualityCheck.FRESHNESS,
                severity=Severity.P0,
                detail="在 asof 之前沒有任何可見的價格資料",
                event_time=boundary,
                ingest_time=boundary,
            )
        ]
    newest = max(visible, key=lambda price: price.ingest_time)
    age = boundary - newest.ingest_time
    if age > timedelta(minutes=limits.max_staleness_minutes):
        return [
            _alert(
                entity_id=entity_id,
                check=QualityCheck.FRESHNESS,
                severity=Severity.P0,
                detail=(
                    f"最新資料已過期 {age.total_seconds() / 60:.0f} 分鐘，"
                    f"上限為 {limits.max_staleness_minutes} 分鐘"
                ),
                event_time=boundary,
                ingest_time=boundary,
            )
        ]
    return []


def check_completeness(
    prices: Sequence[RawPrice],
    expected_sessions: Sequence[date],
    *,
    entity_id: EntityId,
    asof: datetime,
    limits: QualityLimits,
) -> list[DataQualityAlert]:
    """完整性：交易日缺 bar，以及成交量為零但價格有變動。"""
    boundary = ensure_utc(asof)
    visible = [price for price in prices if price.ingest_time <= boundary]
    by_date = {price.trade_date: price for price in visible}
    alerts: list[DataQualityAlert] = []

    missing = [session for session in expected_sessions if session not in by_date]
    if expected_sessions and missing:
        ratio = len(missing) / len(expected_sessions)
        if ratio > limits.max_missing_session_ratio:
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.COMPLETENESS,
                    severity=Severity.P0,
                    detail=(
                        f"缺少 {len(missing)}/{len(expected_sessions)} 個交易日"
                        f"（{ratio:.1%}），首個缺漏日 {missing[0]}"
                    ),
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )

    # 成交量為零卻有價格變動：交易所不會這樣，必是資料錯誤。
    ordered = sorted(by_date.values(), key=lambda price: price.trade_date)
    for previous, current in pairwise(ordered):
        if current.volume == 0 and current.close != previous.close:
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.COMPLETENESS,
                    severity=Severity.P1,
                    detail=(
                        f"{current.trade_date} 成交量為零但價格自 {previous.close} "
                        f"變動為 {current.close}"
                    ),
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )
    return alerts


def check_raw_sanity(
    rows: Sequence[tuple[date, Decimal, int]],
    *,
    entity_id: EntityId,
    asof: datetime,
) -> list[DataQualityAlert]:
    """合理性（原始值層）：負成交量、非正價格。

    參數為 ``(日期, close, volume)``，刻意不吃 ``RawPrice``——因為 ``RawPrice``
    的欄位限制讓這些值根本建構不起來。品質閘門的職責正是在資料變成型別**之前**
    把關；等到型別驗證失敗才發現，就只剩下一個例外，沒有告警、沒有 NO_TRADE 狀態。
    """
    boundary = ensure_utc(asof)
    alerts: list[DataQualityAlert] = []
    for trade_date, close, volume in rows:
        if volume < 0:
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.SANITY,
                    severity=Severity.P0,
                    detail=f"{trade_date} 成交量為負數：{volume}",
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )
        if close <= 0:
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.SANITY,
                    severity=Severity.P0,
                    detail=f"{trade_date} 收盤價非正數：{close}",
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )
    return alerts


def check_sanity(
    prices: Sequence[RawPrice],
    *,
    entity_id: EntityId,
    market: Market,
    asof: datetime,
    limits: QualityLimits,
) -> list[DataQualityAlert]:
    """合理性（序列層）：單日報酬超過該市場的漲跌幅限制。

    台股有 10% 漲跌幅，因此更大的跳動必然是未登錄的企業行動而非真實報酬。
    這正是既有 sector_rotation 專案用來偵測企業行動的領域知識，
    差別在於這裡只發告警，不回頭修改歷史價格（見 ADR 0005）。
    """
    boundary = ensure_utc(asof)
    visible = sorted(
        (price for price in prices if price.ingest_time <= boundary),
        key=lambda price: price.trade_date,
    )
    limit = limits.tw_daily_return_limit if market is Market.TW else limits.us_daily_return_limit
    alerts: list[DataQualityAlert] = []
    for previous, current in pairwise(visible):
        if previous.close <= 0:
            continue
        move = abs(current.close / previous.close - Decimal("1"))
        if move > Decimal(str(limit)):
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.SANITY,
                    severity=Severity.P0,
                    detail=(
                        f"{current.trade_date} 單日報酬 {float(move):.1%} "
                        f"超過 {market.value} 市場的 {limit:.1%} 限制"
                        "（可能是未登錄的企業行動）"
                    ),
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )
    return alerts


def check_consistency(
    bars: Sequence[tuple[date, Decimal, Decimal, Decimal, Decimal]],
    *,
    entity_id: EntityId,
    asof: datetime,
) -> list[DataQualityAlert]:
    """一致性：high < low，或 close 落在 high low 之外。

    參數為 ``(日期, open, high, low, close)``，因為這一層檢查的正是那些
    連 ``Bar`` 型別都建構不起來的原始資料。
    """
    boundary = ensure_utc(asof)
    alerts: list[DataQualityAlert] = []
    for trade_date, open_, high, low, close in bars:
        problems: list[str] = []
        if high < low:
            problems.append(f"high {high} < low {low}")
        if not low <= close <= high:
            problems.append(f"close {close} 落在 [{low}, {high}] 之外")
        if not low <= open_ <= high:
            problems.append(f"open {open_} 落在 [{low}, {high}] 之外")
        if problems:
            alerts.append(
                _alert(
                    entity_id=entity_id,
                    check=QualityCheck.CONSISTENCY,
                    severity=Severity.P0,
                    detail=f"{trade_date} OHLC 不一致：{'；'.join(problems)}",
                    event_time=boundary,
                    ingest_time=boundary,
                )
            )
    return alerts


def population_stability_index(
    baseline: Sequence[float],
    current: Sequence[float],
    *,
    buckets: int = 10,
) -> float:
    """PSI：衡量分布相對訓練期漂移的程度（SPEC 3.3）。

    以 baseline 的分位數切桶，比較兩者的落點比例。
    空桶以一個極小值取代，避免 log(0)。
    """
    if not baseline or not current:
        msg = "baseline 與 current 都不得為空"
        raise ValueError(msg)
    if buckets < 2:
        msg = "buckets 至少為 2"
        raise ValueError(msg)

    ordered = sorted(baseline)
    edges = [
        ordered[min(len(ordered) - 1, round(index * len(ordered) / buckets))]
        for index in range(1, buckets)
    ]

    def distribute(values: Sequence[float]) -> list[float]:
        counts = [0] * buckets
        for value in values:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        floor = 1e-6
        return [max(count / len(values), floor) for count in counts]

    base_ratio = distribute(baseline)
    current_ratio = distribute(current)
    return sum(
        (curr - base) * math.log(curr / base)
        for base, curr in zip(base_ratio, current_ratio, strict=True)
    )


def check_drift(
    baseline: Sequence[float],
    current: Sequence[float],
    *,
    entity_id: EntityId | None,
    feature_name: str,
    asof: datetime,
    limits: QualityLimits,
) -> list[DataQualityAlert]:
    """分布漂移：PSI 超過 0.25 警示，超過 0.5 停用該特徵（SPEC 3.3）。"""
    boundary = ensure_utc(asof)
    psi = population_stability_index(baseline, current)
    if psi > limits.psi_disable:
        return [
            _alert(
                entity_id=entity_id,
                check=QualityCheck.DRIFT,
                severity=Severity.P0,
                detail=f"特徵 {feature_name} 的 PSI {psi:.3f} 超過停用門檻 {limits.psi_disable}",
                event_time=boundary,
                ingest_time=boundary,
                state=TradingState.NO_TRADE,
            )
        ]
    if psi > limits.psi_warn:
        return [
            _alert(
                entity_id=entity_id,
                check=QualityCheck.DRIFT,
                severity=Severity.P1,
                detail=f"特徵 {feature_name} 的 PSI {psi:.3f} 超過警示門檻 {limits.psi_warn}",
                event_time=boundary,
                ingest_time=boundary,
                # 僅警示，尚不停止交易。
                state=TradingState.TRADABLE,
            )
        ]
    return []


def resulting_state(alerts: Sequence[DataQualityAlert]) -> TradingState:
    """把一組告警彙整成該標的的交易狀態。

    只要有任何一個告警要求停止交易，結果就是停止——不做平均、不做投票。
    """
    for alert in alerts:
        if alert.resulting_state is not TradingState.TRADABLE:
            return alert.resulting_state
    return TradingState.TRADABLE
