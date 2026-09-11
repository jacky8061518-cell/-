"""訊號融合（SPEC 6）。

**禁止對訊號分數做簡單平均。** SPEC 給的理由一句話講完：
「訊號間相關性高時等權等於重壓單一因子。」

具體一點：你有五個訊號，其中四個都是動量的變形。等權平均的結果是 80% 押在動量上，
但儀表板會顯示「五個訊號分散配置」。這個錯誤不會報錯，而且看起來像是在分散風險。

本模組的做法是：
1. 以 Ledoit-Wolf 收縮估計訊號之間的協方差（樣本協方差在訊號數接近樣本數時極不穩定）；
2. 由協方差推導權重，相關性高的訊號自動被合併計權；
3. 依 regime 條件調整；
4. 每個訊號按半衰期衰減，超過半衰期線性衰減至零。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from trading_intel.core.clock import ensure_utc
from trading_intel.core.errors import SchemaValidationError
from trading_intel.models.regime import MarketRegime, VolatilityState

FloatArray = npt.NDArray[np.float64]


def ledoit_wolf_shrinkage(returns: npt.ArrayLike) -> tuple[FloatArray, float]:
    """Ledoit-Wolf 收縮協方差估計。

    回傳 (收縮後的協方差矩陣, 收縮強度)。

    為什麼不用樣本協方差：當訊號數 N 接近樣本數 T 時，樣本協方差矩陣的
    最小特徵值會被嚴重低估，而組合最佳化恰好會把最大權重押在那些方向上。
    結果是最佳化器專挑估計誤差最大的地方下注。收縮把矩陣往對角拉，
    犧牲一點無偏性換取穩定性。
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise SchemaValidationError("報酬矩陣必須是二維（期數 × 訊號數）")
    n_samples, n_features = matrix.shape
    if n_samples < 2 or n_features < 1:
        raise SchemaValidationError("樣本不足以估計協方差", samples=n_samples, features=n_features)

    centered = matrix - matrix.mean(axis=0)
    sample = centered.T @ centered / n_samples

    # 收縮目標：等變異數的對角矩陣。
    mean_variance = float(np.trace(sample)) / n_features
    target = mean_variance * np.eye(n_features)

    # 依 Ledoit-Wolf 推導估計最適收縮強度。
    delta = float(np.sum((sample - target) ** 2))
    if delta == 0:
        return sample, 0.0
    beta_sum = 0.0
    for row in centered:
        outer = np.outer(row, row)
        beta_sum += float(np.sum((outer - sample) ** 2))
    beta = beta_sum / (n_samples**2)
    shrinkage = max(0.0, min(1.0, beta / delta))
    return (1 - shrinkage) * sample + shrinkage * target, shrinkage


def correlation_from_covariance(covariance: npt.ArrayLike) -> FloatArray:
    """由協方差矩陣求相關矩陣。變異數為零的維度回傳 NaN。"""
    matrix = np.asarray(covariance, dtype=np.float64)
    deviations = np.sqrt(np.diag(matrix))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = matrix / np.outer(deviations, deviations)
    return np.asarray(correlation, dtype=np.float64)


def inverse_variance_weights(covariance: npt.ArrayLike) -> FloatArray:
    """以變異數倒數配權，並依相關性結構調整。

    這不是簡單平均：兩個相關係數 0.95 的訊號，合計拿到的權重接近單一訊號，
    而不是兩份。實作方式是用協方差矩陣的列和作為「有效重複度」的代理。
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    size = matrix.shape[0]
    if size == 0:
        return np.zeros(0)
    variances = np.diag(matrix).copy()
    variances[variances <= 0] = np.nan

    correlation = correlation_from_covariance(matrix)
    np.fill_diagonal(correlation, 1.0)
    # 有效重複度：與其他訊號的相關性總和。高度相關者的權重被稀釋。
    redundancy = np.nansum(np.abs(correlation), axis=1)
    redundancy[~np.isfinite(redundancy)] = 1.0
    redundancy[redundancy <= 0] = 1.0

    raw = 1.0 / (variances * redundancy)
    raw[~np.isfinite(raw)] = 0.0
    total = float(np.sum(raw))
    if total <= 0:
        return np.full(size, 1.0 / size)
    return np.asarray(raw / total, dtype=np.float64)


def half_life_decay(
    signal_time: datetime,
    now: datetime,
    *,
    half_life_days: float,
) -> float:
    """訊號的時間衰減係數（SPEC 6）。

    SPEC 要求「超過半衰期線性衰減至零，不得無限期持有」。
    這裡採指數衰減至半衰期，之後線性歸零於兩倍半衰期——
    純指數衰減永遠不會到零，而「永遠留著一點點」就是無限期持有。
    """
    if half_life_days <= 0:
        raise SchemaValidationError("半衰期必須為正數", half_life_days=half_life_days)
    elapsed_days = (ensure_utc(now) - ensure_utc(signal_time)).total_seconds() / 86400
    if elapsed_days < 0:
        # 訊號時間在未來，視為尚未生效。
        return 0.0
    if elapsed_days <= half_life_days:
        return float(math.pow(0.5, elapsed_days / half_life_days))
    if elapsed_days >= 2 * half_life_days:
        return 0.0
    # 半衰期到兩倍半衰期之間，由 0.5 線性降到 0。
    fraction = (elapsed_days - half_life_days) / half_life_days
    return float(0.5 * (1 - fraction))


#: 各 regime 下的訊號類別權重乘數（SPEC 6：依 regime 條件加權）。
#: 高波動時降低動量權重、提高均值回歸權重，是文獻上較穩健的調整方向。
_REGIME_MULTIPLIERS: Mapping[VolatilityState, Mapping[str, float]] = {
    VolatilityState.LOW: {"momentum": 1.2, "mean_reversion": 0.8, "event": 1.0, "flow": 1.0},
    VolatilityState.NORMAL: {"momentum": 1.0, "mean_reversion": 1.0, "event": 1.0, "flow": 1.0},
    VolatilityState.HIGH: {"momentum": 0.7, "mean_reversion": 1.2, "event": 0.9, "flow": 1.0},
    VolatilityState.EXTREME: {"momentum": 0.4, "mean_reversion": 0.8, "event": 0.5, "flow": 0.8},
}


def regime_multiplier(regime: MarketRegime, signal_family: str) -> float:
    """取某個訊號類別在當前 regime 下的權重乘數。"""
    table = _REGIME_MULTIPLIERS.get(regime.volatility, {})
    return table.get(signal_family, 1.0)


@dataclass(frozen=True)
class SignalInput:
    """融合的輸入單位。"""

    name: str
    family: str
    score: float
    confidence: float
    signal_time: datetime
    half_life_days: float


@dataclass(frozen=True)
class FusionResult:
    """融合結果，含每個訊號的最終權重，讓歸因看得出來源。"""

    combined_score: float
    weights: Mapping[str, float]
    shrinkage: float
    dropped: tuple[str, ...]


def fuse(
    signals: Sequence[SignalInput],
    *,
    history: npt.ArrayLike | None,
    regime: MarketRegime,
    now: datetime,
    min_weight: float = 0.05,
) -> FusionResult:
    """融合多個訊號成單一分數。

    ``history`` 是 (期數 × 訊號數) 的歷史報酬矩陣，用於估計協方差。
    為 None 時退回等權——但這是最後手段，不是預設路徑。
    """
    if not signals:
        return FusionResult(combined_score=0.0, weights={}, shrinkage=0.0, dropped=())

    decays = np.array(
        [
            half_life_decay(item.signal_time, now, half_life_days=item.half_life_days)
            for item in signals
        ]
    )

    if history is not None:
        covariance, shrinkage = ledoit_wolf_shrinkage(history)
        if covariance.shape[0] != len(signals):
            raise SchemaValidationError(
                "歷史矩陣的欄數與訊號數不符",
                history_columns=covariance.shape[0],
                signals=len(signals),
            )
        base = inverse_variance_weights(covariance)
    else:
        shrinkage = 0.0
        base = np.full(len(signals), 1.0 / len(signals))

    multipliers = np.array([regime_multiplier(regime, item.family) for item in signals])
    confidences = np.array([item.confidence for item in signals])
    effective = base * multipliers * confidences * decays

    dropped = tuple(
        item.name
        for item, weight in zip(signals, effective, strict=True)
        if weight < min_weight * max(float(np.sum(effective)), 1e-12)
    )
    effective = np.where(
        effective < min_weight * max(float(np.sum(effective)), 1e-12), 0.0, effective
    )

    total = float(np.sum(effective))
    if total <= 0:
        return FusionResult(
            combined_score=0.0,
            weights=dict.fromkeys((item.name for item in signals), 0.0),
            shrinkage=shrinkage,
            dropped=tuple(item.name for item in signals),
        )
    normalized = effective / total
    scores = np.array([item.score for item in signals])
    combined = float(np.sum(normalized * scores))

    return FusionResult(
        combined_score=max(-1.0, min(1.0, combined)),
        weights={
            item.name: float(weight) for item, weight in zip(signals, normalized, strict=True)
        },
        shrinkage=shrinkage,
        dropped=dropped,
    )
