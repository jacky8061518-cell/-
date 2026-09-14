"""組合建構與波動率目標化（SPEC 6）。

從「訊號分數」到「目標權重」的轉換。這一步有兩個容易被低估的地方：

**一、約束不是事後修剪。** 先算出理想權重再砍到限額內，會得到一個既不理想
也不符合原意的組合。約束應該在配置時就生效。

**二、換手率懲罰是必要的，不是選配。** 沒有它，訊號的微小變動會導致大幅換倉，
而台股的證交稅會把那些換倉的價值全部吃掉（SPEC 第 12 節）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from trading_intel.core.ids import EntityId
from trading_intel.core.settings import PortfolioSettings, RiskLimits

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ConstructionResult:
    """組合建構的結果，含被約束修剪掉的部分，讓決策可追溯。"""

    weights: Mapping[EntityId, float]
    #: 波動率目標化推導出的曝險倍數。
    exposure_scalar: float
    #: 相對前一期的換手率。
    turnover: float
    #: 因約束而被調整的標的。
    constrained: tuple[EntityId, ...]


def volatility_target_scalar(
    forecast_annual_volatility: float,
    settings: PortfolioSettings,
) -> float:
    """由目標波動反推總曝險倍數（SPEC 6）。

    市場波動上升時自動降槓桿——這是波動率目標化唯一的作用，
    但它是少數在多數市場環境下都站得住腳的風控機制。
    """
    if forecast_annual_volatility <= 0 or not math.isfinite(forecast_annual_volatility):
        return 0.0
    scalar = settings.target_annual_volatility / forecast_annual_volatility
    return max(0.0, min(scalar, settings.max_leverage_from_vol_target))


def build_portfolio(
    scores: Mapping[EntityId, float],
    *,
    previous: Mapping[EntityId, float] | None = None,
    sectors: Mapping[EntityId, str] | None = None,
    forecast_annual_volatility: float,
    limits: RiskLimits,
    settings: PortfolioSettings,
) -> ConstructionResult:
    """把訊號分數轉成符合所有約束的目標權重。

    順序是刻意的：先依分數配權 → 套單一標的上限 → 套產業上限 →
    波動率目標化 → 套總曝險上限 → 換手率懲罰 → 清掉碎股。

    單一標的上限放在產業上限之前，因為前者是硬性的（法規與流動性），
    後者是政策性的（可以透過調整組合達成）。
    """
    if not scores:
        return ConstructionResult(weights={}, exposure_scalar=0.0, turnover=0.0, constrained=())

    entity_ids = list(scores)
    raw = np.array([scores[entity_id] for entity_id in entity_ids], dtype=np.float64)
    raw[~np.isfinite(raw)] = 0.0

    magnitude = np.abs(raw).sum()
    if magnitude <= 0:
        return ConstructionResult(
            weights=dict.fromkeys(entity_ids, 0.0),
            exposure_scalar=0.0,
            turnover=0.0,
            constrained=(),
        )
    weights = raw / magnitude
    constrained: set[EntityId] = set()

    # 單一標的上限。
    capped = np.clip(weights, -limits.max_position_weight, limits.max_position_weight)
    for index, entity_id in enumerate(entity_ids):
        if not math.isclose(capped[index], weights[index], rel_tol=1e-12):
            constrained.add(entity_id)
    weights = capped

    # 產業上限。
    if sectors:
        weights, sector_constrained = _apply_sector_cap(
            weights, entity_ids, sectors, limits.max_sector_weight
        )
        constrained |= sector_constrained

    # 波動率目標化。
    scalar = volatility_target_scalar(forecast_annual_volatility, settings)
    weights = weights * scalar

    # 總曝險上限。
    gross = float(np.abs(weights).sum())
    if gross > limits.max_gross_exposure and gross > 0:
        weights = weights * (limits.max_gross_exposure / gross)

    # 換手率懲罰：朝前一期的權重拉回一點，減少無謂換倉。
    previous_weights = previous or {}
    if previous_weights and settings.turnover_penalty > 0:
        prior = np.array(
            [previous_weights.get(entity_id, 0.0) for entity_id in entity_ids],
            dtype=np.float64,
        )
        # 懲罰係數越大，越傾向維持原部位。
        blend = min(0.9, settings.turnover_penalty * 100)
        weights = (1 - blend) * weights + blend * prior

    # 清掉碎股：低於最小權重直接歸零，避免產生無意義的小單。
    weights = np.where(np.abs(weights) < settings.min_position_weight, 0.0, weights)

    final = {
        entity_id: float(weight) for entity_id, weight in zip(entity_ids, weights, strict=True)
    }
    turnover = _turnover(final, previous_weights)
    return ConstructionResult(
        weights=final,
        exposure_scalar=scalar,
        turnover=turnover,
        constrained=tuple(sorted(constrained)),
    )


def _apply_sector_cap(
    weights: FloatArray,
    entity_ids: list[EntityId],
    sectors: Mapping[EntityId, str],
    cap: float,
) -> tuple[FloatArray, set[EntityId]]:
    """把超過產業上限的部分等比例縮回。"""
    adjusted = weights.copy()
    constrained: set[EntityId] = set()
    by_sector: dict[str, list[int]] = {}
    for index, entity_id in enumerate(entity_ids):
        by_sector.setdefault(sectors.get(entity_id, "未分類"), []).append(index)

    for indices in by_sector.values():
        exposure = float(np.abs(adjusted[indices]).sum())
        if exposure > cap and exposure > 0:
            adjusted[indices] *= cap / exposure
            constrained.update(entity_ids[index] for index in indices)
    return adjusted, constrained


def _turnover(
    current: Mapping[EntityId, float],
    previous: Mapping[EntityId, float],
) -> float:
    if not previous:
        return float(sum(abs(weight) for weight in current.values()))
    keys = set(current) | set(previous)
    return float(sum(abs(current.get(key, 0.0) - previous.get(key, 0.0)) for key in keys))
