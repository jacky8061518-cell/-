"""市場狀態分類（SPEC 4.2）。

**這是純程式碼，不使用 LLM。** 原本規劃為 agent，改為確定性模組的理由是
它每天都要跑、輸入是數值、輸出是離散標籤——那正是 LLM 最不擅長而程式碼
最擅長的工作。

SPEC 要求「輸出必須離散且穩定，避免每日跳動」。穩定性是這個模組的核心難點：
一個每天在「趨勢」與「盤整」之間跳來跳去的分類器，會讓依 regime 加權的
訊號融合每天大幅換倉，成本吃光一切。因此本模組用遲滯機制
（hysteresis）：進入一個狀態的門檻比離開它的門檻嚴格。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from trading_intel.features.statistics import ewma_volatility

FloatArray = npt.NDArray[np.float64]

#: 狀態至少要維持這麼多期才允許切換。遲滯的核心參數。
MIN_REGIME_DURATION: Final = 5

TRADING_DAYS_PER_YEAR: Final = 252


class TrendState(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"


class VolatilityState(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class LiquidityState(StrEnum):
    NORMAL = "NORMAL"
    STRESSED = "STRESSED"


@dataclass(frozen=True)
class MarketRegime:
    """某一時點的市場狀態。四個維度各自離散。"""

    trend: TrendState
    volatility: VolatilityState
    liquidity: LiquidityState
    #: 橫斷面相關性。市場恐慌時個股齊漲齊跌，分散效果消失。
    cross_sectional_correlation: float

    @property
    def label(self) -> str:
        return f"{self.trend.value}/{self.volatility.value}/{self.liquidity.value}"

    @property
    def is_risk_off(self) -> bool:
        """需要降低曝險的狀態組合。"""
        return (
            self.volatility in {VolatilityState.HIGH, VolatilityState.EXTREME}
            or self.liquidity is LiquidityState.STRESSED
            or self.cross_sectional_correlation > 0.7
        )


def classify_trend(
    prices: npt.ArrayLike,
    *,
    short_window: int = 20,
    long_window: int = 60,
    threshold: float = 0.02,
) -> TrendState:
    """以短長期均線的相對位置判斷趨勢。

    ``threshold`` 提供一個中性區間：兩條均線只差 0.5% 不算趨勢，
    那只是雜訊。沒有這個區間，分類會每天跳動。
    """
    series = np.asarray(prices, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]
    if series.size < long_window:
        return TrendState.RANGING
    short = float(np.mean(series[-short_window:]))
    long = float(np.mean(series[-long_window:]))
    if long <= 0:
        return TrendState.RANGING
    deviation = short / long - 1.0
    if deviation > threshold:
        return TrendState.TRENDING_UP
    if deviation < -threshold:
        return TrendState.TRENDING_DOWN
    return TrendState.RANGING


def classify_volatility(
    returns: npt.ArrayLike,
    *,
    low_threshold: float = 0.12,
    high_threshold: float = 0.25,
    extreme_threshold: float = 0.40,
) -> VolatilityState:
    """以年化 EWMA 波動率分層。門檻為年化值。"""
    volatility = ewma_volatility(returns, span=20)
    if not math.isfinite(volatility):
        return VolatilityState.NORMAL
    if volatility >= extreme_threshold:
        return VolatilityState.EXTREME
    if volatility >= high_threshold:
        return VolatilityState.HIGH
    if volatility <= low_threshold:
        return VolatilityState.LOW
    return VolatilityState.NORMAL


def cross_sectional_correlation(returns_matrix: npt.ArrayLike) -> float:
    """橫斷面平均相關性。

    輸入為 (期數, 標的數) 的報酬矩陣。相關性升高代表個股齊漲齊跌，
    此時分散投資的保護作用下降——這是市場壓力最可靠的指標之一。
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[0] < 3:
        return float("nan")
    usable = matrix[:, np.isfinite(matrix).all(axis=0)]
    if usable.shape[1] < 2:
        return float("nan")
    # 標準差為零的欄位會讓相關係數變成 NaN。
    stds = usable.std(axis=0)
    usable = usable[:, stds > 0]
    if usable.shape[1] < 2:
        return float("nan")
    # np.corrcoef 在只有兩欄時的回傳型別讓靜態檢查無法收斂，明確轉成 2D 陣列。
    correlation: FloatArray = np.atleast_2d(
        np.asarray(np.corrcoef(usable, rowvar=False), dtype=np.float64)
    )
    rows, cols = np.triu_indices(correlation.shape[0], k=1)
    upper = correlation[rows, cols]
    finite = upper[np.isfinite(upper)]
    return float(np.mean(finite)) if finite.size else float("nan")


def classify_liquidity(
    volumes: npt.ArrayLike,
    *,
    lookback: int = 60,
    stress_ratio: float = 0.6,
) -> LiquidityState:
    """以近期成交量相對長期均量判斷流動性。

    量能萎縮到長期均量的六成以下，代表想出場時不一定出得掉。
    """
    series = np.asarray(volumes, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]
    if series.size < lookback:
        return LiquidityState.NORMAL
    recent = float(np.mean(series[-5:]))
    baseline = float(np.mean(series[-lookback:]))
    if baseline <= 0:
        return LiquidityState.NORMAL
    return LiquidityState.STRESSED if recent / baseline < stress_ratio else LiquidityState.NORMAL


@dataclass
class RegimeClassifier:
    """帶遲滯的狀態分類器。

    遲滯（hysteresis）是這個類別存在的唯一理由：純函式分類每天算一次會跳動，
    而跳動會讓下游每天大幅換倉。這裡要求新狀態連續出現
    ``min_duration`` 期才真正切換。
    """

    min_duration: int = MIN_REGIME_DURATION
    _current: MarketRegime | None = None
    _candidate: MarketRegime | None = None
    _candidate_count: int = 0

    @property
    def current(self) -> MarketRegime | None:
        return self._current

    def update(self, observed: MarketRegime) -> MarketRegime:
        """餵入當期觀察，回傳**生效中**的狀態（可能仍是舊的）。"""
        if self._current is None:
            self._current = observed
            self._candidate = None
            self._candidate_count = 0
            return self._current

        if observed.label == self._current.label:
            # 回到目前狀態，候選歸零。
            self._candidate = None
            self._candidate_count = 0
            return self._current

        if self._candidate is not None and observed.label == self._candidate.label:
            self._candidate_count += 1
        else:
            self._candidate = observed
            self._candidate_count = 1

        if self._candidate_count >= self.min_duration:
            self._current = self._candidate
            self._candidate = None
            self._candidate_count = 0
        return self._current


def classify(
    prices: npt.ArrayLike,
    returns: npt.ArrayLike,
    volumes: npt.ArrayLike,
    returns_matrix: npt.ArrayLike | None = None,
) -> MarketRegime:
    """一次算出四個維度的市場狀態。"""
    correlation = (
        cross_sectional_correlation(returns_matrix) if returns_matrix is not None else float("nan")
    )
    return MarketRegime(
        trend=classify_trend(prices),
        volatility=classify_volatility(returns),
        liquidity=classify_liquidity(volumes),
        cross_sectional_correlation=0.0 if math.isnan(correlation) else correlation,
    )
