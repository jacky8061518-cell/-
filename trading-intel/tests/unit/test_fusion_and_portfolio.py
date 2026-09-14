"""訊號融合與組合建構（SPEC 6）。

核心驗證：**禁止簡單平均**。相關性高的訊號合計權重必須接近單一訊號，
而不是每個各拿一份。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.core.ids import make_entity_id
from trading_intel.core.settings import PortfolioSettings, RiskLimits
from trading_intel.models.fusion import (
    SignalInput,
    correlation_from_covariance,
    fuse,
    half_life_decay,
    inverse_variance_weights,
    ledoit_wolf_shrinkage,
    regime_multiplier,
)
from trading_intel.models.regime import (
    LiquidityState,
    MarketRegime,
    TrendState,
    VolatilityState,
)
from trading_intel.portfolio.construction import (
    build_portfolio,
    volatility_target_scalar,
)

A = make_entity_id(Market.TW, "1111")
B = make_entity_id(Market.TW, "2222")
C = make_entity_id(Market.TW, "3333")
T0 = datetime(2024, 3, 19, tzinfo=UTC)
SEED = 20240319

LIMITS = RiskLimits(
    max_position_weight=0.05,
    max_sector_weight=0.25,
    max_gross_exposure=1.0,
    max_net_exposure=0.6,
    drawdown_derisk=0.08,
    drawdown_flatten=0.15,
    adv_participation_cap=0.05,
    top5_concentration_cap=0.4,
)

PORTFOLIO = PortfolioSettings(
    target_annual_volatility=0.15,
    max_leverage_from_vol_target=1.0,
    turnover_penalty=0.001,
    min_position_weight=0.005,
)

NORMAL_REGIME = MarketRegime(
    trend=TrendState.RANGING,
    volatility=VolatilityState.NORMAL,
    liquidity=LiquidityState.NORMAL,
    cross_sectional_correlation=0.3,
)

EXTREME_REGIME = MarketRegime(
    trend=TrendState.TRENDING_DOWN,
    volatility=VolatilityState.EXTREME,
    liquidity=LiquidityState.STRESSED,
    cross_sectional_correlation=0.9,
)


# --- Ledoit-Wolf 收縮 ------------------------------------------------------


def test_shrinkage_is_between_zero_and_one() -> None:
    rng = np.random.default_rng(SEED)
    _covariance, shrinkage = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (200, 5)))
    assert 0.0 <= shrinkage <= 1.0


def correlated_history(n_samples: int, *, seed: int = SEED) -> np.ndarray:
    """有共同因子的報酬矩陣。純獨立資料的收縮目標本來就正確，
    會一路收縮到 1.0，看不出樣本多寡的差異。"""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.015, n_samples)
    return np.column_stack(
        [
            factor * loading + rng.normal(0, 0.005, n_samples)
            for loading in (1.0, 0.8, 0.6, 0.4, 0.2)
        ]
    )


def test_shrinkage_is_higher_when_samples_are_scarce() -> None:
    """訊號數接近樣本數時，樣本協方差極不穩定，收縮強度應提高。"""
    _c1, plenty = ledoit_wolf_shrinkage(correlated_history(2000))
    _c2, scarce = ledoit_wolf_shrinkage(correlated_history(12))
    assert scarce > plenty


def test_shrinkage_is_modest_with_plenty_of_data() -> None:
    """樣本充足時不該過度收縮——那會把真實的相關結構也抹掉。"""
    _covariance, shrinkage = ledoit_wolf_shrinkage(correlated_history(2000))
    assert shrinkage < 0.5


def test_shrunk_matrix_is_symmetric() -> None:
    rng = np.random.default_rng(SEED)
    covariance, _ = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (100, 4)))
    assert np.allclose(covariance, covariance.T)


def test_shrinkage_rejects_bad_shapes() -> None:
    with pytest.raises(Exception, match="二維"):
        ledoit_wolf_shrinkage([1.0, 2.0, 3.0])
    with pytest.raises(Exception, match="樣本不足"):
        ledoit_wolf_shrinkage(np.zeros((1, 3)))


def test_correlation_diagonal_is_one() -> None:
    rng = np.random.default_rng(SEED)
    covariance, _ = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (200, 3)))
    correlation = correlation_from_covariance(covariance)
    assert np.allclose(np.diag(correlation), 1.0)


# --- 禁止簡單平均 ----------------------------------------------------------


def test_correlated_signals_do_not_each_get_a_full_share() -> None:
    """SPEC 6 的核心：訊號相關性高時，等權等於重壓單一因子。

    三個訊號中有兩個幾乎完全相關。那兩個合計拿到的權重，
    必須明顯少於「各拿三分之一」的等權結果。
    """
    rng = np.random.default_rng(SEED)
    base = rng.normal(0, 0.01, 500)
    independent = rng.normal(0, 0.01, 500)
    history = np.column_stack([base, base + rng.normal(0, 0.0005, 500), independent])

    covariance, _ = ledoit_wolf_shrinkage(history)
    weights = inverse_variance_weights(covariance)

    correlated_share = weights[0] + weights[1]
    assert correlated_share < 2 / 3, (
        f"兩個高度相關的訊號合計拿到 {correlated_share:.3f}，等權會是 0.667——沒有被稀釋代表融合失效"
    )
    assert weights[2] > 1 / 3


def test_weights_sum_to_one() -> None:
    rng = np.random.default_rng(SEED)
    covariance, _ = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (300, 4)))
    assert inverse_variance_weights(covariance).sum() == pytest.approx(1.0)


def test_lower_variance_gets_more_weight() -> None:
    rng = np.random.default_rng(SEED)
    history = np.column_stack([rng.normal(0, 0.002, 500), rng.normal(0, 0.05, 500)])
    covariance, _ = ledoit_wolf_shrinkage(history)
    weights = inverse_variance_weights(covariance)
    assert weights[0] > weights[1]


def test_empty_covariance_yields_no_weights() -> None:
    assert inverse_variance_weights(np.zeros((0, 0))).size == 0


# --- 半衰期衰減 ------------------------------------------------------------


def test_decay_starts_at_one() -> None:
    assert half_life_decay(T0, T0, half_life_days=5) == pytest.approx(1.0)


def test_decay_is_half_at_the_half_life() -> None:
    assert half_life_decay(T0, T0 + timedelta(days=5), half_life_days=5) == pytest.approx(0.5)


def test_decay_reaches_zero_at_twice_the_half_life() -> None:
    """SPEC 6：超過半衰期線性衰減至零，不得無限期持有。

    純指數衰減永遠不會到零，而「永遠留著一點點」就是無限期持有。
    """
    assert half_life_decay(T0, T0 + timedelta(days=10), half_life_days=5) == 0.0
    assert half_life_decay(T0, T0 + timedelta(days=30), half_life_days=5) == 0.0


def test_decay_is_monotonic() -> None:
    values = [
        half_life_decay(T0, T0 + timedelta(days=day), half_life_days=5) for day in range(0, 12)
    ]
    assert values == sorted(values, reverse=True)


def test_future_signal_has_no_weight() -> None:
    assert half_life_decay(T0 + timedelta(days=1), T0, half_life_days=5) == 0.0


def test_non_positive_half_life_is_rejected() -> None:
    with pytest.raises(Exception, match="半衰期必須為正數"):
        half_life_decay(T0, T0, half_life_days=0)


# --- regime 條件加權 -------------------------------------------------------


def test_extreme_volatility_suppresses_momentum() -> None:
    assert regime_multiplier(EXTREME_REGIME, "momentum") < regime_multiplier(
        NORMAL_REGIME, "momentum"
    )


def test_unknown_family_gets_neutral_multiplier() -> None:
    assert regime_multiplier(NORMAL_REGIME, "未知類別") == 1.0


# --- 融合整體 --------------------------------------------------------------


def signal(name: str, family: str, score: float, **kwargs: float) -> SignalInput:
    return SignalInput(
        name=name,
        family=family,
        score=score,
        confidence=kwargs.get("confidence", 1.0),
        signal_time=T0,
        half_life_days=kwargs.get("half_life_days", 10.0),
    )


def test_fusion_of_no_signals_is_zero() -> None:
    result = fuse([], history=None, regime=NORMAL_REGIME, now=T0)
    assert result.combined_score == 0.0


def test_fusion_weights_sum_to_one() -> None:
    signals = [signal("a", "momentum", 0.5), signal("b", "flow", 0.3)]
    result = fuse(signals, history=None, regime=NORMAL_REGIME, now=T0)
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_fusion_score_is_bounded() -> None:
    signals = [signal("a", "momentum", 1.0), signal("b", "flow", 1.0)]
    result = fuse(signals, history=None, regime=NORMAL_REGIME, now=T0)
    assert -1.0 <= result.combined_score <= 1.0


def test_decayed_signals_lose_influence() -> None:
    fresh = fuse([signal("a", "momentum", 1.0)], history=None, regime=NORMAL_REGIME, now=T0)
    stale = fuse(
        [signal("a", "momentum", 1.0)],
        history=None,
        regime=NORMAL_REGIME,
        now=T0 + timedelta(days=30),
    )
    assert fresh.combined_score > 0
    assert stale.combined_score == 0.0
    assert "a" in stale.dropped


def test_low_confidence_reduces_weight() -> None:
    signals = [
        signal("high", "momentum", 0.5, confidence=1.0),
        signal("low", "momentum", 0.5, confidence=0.2),
    ]
    result = fuse(signals, history=None, regime=NORMAL_REGIME, now=T0)
    assert result.weights["high"] > result.weights["low"]


def test_history_column_mismatch_is_rejected() -> None:
    rng = np.random.default_rng(SEED)
    with pytest.raises(Exception, match="欄數與訊號數不符"):
        fuse(
            [signal("a", "momentum", 0.5)],
            history=rng.normal(0, 0.01, (100, 3)),
            regime=NORMAL_REGIME,
            now=T0,
        )


# --- 波動率目標化 ----------------------------------------------------------


def test_high_forecast_volatility_reduces_exposure() -> None:
    """市場波動上升時自動降槓桿。"""
    calm = volatility_target_scalar(0.10, PORTFOLIO)
    stormy = volatility_target_scalar(0.30, PORTFOLIO)
    assert stormy < calm
    assert stormy == pytest.approx(0.5)


def test_scalar_is_capped_at_max_leverage() -> None:
    assert volatility_target_scalar(0.01, PORTFOLIO) == 1.0


def test_zero_volatility_yields_zero_exposure() -> None:
    assert volatility_target_scalar(0.0, PORTFOLIO) == 0.0


# --- 組合建構 --------------------------------------------------------------


def test_empty_scores_yield_empty_portfolio() -> None:
    result = build_portfolio({}, forecast_annual_volatility=0.15, limits=LIMITS, settings=PORTFOLIO)
    assert result.weights == {}


def test_position_cap_is_enforced() -> None:
    result = build_portfolio(
        {A: 10.0, B: 0.1},
        forecast_annual_volatility=0.15,
        limits=LIMITS,
        settings=PORTFOLIO,
    )
    assert all(abs(weight) <= LIMITS.max_position_weight for weight in result.weights.values())
    assert A in result.constrained


def test_sector_cap_is_enforced() -> None:
    sectors = {A: "半導體", B: "半導體", C: "半導體"}
    loose = RiskLimits(**{**LIMITS.model_dump(), "max_position_weight": 0.5})
    result = build_portfolio(
        {A: 1.0, B: 1.0, C: 1.0},
        sectors=sectors,
        forecast_annual_volatility=0.15,
        limits=loose,
        settings=PORTFOLIO,
    )
    total = sum(abs(weight) for weight in result.weights.values())
    assert total <= loose.max_sector_weight + 1e-9


def test_gross_exposure_cap_is_enforced() -> None:
    scores = {make_entity_id(Market.TW, f"{index:04d}"): 1.0 for index in range(1, 60)}
    result = build_portfolio(
        scores, forecast_annual_volatility=0.15, limits=LIMITS, settings=PORTFOLIO
    )
    gross = sum(abs(weight) for weight in result.weights.values())
    assert gross <= LIMITS.max_gross_exposure + 1e-9


def test_small_positions_are_removed() -> None:
    """低於最小權重直接歸零，避免產生無意義的小單。"""
    scores = {make_entity_id(Market.TW, f"{index:04d}"): 1.0 for index in range(1, 500)}
    result = build_portfolio(
        scores, forecast_annual_volatility=0.15, limits=LIMITS, settings=PORTFOLIO
    )
    nonzero = [weight for weight in result.weights.values() if weight != 0]
    assert all(abs(weight) >= PORTFOLIO.min_position_weight for weight in nonzero)


def test_turnover_penalty_pulls_towards_the_previous_portfolio() -> None:
    """沒有換手率懲罰，訊號的微小變動會導致大幅換倉，而證交稅會吃掉價值。"""
    previous = {A: 0.05, B: 0.0}
    heavy = PortfolioSettings(**{**PORTFOLIO.model_dump(), "turnover_penalty": 0.008})
    light = PortfolioSettings(**{**PORTFOLIO.model_dump(), "turnover_penalty": 0.0})
    scores = {A: 0.0, B: 1.0}
    with_penalty = build_portfolio(
        scores,
        previous=previous,
        forecast_annual_volatility=0.15,
        limits=LIMITS,
        settings=heavy,
    )
    without = build_portfolio(
        scores,
        previous=previous,
        forecast_annual_volatility=0.15,
        limits=LIMITS,
        settings=light,
    )
    assert with_penalty.turnover < without.turnover


def test_turnover_is_reported() -> None:
    result = build_portfolio(
        {A: 1.0}, forecast_annual_volatility=0.15, limits=LIMITS, settings=PORTFOLIO
    )
    assert result.turnover > 0


def test_all_zero_scores_yield_no_positions() -> None:
    result = build_portfolio(
        {A: 0.0, B: 0.0},
        forecast_annual_volatility=0.15,
        limits=LIMITS,
        settings=PORTFOLIO,
    )
    assert all(weight == 0.0 for weight in result.weights.values())


def test_negative_scores_produce_short_positions() -> None:
    result = build_portfolio(
        {A: -1.0, B: 1.0},
        forecast_annual_volatility=0.15,
        limits=LIMITS,
        settings=PORTFOLIO,
    )
    assert result.weights[A] < 0
    assert result.weights[B] > 0
