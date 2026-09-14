"""市場狀態分類與衝突仲裁。兩者都是純程式碼，不使用 LLM（SPEC 4.2）。"""

from __future__ import annotations

import numpy as np
import pytest

from trading_intel.agents.orchestrator import (
    Arbitration,
    HumanQueue,
    Opinion,
    Resolution,
    arbitrate,
    consistency_gate,
)
from trading_intel.core.enums import DecisionLevel, Direction, Market
from trading_intel.core.ids import make_entity_id
from trading_intel.models.regime import (
    LiquidityState,
    MarketRegime,
    RegimeClassifier,
    TrendState,
    VolatilityState,
    classify,
    classify_liquidity,
    classify_trend,
    classify_volatility,
    cross_sectional_correlation,
)

TSMC = make_entity_id(Market.TW, "2330")
SEED = 20240319


# --- 趨勢 ------------------------------------------------------------------


def test_rising_series_is_trending_up() -> None:
    prices = np.linspace(100, 150, 100)
    assert classify_trend(prices) is TrendState.TRENDING_UP


def test_falling_series_is_trending_down() -> None:
    prices = np.linspace(150, 100, 100)
    assert classify_trend(prices) is TrendState.TRENDING_DOWN


def test_flat_series_is_ranging() -> None:
    rng = np.random.default_rng(SEED)
    prices = 100 + rng.normal(0, 0.3, 200)
    assert classify_trend(prices) is TrendState.RANGING


def test_tiny_deviation_is_not_a_trend() -> None:
    """沒有中性區間，分類會每天跳動。"""
    prices = np.concatenate([np.full(60, 100.0), np.full(20, 100.5)])
    assert classify_trend(prices) is TrendState.RANGING


def test_short_history_defaults_to_ranging() -> None:
    assert classify_trend([100.0, 101.0]) is TrendState.RANGING


# --- 波動率 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [
        # 日波動率換算年化約為 sigma * sqrt(252)。EWMA 以 span=20 加權，
        # 有效樣本少於全序列，因此估計值略低於理論值——這是估計量的正常性質，
        # 所以測試取穩穩落在各區間內的參數，而非貼著門檻。
        (0.003, VolatilityState.LOW),
        (0.011, VolatilityState.NORMAL),
        (0.022, VolatilityState.HIGH),
        (0.04, VolatilityState.EXTREME),
    ],
)
def test_volatility_layers(sigma: float, expected: VolatilityState) -> None:
    rng = np.random.default_rng(SEED)
    assert classify_volatility(rng.normal(0, sigma, 500)) is expected


def test_volatility_thresholds_are_ordered() -> None:
    """四個層級必須是單調的：波動越大，層級越高。"""
    rng = np.random.default_rng(SEED)
    order = [
        VolatilityState.LOW,
        VolatilityState.NORMAL,
        VolatilityState.HIGH,
        VolatilityState.EXTREME,
    ]
    observed = [
        classify_volatility(rng.normal(0, sigma, 500)) for sigma in (0.003, 0.011, 0.022, 0.04)
    ]
    assert [order.index(item) for item in observed] == [0, 1, 2, 3]


def test_volatility_of_too_short_series_is_normal() -> None:
    assert classify_volatility([0.01]) is VolatilityState.NORMAL


# --- 流動性 ----------------------------------------------------------------


def test_shrinking_volume_is_stressed() -> None:
    volumes = np.concatenate([np.full(60, 1000.0), np.full(5, 300.0)])
    assert classify_liquidity(volumes) is LiquidityState.STRESSED


def test_stable_volume_is_normal() -> None:
    assert classify_liquidity(np.full(100, 1000.0)) is LiquidityState.NORMAL


# --- 橫斷面相關性 ----------------------------------------------------------


def test_correlated_assets_have_high_correlation() -> None:
    """市場恐慌時個股齊漲齊跌，分散效果消失。"""
    rng = np.random.default_rng(SEED)
    common = rng.normal(0, 0.02, 200)
    matrix = np.column_stack([common + rng.normal(0, 0.002, 200) for _ in range(5)])
    assert cross_sectional_correlation(matrix) > 0.9


def test_independent_assets_have_low_correlation() -> None:
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.02, (500, 5))
    assert abs(cross_sectional_correlation(matrix)) < 0.2


def test_correlation_of_degenerate_input_is_nan() -> None:
    assert np.isnan(cross_sectional_correlation(np.ones((10, 1))))
    assert np.isnan(cross_sectional_correlation(np.ones((10, 3))))


# --- 組合狀態 --------------------------------------------------------------


def test_risk_off_on_extreme_volatility() -> None:
    regime = MarketRegime(
        trend=TrendState.RANGING,
        volatility=VolatilityState.EXTREME,
        liquidity=LiquidityState.NORMAL,
        cross_sectional_correlation=0.3,
    )
    assert regime.is_risk_off


def test_risk_off_on_high_correlation() -> None:
    regime = MarketRegime(
        trend=TrendState.TRENDING_UP,
        volatility=VolatilityState.NORMAL,
        liquidity=LiquidityState.NORMAL,
        cross_sectional_correlation=0.85,
    )
    assert regime.is_risk_off


def test_calm_market_is_not_risk_off() -> None:
    regime = MarketRegime(
        trend=TrendState.TRENDING_UP,
        volatility=VolatilityState.LOW,
        liquidity=LiquidityState.NORMAL,
        cross_sectional_correlation=0.2,
    )
    assert not regime.is_risk_off


def test_classify_combines_all_dimensions() -> None:
    rng = np.random.default_rng(SEED)
    prices = np.linspace(100, 140, 200)
    returns = rng.normal(0, 0.01, 200)
    volumes = np.full(200, 1000.0)
    regime = classify(prices, returns, volumes)
    assert regime.trend is TrendState.TRENDING_UP
    assert "/" in regime.label


# --- 遲滯：避免每日跳動 ----------------------------------------------------


def calm() -> MarketRegime:
    return MarketRegime(
        trend=TrendState.RANGING,
        volatility=VolatilityState.NORMAL,
        liquidity=LiquidityState.NORMAL,
        cross_sectional_correlation=0.3,
    )


def stormy() -> MarketRegime:
    return MarketRegime(
        trend=TrendState.TRENDING_DOWN,
        volatility=VolatilityState.HIGH,
        liquidity=LiquidityState.STRESSED,
        cross_sectional_correlation=0.8,
    )


def test_first_observation_is_adopted_immediately() -> None:
    classifier = RegimeClassifier()
    assert classifier.update(calm()).label == calm().label


def test_single_divergent_observation_does_not_switch() -> None:
    """SPEC：輸出必須離散且穩定，避免每日跳動。"""
    classifier = RegimeClassifier(min_duration=5)
    classifier.update(calm())
    assert classifier.update(stormy()).label == calm().label


def test_sustained_change_eventually_switches() -> None:
    classifier = RegimeClassifier(min_duration=5)
    classifier.update(calm())
    for _ in range(4):
        assert classifier.update(stormy()).label == calm().label
    assert classifier.update(stormy()).label == stormy().label


def test_reverting_resets_the_candidate() -> None:
    """新狀態出現幾次又縮回去，計數必須歸零。"""
    classifier = RegimeClassifier(min_duration=3)
    classifier.update(calm())
    classifier.update(stormy())
    classifier.update(stormy())
    classifier.update(calm())  # 回到原狀態
    classifier.update(stormy())
    assert classifier.current is not None
    assert classifier.current.label == calm().label


# --- 衝突仲裁 --------------------------------------------------------------


def test_agreeing_high_confidence_opinions_are_accepted() -> None:
    opinions = [
        Opinion("A", TSMC, Direction.LONG, 0.85),
        Opinion("B", TSMC, Direction.LONG, 0.8),
    ]
    result = arbitrate(opinions, entity_id=TSMC)
    assert result.resolution is Resolution.ACCEPTED
    assert result.direction is Direction.LONG
    assert result.decision_level is DecisionLevel.CONFIRM


def test_opposing_high_confidence_opinions_escalate_to_a_human() -> None:
    """SPEC 4.2 的核心規則：高信心的相反結論升級人工，**不取平均**。"""
    opinions = [
        Opinion("A", TSMC, Direction.LONG, 0.9),
        Opinion("B", TSMC, Direction.SHORT, 0.85),
    ]
    result = arbitrate(opinions, entity_id=TSMC)
    assert result.resolution is Resolution.ESCALATED
    assert result.needs_human
    assert result.direction is Direction.FLAT
    assert "升級人工佇列" in result.reason


def test_opposing_low_confidence_opinions_are_downgraded() -> None:
    opinions = [
        Opinion("A", TSMC, Direction.LONG, 0.4),
        Opinion("B", TSMC, Direction.SHORT, 0.3),
    ]
    result = arbitrate(opinions, entity_id=TSMC)
    assert result.resolution is Resolution.DOWNGRADED
    assert result.decision_level is DecisionLevel.RESEARCH_ONLY
    assert not result.needs_human


def test_no_opinions_yields_no_opinion() -> None:
    assert arbitrate([], entity_id=TSMC).resolution is Resolution.NO_OPINION


def test_all_flat_yields_no_opinion() -> None:
    opinions = [Opinion("A", TSMC, Direction.FLAT, 0.9)]
    assert arbitrate(opinions, entity_id=TSMC).resolution is Resolution.NO_OPINION


def test_agreeing_but_low_confidence_is_research_only() -> None:
    opinions = [
        Opinion("A", TSMC, Direction.LONG, 0.5),
        Opinion("B", TSMC, Direction.LONG, 0.4),
    ]
    result = arbitrate(opinions, entity_id=TSMC)
    assert result.resolution is Resolution.ACCEPTED
    assert result.decision_level is DecisionLevel.RESEARCH_ONLY


def test_one_high_one_low_opposing_is_not_escalated() -> None:
    """只有一方高信心時不算真正的衝突，降級即可。"""
    opinions = [
        Opinion("A", TSMC, Direction.LONG, 0.9),
        Opinion("B", TSMC, Direction.SHORT, 0.3),
    ]
    assert arbitrate(opinions, entity_id=TSMC).resolution is Resolution.DOWNGRADED


# --- 人工佇列 --------------------------------------------------------------


def test_only_escalations_enter_the_queue() -> None:
    queue = HumanQueue()
    queue.push(arbitrate([Opinion("A", TSMC, Direction.LONG, 0.9)], entity_id=TSMC))
    assert len(queue) == 0

    escalated = arbitrate(
        [Opinion("A", TSMC, Direction.LONG, 0.9), Opinion("B", TSMC, Direction.SHORT, 0.9)],
        entity_id=TSMC,
    )
    queue.push(escalated)
    assert len(queue) == 1


def test_queue_items_do_not_expire_on_their_own() -> None:
    """沒有人看的升級項目應該永遠卡著，而不是安靜地自己通過。"""
    queue = HumanQueue()
    escalated = arbitrate(
        [Opinion("A", TSMC, Direction.LONG, 0.9), Opinion("B", TSMC, Direction.SHORT, 0.9)],
        entity_id=TSMC,
    )
    queue.push(escalated)
    assert len(queue.pending) == 1
    assert isinstance(queue.pending[0], Arbitration)


def test_resolving_removes_from_the_queue() -> None:
    queue = HumanQueue()
    queue.push(
        arbitrate(
            [Opinion("A", TSMC, Direction.LONG, 0.9), Opinion("B", TSMC, Direction.SHORT, 0.9)],
            entity_id=TSMC,
        )
    )
    assert queue.resolve(TSMC) is not None
    assert len(queue) == 0
    assert queue.resolve(TSMC) is None


# --- 一致性閘門 ------------------------------------------------------------


def test_consistency_gate_passes_on_agreement() -> None:
    from trading_intel.agents.schemas import LibrarianVerdict
    from trading_intel.core.types import AgentOutput

    verdict = LibrarianVerdict(verdict="NOVEL", confidence=0.8, reasoning_digest="新的")
    outputs = [
        AgentOutput[LibrarianVerdict](
            payload=verdict,
            confidence=0.8,
            evidence_ids=(),
            reasoning_digest="d",
            model_name="m",
            prompt_hash="h",
        )
        for _ in range(3)
    ]
    passed, score = consistency_gate(outputs)
    assert passed
    assert score == 1.0
