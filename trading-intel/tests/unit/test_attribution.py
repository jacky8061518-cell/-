"""損益歸因：純程式碼，分項恆等式必須成立（SPEC 4.2）。"""

from __future__ import annotations

import numpy as np
import pytest

from trading_intel.surface.attribution import (
    AttributionResult,
    attribute_returns,
    summarize_hit_rate,
)


def test_components_must_sum_to_total() -> None:
    """歸因的核心不變條件：分項總和必須等於總報酬，否則某一項算錯了。"""
    with pytest.raises(ValueError, match="不一致"):
        AttributionResult(
            total_return=0.10,
            beta_return=0.05,
            alpha_return=0.03,
            style_return=0.0,
            cost_drag=-0.01,
            timing_return=0.0,
        )


def test_consistent_components_are_accepted() -> None:
    result = AttributionResult(
        total_return=0.07,
        beta_return=0.05,
        alpha_return=0.03,
        style_return=0.0,
        cost_drag=-0.01,
        timing_return=0.0,
    )
    assert result.total_return == pytest.approx(0.07)


def test_pure_beta_exposure_attributes_entirely_to_beta() -> None:
    """組合報酬完全跟隨基準時，alpha 應接近零。"""
    rng = np.random.default_rng(42)
    benchmark = rng.normal(0.0005, 0.01, 100)
    portfolio = benchmark.copy()  # beta = 1，完全跟隨
    zeros = np.zeros(100)
    result = attribute_returns(portfolio, benchmark, None, None, zeros, portfolio)
    assert result.alpha_return == pytest.approx(0.0, abs=1e-9)
    assert result.beta_return == pytest.approx(result.total_return, rel=1e-6)


def test_costs_must_be_non_positive() -> None:
    """成本項是正值代表把成本算成了收益——這是不允許的方向錯誤。"""
    portfolio = np.array([0.01, 0.02])
    with pytest.raises(ValueError, match="必須為負值或零"):
        attribute_returns(portfolio, portfolio, None, None, np.array([0.001, 0.001]), portfolio)


def test_zero_cost_is_allowed() -> None:
    portfolio = np.array([0.01, 0.02])
    result = attribute_returns(portfolio, portfolio, None, None, np.zeros(2), portfolio)
    assert result.cost_drag == 0.0


def test_timing_gap_is_isolated() -> None:
    """實際執行與理想執行的落差歸入 timing，不該汙染 alpha。"""
    portfolio = np.array([0.01, 0.01])
    ideal = np.array([0.012, 0.012])  # 理想執行應該更好
    result = attribute_returns(portfolio, portfolio, None, None, np.zeros(2), ideal)
    assert result.timing_return < 0
    assert result.timing_return == pytest.approx(-0.004, abs=1e-9)


def test_style_factors_are_optional() -> None:
    """沒有做因子分解時，風格項為零——不假裝分解過。"""
    portfolio = np.array([0.01])
    result = attribute_returns(portfolio, portfolio, None, None, np.zeros(1), portfolio)
    assert result.style_return == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="長度必須一致"):
        attribute_returns([0.01, 0.02], [0.01], None, None, [0.0, 0.0], [0.01, 0.02])


# --- 命中率 ------------------------------------------------------------------


def test_hit_rate_of_perfect_predictions() -> None:
    summary = summarize_hit_rate([1, 1, -1, -1], [0.02, 0.03, -0.01, -0.02])
    assert summary.hit_rate == 1.0
    assert summary.total_signals == 4


def test_hit_rate_ignores_flat_predictions() -> None:
    """方向為 0（無意見）的訊號不計入命中率分母。"""
    summary = summarize_hit_rate([1, 0, -1], [0.01, 0.05, -0.01])
    assert summary.total_signals == 2


def test_hit_rate_separates_correct_and_wrong_returns() -> None:
    summary = summarize_hit_rate([1, 1], [0.05, -0.02])
    assert summary.correct_direction == 1
    assert summary.average_return_when_correct == pytest.approx(0.05)
    assert summary.average_return_when_wrong == pytest.approx(-0.02)


def test_hit_rate_of_empty_input() -> None:
    import math

    summary = summarize_hit_rate([], [])
    assert summary.total_signals == 0
    assert math.isnan(summary.hit_rate)


def test_hit_rate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="長度必須一致"):
        summarize_hit_rate([1, -1], [0.01])
