"""Contract, point-in-time, risk and decision tests.

The point-in-time tests are the ones that matter most. Every other bug costs a
bad signal; a look-ahead leak costs a strategy that backtests beautifully and
loses money in production.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_desk.agents import AgentRunner, MarketContext, ScannerAgent
from trading_desk.agents.base import Agent
from trading_desk.blackboard import Blackboard
from trading_desk.contracts import (
    CounterEvidence,
    DataQuality,
    Evidence,
    Quality,
    RiskBudget,
    Severity,
    SignalCard,
    stable_id,
)
from trading_desk.decision import DecisionConfig, DecisionLayer
from trading_desk.desk import DeskConfig, run_desk
from trading_desk.risk import RiskEngine, RiskLimits


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def panel() -> pd.DataFrame:
    """Synthetic price panel: one uptrend, one downtrend, one benchmark."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=400)
    frame = pd.DataFrame(index=dates)
    frame["UP"] = 100 * np.cumprod(1 + rng.normal(0.0012, 0.015, len(dates)))
    frame["DOWN"] = 100 * np.cumprod(1 + rng.normal(-0.0012, 0.015, len(dates)))
    frame["FLAT"] = 100 * np.cumprod(1 + rng.normal(0.0, 0.010, len(dates)))
    frame["BENCH"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(dates)))
    return frame


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {"Name": ["上漲股", "下跌股", "橫盤股", "基準"], "Industry": ["半導體"] * 3 + ["指數"]},
        index=["UP", "DOWN", "FLAT", "BENCH"],
    )


def make_context(panel, metadata, as_of=None, **kwargs) -> MarketContext:
    return MarketContext(
        as_of=pd.Timestamp(as_of) if as_of is not None else panel.index[-1],
        prices=panel,
        metadata=metadata,
        benchmark="BENCH",
        extras={"master": pd.DataFrame()},
        **kwargs,
    )


# --- Point-in-time --------------------------------------------------------


def test_visible_prices_never_include_the_future(panel, metadata):
    cutoff = panel.index[200]
    context = make_context(panel, metadata, as_of=cutoff)
    visible = context.visible_prices()
    assert visible.index.max() == cutoff
    assert len(visible) == 201


def test_agents_cannot_see_past_the_as_of_date(panel, metadata):
    """The same as-of date must produce the same features regardless of what
    data exists after it, which is what makes a replay trustworthy."""
    cutoff = panel.index[300]
    truncated = panel.loc[:cutoff]

    def features_for(prices):
        board = Blackboard(as_of=cutoff.to_pydatetime())
        ScannerAgent().run(make_context(prices, metadata, as_of=cutoff), board)
        return board.feature_frame()

    with_future = features_for(panel)
    without_future = features_for(truncated)
    pd.testing.assert_frame_equal(
        with_future.sort_index(axis=1), without_future.sort_index(axis=1)
    )


def test_flows_are_also_cut_at_the_as_of_date(panel, metadata):
    flows = pd.DataFrame(
        {
            "Date": [panel.index[100], panel.index[300]],
            "Ticker": ["UP", "UP"],
            "Total net shares": [1000, 2000],
        }
    )
    context = make_context(panel, metadata, as_of=panel.index[200], flows=flows)
    assert len(context.visible_flows()) == 1


# --- Contracts ------------------------------------------------------------


def _budget() -> RiskBudget:
    return RiskBudget(2.0, 90.0, "2×ATR", 0.2, 120.0, 1.5, "波動率目標")


def _card(**overrides) -> SignalCard:
    defaults = dict(
        signal_id="sig_1",
        decision_id="dec_1",
        created_at=datetime.now(timezone.utc),
        symbol="UP",
        name="上漲股",
        direction="long",
        conviction=0.7,
        horizon="5-15 個交易日",
        thesis="測試",
        evidence=[Evidence("flow", "測試", 0.3, 1, "flow")],
        counter_evidence=[],
        regime_context="中性",
        risk=_budget(),
        invalidation="跌破 90",
        data_quality=DataQuality(True),
        model_provenance={},
    )
    defaults.update(overrides)
    return SignalCard(**defaults)


def test_a_signal_without_evidence_cannot_be_constructed():
    with pytest.raises(ValueError, match="evidence"):
        _card(evidence=[])


def test_a_signal_without_an_invalidation_condition_cannot_be_constructed():
    with pytest.raises(ValueError, match="invalidation"):
        _card(invalidation="")


def test_conviction_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        _card(conviction=1.4)


def test_data_quality_penalty_compounds_with_each_problem():
    clean = DataQuality(True)
    stale = DataQuality(False, stale_sources=("news",))
    conflicted = DataQuality(False, stale_sources=("news",), conflicts=("矛盾",))
    assert clean.penalty == 1.0
    assert stale.penalty < clean.penalty
    assert conflicted.penalty < stale.penalty


def test_stable_id_is_deterministic():
    assert stable_id("dec", "run", "UP") == stable_id("dec", "run", "UP")
    assert stable_id("dec", "run", "UP") != stable_id("dec", "run", "DOWN")


def test_blackboard_distinguishes_missing_from_zero():
    board = Blackboard(as_of=datetime.now(timezone.utc))
    board.write_feature("UP", "flow_score", float("nan"), "flow", Quality.MISSING)
    board.write_feature("UP", "momentum_score", 0.0, "scanner")
    view = board.view("UP")
    assert not view.is_usable("flow_score")
    assert view.is_usable("momentum_score")


# --- Risk engine ----------------------------------------------------------


def test_low_conviction_is_refused_outright():
    budget = RiskEngine().size(
        conviction=0.50, annual_volatility=0.30, last_price=100.0, direction="long"
    )
    assert not budget.is_tradeable
    assert "信心" in budget.binding_constraint


def test_higher_volatility_gets_a_smaller_position():
    engine = RiskEngine()
    calm = engine.size(conviction=0.75, annual_volatility=0.20, last_price=100.0, direction="long")
    wild = engine.size(conviction=0.75, annual_volatility=0.60, last_price=100.0, direction="long")
    assert wild.suggested_weight_pct < calm.suggested_weight_pct


def test_higher_conviction_gets_a_larger_position():
    engine = RiskEngine()
    weak = engine.size(conviction=0.60, annual_volatility=0.35, last_price=100.0, direction="long")
    strong = engine.size(conviction=0.80, annual_volatility=0.35, last_price=100.0, direction="long")
    assert strong.suggested_weight_pct > weak.suggested_weight_pct


def test_position_budget_scales_down_with_the_position_count():
    """Sizing every name to the full portfolio target would over-risk the book."""
    # Caps are lifted so the volatility budget is the only thing that binds.
    wide = dict(max_weight_per_name=1.0, max_weight_per_cluster=1.0)
    few = RiskEngine(RiskLimits(max_positions=4, **wide))
    many = RiskEngine(RiskLimits(max_positions=25, **wide))
    arguments = dict(conviction=0.8, annual_volatility=0.30, last_price=100.0, direction="long")
    assert few.size(**arguments).suggested_weight_pct > many.size(**arguments).suggested_weight_pct


def test_a_tripped_breaker_blocks_every_new_position():
    engine = RiskEngine()
    breaker = engine.evaluate_breakers(
        daily_pnl_pct=-5.0, rolling_5d_pnl_pct=-1.0, drawdown_pct=-3.0
    )
    assert breaker.tripped
    budget = engine.size(
        conviction=0.95, annual_volatility=0.20, last_price=100.0,
        direction="long", breaker=breaker,
    )
    assert not budget.is_tradeable


def test_breakers_report_the_most_severe_condition_first():
    engine = RiskEngine()
    breaker = engine.evaluate_breakers(
        daily_pnl_pct=-2.5, rolling_5d_pnl_pct=-7.0, drawdown_pct=-25.0
    )
    assert breaker.level == "P0"
    assert "回撤" in breaker.reason


def test_cluster_limit_binds_before_the_gross_limit():
    engine = RiskEngine(RiskLimits(max_weight_per_cluster=0.10, max_weight_per_name=0.05))
    proposals = [(name, "long", RiskBudget(5.0, 90.0, "2×ATR", 0.2, 120.0, 1.5, "上限")) for name in "ABCD"]
    clusters = {name: "同一族群" for name in "ABCD"}
    approved, notes = engine.apply_portfolio_limits(proposals, clusters)
    assert sum(approved.values()) <= 10.0 + 1e-9
    assert len(approved) < 4
    assert notes


def test_full_kelly_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Kelly"):
        RiskLimits(kelly_fraction=1.0)


def test_a_name_cannot_be_allowed_more_than_its_cluster():
    with pytest.raises(ValueError, match="cluster"):
        RiskLimits(max_weight_per_name=0.30, max_weight_per_cluster=0.20)


# --- Decision layer -------------------------------------------------------


def _board_with(evidence: list[Evidence], counter=None) -> Blackboard:
    board = Blackboard(as_of=datetime.now(timezone.utc))
    board.write_feature("UP", "vol_ewma", 0.30, "scanner")
    board.write_feature("UP", "last_close", 100.0, "scanner")
    for item in evidence:
        board.add_evidence("UP", item)
    for item in counter or []:
        board.add_counter_evidence("UP", item)
    board.view("UP").notes["corroboration_score"] = 1.1
    return board


def test_a_single_evidence_kind_never_becomes_a_signal():
    board = _board_with([Evidence("flow", "只有資金流", 0.35, 1, "flow")])
    signals, _ = DecisionLayer().build(board, run_id="run_1")
    assert signals == []


def test_anomaly_plus_news_without_confirmation_is_not_a_signal():
    """A story with no money or trend behind it is intel, not a position."""
    board = _board_with(
        [
            Evidence("anomaly", "跳空", 0.20, 1, "anomaly"),
            Evidence("sentiment", "利多新聞", 0.20, 1, "sentiment"),
        ]
    )
    signals, _ = DecisionLayer().build(board, run_id="run_1")
    assert signals == []


def test_two_agreeing_families_can_produce_a_signal():
    board = _board_with(
        [
            Evidence("flow", "法人買超", 0.35, 1, "flow"),
            Evidence("momentum", "動能轉強", 0.25, 1, "scanner"),
        ]
    )
    signals, _ = DecisionLayer().build(board, run_id="run_1")
    assert len(signals) == 1
    assert signals[0].direction == "long"
    assert signals[0].invalidation


def test_conviction_never_reaches_certainty():
    board = _board_with(
        [
            Evidence("flow", "極強", 5.0, 1, "flow"),
            Evidence("momentum", "極強", 5.0, 1, "scanner"),
            Evidence("sentiment", "極強", 5.0, 1, "sentiment"),
            Evidence("anomaly", "極強", 5.0, 1, "anomaly"),
        ]
    )
    board.view("UP").notes["corroboration_score"] = 1.3
    signals, _ = DecisionLayer().build(board, run_id="run_1")
    assert signals[0].conviction <= 0.90


def test_counter_evidence_lowers_conviction():
    evidence = [
        Evidence("flow", "法人買超", 0.35, 1, "flow"),
        Evidence("momentum", "動能轉強", 0.25, 1, "scanner"),
    ]
    clean = DecisionLayer().build(_board_with(evidence), run_id="r")[0][0]
    penalised = DecisionLayer().build(
        _board_with(evidence, [CounterEvidence("風險", "high", "anomaly")]), run_id="r"
    )[0][0]
    assert penalised.conviction < clean.conviction


def test_signal_budget_suppresses_the_overflow():
    board = Blackboard(as_of=datetime.now(timezone.utc))
    for index in range(8):
        symbol = f"S{index}"
        board.write_feature(symbol, "vol_ewma", 0.30, "scanner")
        board.write_feature(symbol, "last_close", 100.0, "scanner")
        board.add_evidence(symbol, Evidence("flow", "買超", 0.35, 1, "flow"))
        board.add_evidence(symbol, Evidence("momentum", "轉強", 0.25, 1, "scanner"))
    signals, suppressed = DecisionLayer(DecisionConfig(signal_budget=3)).build(board, run_id="r")
    assert len(signals) == 3
    assert suppressed == 5


# --- End to end -----------------------------------------------------------


def test_pipeline_runs_end_to_end_and_stays_consistent(panel, metadata):
    state = run_desk(make_context(panel, metadata), DeskConfig())
    assert state.run_id
    assert state.regime is not None
    assert state.elapsed_seconds >= 0
    for signal in state.signals:
        assert signal.evidence and signal.invalidation
        assert 0 <= signal.conviction <= 0.90
        # The routed alert band and the card must agree, always.
        alert = next(a for a in state.alerts if a.signal_id == signal.signal_id)
        assert alert.severity == signal.severity


def test_the_same_as_of_produces_the_same_run_id(panel, metadata):
    first = run_desk(make_context(panel, metadata), DeskConfig())
    second = run_desk(make_context(panel, metadata), DeskConfig())
    assert first.run_id == second.run_id
    assert [s.decision_id for s in first.signals] == [s.decision_id for s in second.signals]


def test_a_failing_agent_degrades_the_run_instead_of_killing_it(panel, metadata):
    class Broken(Agent):
        name = "broken"
        version = "1.0"

        def run(self, context, board):
            raise RuntimeError("上游掛了")

    board = Blackboard(as_of=panel.index[-1].to_pydatetime())
    runs = AgentRunner([Broken(), ScannerAgent()]).run_all(make_context(panel, metadata), board)
    assert runs[0].status == "failed"
    assert runs[1].status == "ok", "one broken agent must not stop the ones after it"
    assert board.degradations
