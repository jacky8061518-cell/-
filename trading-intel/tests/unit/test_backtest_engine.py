"""事件驅動回測引擎的驗收測試。

SPEC 第 11 節 Phase 2 驗收條件：
- 前視偏誤測試：構造明顯用到未來資訊的特徵，引擎必須偵測並報錯
- 對已知答案的合成資料，回測損益與手算結果誤差在浮點誤差內
- 同一策略跑兩次，結果位元層級相同
"""

from __future__ import annotations

import socket
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.backtest.engine import (
    BacktestResult,
    MarketView,
    PriceBar,
    run_backtest,
)
from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.core.errors import LookaheadError, NetworkAccessDenied
from trading_intel.core.ids import EntityId, make_entity_id
from trading_intel.core.settings import CostModel

A = make_entity_id(Market.TW, "1111")
B = make_entity_id(Market.TW, "2222")
ASOF = datetime(2026, 1, 1, tzinfo=UTC)

#: 零成本，用於手算比對。
FREE = CostModel(
    tw_transaction_tax=0.0,
    tw_daytrade_tax=0.0,
    commission_bps=0.0,
    slippage_bps=0.0,
    impact_coefficient=0.0,
)
#: 真實台股成本。
REAL = CostModel(
    tw_transaction_tax=0.003,
    tw_daytrade_tax=0.0015,
    commission_bps=14.25,
    slippage_bps=5.0,
    impact_coefficient=0.1,
)


def sessions(n: int, start: date = date(2024, 1, 1)) -> list[date]:
    return [start + timedelta(days=index) for index in range(n)]


def bars(entity: EntityId, closes: Sequence[str], days: Sequence[date]) -> list[PriceBar]:
    return [
        PriceBar(
            trade_date=day,
            entity_id=entity,
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=1_000_000,
            adv_value=Decimal("100000000"),
        )
        for close, day in zip(closes, days, strict=True)
    ]


class BuyAndHold:
    """全額買進單一標的並持有。"""

    def __init__(self, entity_id: EntityId) -> None:
        self._entity_id = entity_id

    @property
    def name(self) -> str:
        return "買進持有"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:  # noqa: ARG002
        return {self._entity_id: 1.0}


class Flat:
    """完全不持有。"""

    @property
    def name(self) -> str:
        return "空手"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:  # noqa: ARG002
        return {}


class PeekingStrategy:
    """刻意作弊的策略：宣稱要看明天的資料。

    這正是 SPEC 驗收條件要求構造的「明顯用到未來資訊」的情況。
    """

    @property
    def name(self) -> str:
        return "偷看未來"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
        view.assert_visible(view.as_of_date + timedelta(days=1))
        return {A: 1.0}


class SneakyStrategy:
    """試圖從 MarketView 直接翻出未來資料的策略。"""

    @property
    def name(self) -> str:
        return "翻找未來"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
        history = view.history(A)
        # 引擎保證這裡拿不到未來的 bar。
        assert all(bar.trade_date <= view.as_of_date for bar in history)
        return {A: 1.0}


# --- 驗收條件一：前視偏誤必須被偵測並報錯 ---------------------------------


def test_peeking_strategy_is_detected() -> None:
    days = sessions(5)
    history = {A: bars(A, ["100", "101", "102", "103", "104"], days)}
    with pytest.raises(LookaheadError) as excinfo:
        run_backtest(PeekingStrategy(), history, days, costs=FREE, asof=ASOF)
    assert "尚未發生的資料" in str(excinfo.value)


def test_market_view_never_exposes_future_bars() -> None:
    """引擎的結構性保證：策略拿不到未來，不是因為它有禮貌。"""
    days = sessions(5)
    history = {A: bars(A, ["100", "101", "102", "103", "104"], days)}
    view = MarketView(history, days[2])
    visible = view.history(A)
    assert len(visible) == 3
    assert max(bar.trade_date for bar in visible) == days[2]


def test_sneaky_strategy_finds_nothing() -> None:
    days = sessions(5)
    history = {A: bars(A, ["100", "101", "102", "103", "104"], days)}
    result = run_backtest(SneakyStrategy(), history, days, costs=FREE, asof=ASOF)
    assert len(result.records) == 5


def test_assert_visible_allows_the_present() -> None:
    view = MarketView({}, date(2024, 1, 3))
    view.assert_visible(date(2024, 1, 3))
    view.assert_visible(date(2024, 1, 2))
    with pytest.raises(LookaheadError):
        view.assert_visible(date(2024, 1, 4))


# --- 驗收條件二：合成資料與手算相符 ---------------------------------------


def test_buy_and_hold_matches_hand_calculation() -> None:
    """價格 100 → 110，零成本，全額持有：權益必須剛好 +10%。

    第 1 天不交易（沒有「昨天」），第 2 天以收盤價 105 建倉，
    因此實際持有期間是 105 → 110，報酬為 110/105 - 1。
    """
    days = sessions(3)
    history = {A: bars(A, ["100", "105", "110"], days)}
    result = run_backtest(
        BuyAndHold(A), history, days, costs=FREE, asof=ASOF, initial_equity=Decimal("1000")
    )
    # 第 2 天以 105 建倉，第 3 天 110 收盤。
    expected_final = Decimal("1000") * Decimal("110") / Decimal("105")
    assert float(result.final_equity) == pytest.approx(float(expected_final), rel=1e-9)


def test_flat_strategy_never_changes_equity() -> None:
    days = sessions(5)
    history = {A: bars(A, ["100", "150", "50", "200", "10"], days)}
    result = run_backtest(
        Flat(), history, days, costs=REAL, asof=ASOF, initial_equity=Decimal("1000")
    )
    assert result.final_equity == Decimal("1000")
    assert result.total_costs == Decimal("0")
    assert result.fills == []


def test_costs_reduce_equity() -> None:
    """成本內建於回測：同一策略在有成本時必須賺得比較少。"""
    days = sessions(4)
    history = {A: bars(A, ["100", "100", "100", "100"], days)}
    free = run_backtest(
        BuyAndHold(A), history, days, costs=FREE, asof=ASOF, initial_equity=Decimal("1000")
    )
    real = run_backtest(
        BuyAndHold(A), history, days, costs=REAL, asof=ASOF, initial_equity=Decimal("1000")
    )
    assert free.final_equity == Decimal("1000")
    assert real.final_equity < free.final_equity
    assert real.total_costs > 0


def test_first_session_does_not_trade() -> None:
    """第一天沒有「昨天」，因此不做決策。"""
    days = sessions(3)
    history = {A: bars(A, ["100", "101", "102"], days)}
    result = run_backtest(BuyAndHold(A), history, days, costs=FREE, asof=ASOF)
    assert result.records[0].turnover == 0.0
    assert all(fill.trade_date != days[0] for fill in result.fills)


def test_decision_uses_yesterday_not_today() -> None:
    """策略看到的最後一根 bar 必須是昨天的。"""
    days = sessions(4)
    history = {A: bars(A, ["100", "101", "102", "103"], days)}
    seen: list[date] = []

    class Recorder:
        @property
        def name(self) -> str:
            return "記錄器"

        def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
            seen.append(view.as_of_date)
            return {}

    run_backtest(Recorder(), history, days, costs=FREE, asof=ASOF)
    # 第 2、3、4 個交易日各決策一次，看到的分別是第 1、2、3 天。
    assert seen == days[:3]


# --- 驗收條件三：可重現 ----------------------------------------------------


def test_running_twice_yields_identical_results() -> None:
    """SPEC 驗收：同一策略跑兩次，結果位元層級相同。"""
    days = sessions(20)
    closes = [str(100 + (index * 7) % 23) for index in range(20)]
    history = {A: bars(A, closes, days)}

    def digest(result: BacktestResult) -> list[tuple[str, str, str]]:
        return [
            (record.trade_date.isoformat(), str(record.equity), str(record.total_cost))
            for record in result.records
        ]

    first = run_backtest(BuyAndHold(A), history, days, costs=REAL, asof=ASOF)
    second = run_backtest(BuyAndHold(A), history, days, costs=REAL, asof=ASOF)
    assert digest(first) == digest(second)
    assert first.final_equity == second.final_equity


def test_decimal_arithmetic_is_exact() -> None:
    """權益以 Decimal 計算，不得出現浮點誤差。"""
    days = sessions(3)
    history = {A: bars(A, ["100", "100", "100"], days)}
    result = run_backtest(
        BuyAndHold(A), history, days, costs=FREE, asof=ASOF, initial_equity=Decimal("1000")
    )
    assert isinstance(result.final_equity, Decimal)
    assert result.final_equity == Decimal("1000")


# --- 回測期間禁止網路 ------------------------------------------------------


def test_network_is_blocked_during_the_backtest() -> None:
    """CLAUDE.md 第 3 條：回測執行期間禁止任何網路呼叫。"""
    days = sessions(3)
    history = {A: bars(A, ["100", "101", "102"], days)}

    class NetworkingStrategy:
        @property
        def name(self) -> str:
            return "偷連網"

        def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:  # noqa: ARG002
            socket.create_connection(("example.com", 80), timeout=1)
            return {}

    with pytest.raises(NetworkAccessDenied):
        run_backtest(NetworkingStrategy(), history, days, costs=FREE, asof=ASOF)


def test_socket_is_restored_after_the_backtest() -> None:
    days = sessions(3)
    history = {A: bars(A, ["100", "101", "102"], days)}
    original = socket.socket
    run_backtest(Flat(), history, days, costs=FREE, asof=ASOF)
    assert socket.socket is original


# --- 宇宙限制與權重驗證 ----------------------------------------------------


def test_untradable_entity_is_not_bought() -> None:
    """不可交易的標的不得建倉——這是品質閘門與回測的接點。"""
    days = sessions(4)
    history = {A: bars(A, ["100", "101", "102", "103"], days)}
    result = run_backtest(
        BuyAndHold(A),
        history,
        days,
        costs=FREE,
        asof=ASOF,
        universe_provider=lambda _day: frozenset(),
    )
    assert result.fills == []


def test_leverage_is_rejected() -> None:
    days = sessions(3)
    history = {A: bars(A, ["100", "101", "102"], days)}

    class Levered:
        @property
        def name(self) -> str:
            return "槓桿"

        def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:  # noqa: ARG002
            return {A: 1.5}

    with pytest.raises(ValueError, match="超過 1"):
        run_backtest(Levered(), history, days, costs=FREE, asof=ASOF)


def test_two_asset_portfolio_splits_weight() -> None:
    days = sessions(4)
    history = {
        A: bars(A, ["100", "100", "100", "100"], days),
        B: bars(B, ["50", "50", "50", "50"], days),
    }

    class Half:
        @property
        def name(self) -> str:
            return "各半"

        def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:  # noqa: ARG002
            return {A: 0.5, B: 0.5}

    result = run_backtest(
        Half(), history, days, costs=FREE, asof=ASOF, initial_equity=Decimal("1000")
    )
    assert result.final_equity == Decimal("1000")
    assert result.records[-1].positions == 2


def test_result_exposes_returns_and_turnover() -> None:
    days = sessions(5)
    history = {A: bars(A, ["100", "102", "104", "106", "108"], days)}
    result = run_backtest(BuyAndHold(A), history, days, costs=REAL, asof=ASOF)
    assert len(result.net_returns) == 5
    assert len(result.gross_returns) == 5
    assert result.average_turnover > 0
    # 有成本時淨報酬必須低於毛報酬。
    assert sum(result.net_returns) < sum(result.gross_returns)


def test_empty_session_list_yields_empty_result() -> None:
    result = run_backtest(Flat(), {}, [], costs=FREE, asof=ASOF)
    assert result.records == []
    assert result.final_equity == Decimal("1000000")
