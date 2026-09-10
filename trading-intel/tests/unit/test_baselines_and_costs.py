"""三個基準策略與成本模型。

基準策略的用途是檢查引擎有沒有壞：如果買進持有的績效不合理，
那不是策略的問題，是引擎的問題。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_intel.backtest.baselines import (
    BuyAndHold,
    InstitutionalFlowRanking,
    Momentum12m1,
    equal_weights,
)
from trading_intel.backtest.costs import round_trip_cost_bps, taiwan_trade_cost
from trading_intel.backtest.engine import MarketView, PriceBar, run_backtest
from trading_intel.core.clock import UTC
from trading_intel.core.enums import Market
from trading_intel.core.ids import EntityId, make_entity_id
from trading_intel.core.settings import CostModel, load_settings

A = make_entity_id(Market.TW, "1111")
B = make_entity_id(Market.TW, "2222")
C = make_entity_id(Market.TW, "3333")
ASOF = datetime(2026, 1, 1, tzinfo=UTC)

REAL = CostModel(
    tw_transaction_tax=0.003,
    tw_daytrade_tax=0.0015,
    commission_bps=14.25,
    slippage_bps=5.0,
    impact_coefficient=0.1,
)


def make_bars(entity: EntityId, closes: list[float], start: date) -> list[PriceBar]:
    return [
        PriceBar(
            trade_date=start + timedelta(days=index),
            entity_id=entity,
            open=Decimal(str(close)),
            high=Decimal(str(close)),
            low=Decimal(str(close)),
            close=Decimal(str(close)),
            volume=1_000_000,
            adv_value=Decimal("100000000"),
        )
        for index, close in enumerate(closes)
    ]


# --- 成本模型 --------------------------------------------------------------


def test_buy_has_no_transaction_tax() -> None:
    """證交稅只在賣出時課徵，這個不對稱是台股的關鍵特性。"""
    cost = taiwan_trade_cost(notional=Decimal("1000000"), is_sell=False, costs=REAL)
    assert cost.transaction_tax == Decimal("0")
    assert cost.commission > 0


def test_sell_pays_transaction_tax() -> None:
    cost = taiwan_trade_cost(notional=Decimal("1000000"), is_sell=True, costs=REAL)
    assert cost.transaction_tax == Decimal("3000")  # 100 萬 × 0.3%


def test_day_trade_tax_is_halved() -> None:
    normal = taiwan_trade_cost(notional=Decimal("1000000"), is_sell=True, costs=REAL)
    day = taiwan_trade_cost(
        notional=Decimal("1000000"), is_sell=True, costs=REAL, is_day_trade=True
    )
    assert day.transaction_tax == normal.transaction_tax / 2


def test_commission_matches_the_official_rate() -> None:
    """手續費 0.1425% 的手算驗證。"""
    cost = taiwan_trade_cost(notional=Decimal("1000000"), is_sell=False, costs=REAL)
    assert cost.commission == Decimal("1425.0000")


def test_market_impact_grows_with_participation() -> None:
    """平方根衝擊模型：下單量佔日均量越高，衝擊越大。"""
    small = taiwan_trade_cost(
        notional=Decimal("1000000"),
        is_sell=False,
        costs=REAL,
        average_daily_volume_value=Decimal("1000000000"),
    )
    large = taiwan_trade_cost(
        notional=Decimal("100000000"),
        is_sell=False,
        costs=REAL,
        average_daily_volume_value=Decimal("1000000000"),
    )
    assert large.market_impact > small.market_impact
    # 平方根模型：下單量放大 100 倍，單位成本（每元的衝擊）只放大 10 倍。
    small_unit = float(small.market_impact) / 1_000_000
    large_unit = float(large.market_impact) / 100_000_000
    assert large_unit / small_unit == pytest.approx(10.0, rel=1e-6)


def test_no_adv_means_no_impact() -> None:
    cost = taiwan_trade_cost(notional=Decimal("1000000"), is_sell=False, costs=REAL)
    assert cost.market_impact == Decimal("0")


def test_negative_notional_is_rejected() -> None:
    with pytest.raises(ValueError, match="不得為負數"):
        taiwan_trade_cost(notional=Decimal("-1"), is_sell=False, costs=REAL)


def test_round_trip_cost_is_dominated_by_tax() -> None:
    """一買一賣的來回成本。證交稅 30 bps 是其中最大的一項。"""
    total = round_trip_cost_bps(REAL)
    assert total == pytest.approx(14.25 * 2 + 30.0 + 5.0 * 2)


def test_shipped_config_costs_are_realistic() -> None:
    """簽入的設定檔必須是真實稅率，不是佔位值。"""
    costs = load_settings("dev").costs
    assert costs.tw_transaction_tax == 0.003
    assert costs.tw_daytrade_tax == 0.0015
    # 手續費曾誤填為 1.425（少十倍），此測試防止再犯。
    assert costs.commission_bps > 10


# --- 買進持有 --------------------------------------------------------------


def test_buy_and_hold_splits_weight_evenly() -> None:
    history = {
        A: make_bars(A, [100.0] * 5, date(2024, 1, 1)),
        B: make_bars(B, [50.0] * 5, date(2024, 1, 1)),
    }
    view = MarketView(history, date(2024, 1, 3))
    weights = BuyAndHold((A, B)).target_weights(view)
    assert weights == {A: 0.5, B: 0.5}


def test_buy_and_hold_skips_entities_without_data() -> None:
    history = {A: make_bars(A, [100.0] * 5, date(2024, 1, 1))}
    view = MarketView(history, date(2024, 1, 3))
    weights = BuyAndHold((A, B)).target_weights(view)
    assert weights == {A: 1.0}


def test_buy_and_hold_tracks_the_asset() -> None:
    """引擎的煙霧測試：買進持有的報酬必須接近標的本身的報酬。"""
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(10)]
    closes = [100.0 + index for index in range(10)]
    history = {A: make_bars(A, closes, days[0])}
    free = CostModel(
        tw_transaction_tax=0.0,
        tw_daytrade_tax=0.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        impact_coefficient=0.0,
    )
    result = run_backtest(
        BuyAndHold((A,)), history, days, costs=free, asof=ASOF, initial_equity=Decimal("1000")
    )
    # 第 2 天以 101 建倉，最後一天 109。
    expected = 109 / 101
    assert float(result.final_equity) == pytest.approx(1000 * expected, rel=1e-9)


# --- 12-1 動量 -------------------------------------------------------------


def test_momentum_needs_a_full_lookback() -> None:
    """歷史不足時不出手，而不是用不完整的資料硬算。"""
    history = {A: make_bars(A, [100.0] * 50, date(2020, 1, 1))}
    view = MarketView(history, date(2020, 2, 1))
    assert Momentum12m1((A,), lookback_days=252).target_weights(view) == {}


def test_momentum_picks_the_strongest() -> None:
    start = date(2020, 1, 1)
    rising = [100.0 + index * 0.5 for index in range(300)]
    flat = [100.0] * 300
    falling = [100.0 - index * 0.2 for index in range(300)]
    history = {
        A: make_bars(A, rising, start),
        B: make_bars(B, flat, start),
        C: make_bars(C, falling, start),
    }
    view = MarketView(history, start + timedelta(days=290))
    weights = Momentum12m1((A, B, C), top_n=1).target_weights(view)
    assert weights == {A: 1.0}


def test_momentum_goes_flat_when_nothing_is_rising() -> None:
    """全部動量為負時空手，不硬選最不差的那個。"""
    start = date(2020, 1, 1)
    falling = [100.0 - index * 0.1 for index in range(300)]
    history = {A: make_bars(A, falling, start), B: make_bars(B, falling, start)}
    view = MarketView(history, start + timedelta(days=290))
    assert Momentum12m1((A, B)).target_weights(view) == {}


def test_momentum_skips_the_most_recent_month() -> None:
    """跳過最近一個月是 12-1 動量的關鍵，避開短期反轉效應。"""
    start = date(2020, 1, 1)
    # 長期上漲，但最後 21 天暴跌。跳過機制應讓它仍被選中。
    closes = [100.0 + index * 0.5 for index in range(280)] + [50.0] * 21
    history = {A: make_bars(A, closes, start)}
    view = MarketView(history, start + timedelta(days=300))
    weights = Momentum12m1((A,), top_n=1).target_weights(view)
    assert weights == {A: 1.0}


# --- 三大法人買超排序 ------------------------------------------------------


def test_flow_ranking_picks_the_largest_buyer() -> None:
    history = {
        A: make_bars(A, [100.0] * 30, date(2024, 1, 1)),
        B: make_bars(B, [100.0] * 30, date(2024, 1, 1)),
    }
    flows = {
        (date(2024, 1, 5), A): 1_000_000.0,
        (date(2024, 1, 5), B): 100_000.0,
    }
    view = MarketView(history, date(2024, 1, 10))
    weights = InstitutionalFlowRanking((A, B), flows, top_n=1).target_weights(view)
    assert weights == {A: 1.0}


def test_flow_ranking_ignores_future_flows() -> None:
    """外部訊號源不受引擎的 MarketView 保護，策略必須自己過濾。"""
    history = {A: make_bars(A, [100.0] * 30, date(2024, 1, 1))}
    flows = {(date(2024, 1, 25), A): 5_000_000.0}
    view = MarketView(history, date(2024, 1, 10))
    assert InstitutionalFlowRanking((A,), flows).target_weights(view) == {}


def test_flow_ranking_ignores_net_sellers() -> None:
    history = {A: make_bars(A, [100.0] * 30, date(2024, 1, 1))}
    flows = {(date(2024, 1, 5), A): -1_000_000.0}
    view = MarketView(history, date(2024, 1, 10))
    assert InstitutionalFlowRanking((A,), flows).target_weights(view) == {}


def test_equal_weights_handles_empty() -> None:
    assert equal_weights([]) == {}
    assert equal_weights([A, B]) == {A: 0.5, B: 0.5}
