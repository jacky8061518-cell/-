"""用真實市場資料驗證基準策略績效（SPEC 第 11 節 Phase 2 驗收條件 3）。

這條驗收之前卡住的原因記在 REPORT-PHASE2.md：既有的十四年價格資料有嚴重的
存活者偏誤（Phase 1 已量化），拿它跑選股策略會得到虛高的績效；而網路又連不上
任何資料源，無法補一份乾淨的。

**網路權限開通後，這裡用 Yahoo Finance 現抓的真實資料補上這條驗收。**

但要說清楚這個測試「能證明什麼」與「不能證明什麼」：

- **能**：證明回測引擎的機制是對的——買進持有的報酬確實追蹤標的本身，
  Sharpe／CAGR 這些統計量落在文獻對台股大型權值股的合理範圍。這正是
  SPEC 5.4「三個基準策略作為煙霧測試」的原意：檢查引擎有沒有壞，不是選股。
- **不能**：這裡固定用五檔現在還在市的權值股，不是從某個歷史宇宙裡選股，
  因此**不構成對 Phase 1 存活者偏誤問題的修正**。那個問題仍然只能靠接上
  交易所的上市下市公告解決（ADR 0005），與本測試無關。

網路不可用時整組跳過。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from trading_intel.backtest.baselines import BuyAndHold, Momentum12m1
from trading_intel.backtest.engine import PriceBar, run_backtest
from trading_intel.backtest.validation import sharpe_ratio
from trading_intel.core.clock import utc_now
from trading_intel.core.enums import Market
from trading_intel.core.errors import NetworkAccessDenied
from trading_intel.core.ids import make_entity_id
from trading_intel.core.sandbox import backtest_mode
from trading_intel.core.settings import CostModel
from trading_intel.ingestion.yahoo import fetch_chart, parse_chart

#: 五檔台股大型權值股，涵蓋不同產業。固定名單，不是「選股結果」。
BLUE_CHIPS = ["2330.TW", "2317.TW", "2454.TW", "2412.TW", "1301.TW"]

REAL_COSTS = CostModel(
    tw_transaction_tax=0.003,
    tw_daytrade_tax=0.0015,
    commission_bps=14.25,
    slippage_bps=5.0,
    impact_coefficient=0.1,
)


def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=5).close()
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _network_available(), reason="需要網路存取 Yahoo Finance")


@pytest.fixture(scope="module")
def live_history() -> dict:  # type: ignore[type-arg]
    """抓兩年的真實日線資料。scope=module：五檔股票只抓一次，供本檔多個測試共用。"""
    history = {}
    for ticker in BLUE_CHIPS:
        symbol = ticker.split(".")[0]
        entity_id = make_entity_id(Market.TW, symbol)
        payload, _request = fetch_chart(ticker, range_="2y")
        yahoo_bars = parse_chart(payload, ticker)
        history[entity_id] = [
            PriceBar(
                trade_date=bar.trade_date,
                entity_id=entity_id,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                adv_value=bar.close * Decimal(bar.volume) if bar.volume else None,
            )
            for bar in yahoo_bars
        ]
    return history


def _sessions(history: dict) -> list:  # type: ignore[type-arg]
    all_dates = sorted({bar.trade_date for bars in history.values() for bar in bars})
    return all_dates


def test_fetched_enough_real_history(live_history: dict) -> None:  # type: ignore[type-arg]
    assert all(len(bars) > 200 for bars in live_history.values()), (
        "真實資料量不足以做有意義的回測驗證"
    )


def test_buy_and_hold_matches_hand_calculation_exactly(live_history: dict) -> None:  # type: ignore[type-arg]
    """買進持有的回測報酬，必須與手算的等權平均報酬完全吻合。

    這條測試在開發過程中抓到兩個真實 bug，過程記在
    ``backtest/engine.py`` 與 ``backtest/baselines.py`` 的註解裡：

    1. ``BuyAndHold`` 一度天真地每天回傳同一組等權權重，而引擎當時會把
       權重「沒變」也每天重新換算一次目標股數——價格一波動就被迫交易，
       變成每日再平衡到固定權重，不是買進持有。五檔股票兩年跑出 2420 筆
       成交，正確答案是 5 筆。
    2. 第一次嘗試在策略層修正（記住參考價、用漂移推導權重）沒有解決根本
       問題，因為決策與成交之間有一天落差，策略永遠算不準真正的成交價。

    正確的修法在引擎層：只有目標權重**真的改變**的標的才重新換算股數，
    而不是「今天的權益乘上權重，不管跟昨天是不是同一個數字」。修好之後，
    這裡的手算與回測必須位元層級吻合，不能只是「差不多」。
    """
    entity_ids = tuple(live_history)
    sessions = _sessions(live_history)
    strategy = BuyAndHold(entity_ids)

    result = run_backtest(
        strategy,
        live_history,
        sessions,
        costs=CostModel(
            tw_transaction_tax=0,
            tw_daytrade_tax=0,
            commission_bps=0,
            slippage_bps=0,
            impact_coefficient=0,
        ),
        asof=utc_now(),
        initial_equity=Decimal("1000000"),
    )

    build_day = sessions[1]
    final_day = sessions[-1]
    manual_returns = [
        float(bars_by_date(bars)[final_day].close / bars_by_date(bars)[build_day].close - 1)
        for bars in live_history.values()
    ]
    expected_gross = sum(manual_returns) / len(manual_returns)
    actual_gross = float(result.final_equity / result.initial_equity - 1)

    assert actual_gross == pytest.approx(expected_gross, rel=1e-9), (
        f"回測報酬 {actual_gross:.6%} 與手算報酬 {expected_gross:.6%} 不吻合，"
        "代表引擎的建倉／計價邏輯又壞了"
    )
    # 只有建倉那天真正交易，之後任其漂移——這是買進持有字面上的意思。
    assert len(result.fills) == len(entity_ids)


def bars_by_date(bars: list) -> dict:  # type: ignore[type-arg]
    return {bar.trade_date: bar for bar in bars}


def test_buy_and_hold_with_real_costs_still_trades_only_once(live_history: dict) -> None:  # type: ignore[type-arg]
    """有成本時同樣只交易一次——成本不該讓引擎產生額外的再平衡。"""
    entity_ids = tuple(live_history)
    sessions = _sessions(live_history)
    result = run_backtest(
        BuyAndHold(entity_ids), live_history, sessions, costs=REAL_COSTS, asof=utc_now()
    )
    assert len(result.fills) == len(entity_ids)


def test_buy_and_hold_sharpe_is_in_a_plausible_range(live_history: dict) -> None:  # type: ignore[type-arg]
    """SPEC 驗收：績效數字落在文獻合理範圍。

    台股大型權值股兩年期的年化 Sharpe，合理範圍大致落在 -1.5 到 3 之間；
    落在這之外（例如 Sharpe=50，或報酬為零標準差）代表引擎算錯，而不是
    這五檔股票運氣特別好或特別差。
    """
    entity_ids = tuple(live_history)
    sessions = _sessions(live_history)
    result = run_backtest(
        BuyAndHold(entity_ids),
        live_history,
        sessions,
        costs=REAL_COSTS,
        asof=utc_now(),
    )
    sharpe = sharpe_ratio(result.net_returns, periods_per_year=252)
    assert -1.5 < sharpe < 3.0, f"Sharpe {sharpe} 超出台股大型股的合理範圍，懷疑引擎有誤"


def test_momentum_strategy_runs_on_real_data_without_error(live_history: dict) -> None:  # type: ignore[type-arg]
    """12-1 動量策略在真實資料上完整跑完，不崩潰、不產生 NaN 權益。"""
    import math

    entity_ids = tuple(live_history)
    sessions = _sessions(live_history)
    result = run_backtest(
        Momentum12m1(entity_ids, top_n=2),
        live_history,
        sessions,
        costs=REAL_COSTS,
        asof=utc_now(),
    )
    assert len(result.records) == len(sessions)
    assert math.isfinite(float(result.final_equity))
    assert all(math.isfinite(record.net_return) for record in result.records)


def test_costs_are_charged_on_real_trades(live_history: dict) -> None:  # type: ignore[type-arg]
    """真實資料上，有成本的回測必須比無成本的回測賺得少（除非完全沒有交易）。"""
    entity_ids = tuple(live_history)
    sessions = _sessions(live_history)
    free = CostModel(
        tw_transaction_tax=0,
        tw_daytrade_tax=0,
        commission_bps=0,
        slippage_bps=0,
        impact_coefficient=0,
    )
    with_cost = run_backtest(
        Momentum12m1(entity_ids, top_n=2),
        live_history,
        sessions,
        costs=REAL_COSTS,
        asof=utc_now(),
    )
    without_cost = run_backtest(
        Momentum12m1(entity_ids, top_n=2),
        live_history,
        sessions,
        costs=free,
        asof=utc_now(),
    )
    if with_cost.total_costs > 0:
        assert with_cost.final_equity <= without_cost.final_equity


def test_backtest_mode_blocks_network_even_with_real_tickers() -> None:
    """CLAUDE.md 第 3 條的最終確認：即使資料源真的可用，backtest_mode 內仍不准連網。"""
    import socket

    with backtest_mode(utc_now()), pytest.raises(NetworkAccessDenied):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=5)
