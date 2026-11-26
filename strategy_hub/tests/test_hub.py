"""
tests/test_hub.py
Strategy Hub 的核心不變條件：時間紀律（不偷看未來）、成本不對稱、配置上限、
體制標籤與即時版一致、風控硬規則永遠成立。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hub import costs
from hub.backtest import run_backtest, slot_positions
from hub.data import REPO_ROOT, monthly_universe
from hub.ensemble import MAX_SLEEVE_WEIGHT, cap_and_redistribute, run_ensemble
from hub.evaluation import regime_block
from hub.llm_tracking import simulate_trades, summarize
from hub.regimes import regime_series
from hub.sleeves import mean_reversion_regime


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


# ---------------------------------------------------------------- 成本


def test_costs_match_trading_intel_config():
    """成本數字唯一來源是 trading-intel 的設定檔，兩邊不准悄悄分歧。"""
    text = (REPO_ROOT / "trading-intel" / "configs" / "base.yaml").read_text(encoding="utf-8")
    def val(key):
        return float(re.search(rf"^\s*{key}:\s*([0-9.]+)", text, re.M).group(1))
    assert costs.COMMISSION_BPS == val("commission_bps")
    assert costs.SLIPPAGE_BPS == val("slippage_bps")
    assert costs.TW_TRANSACTION_TAX == val("tw_transaction_tax")


def test_sell_costs_more_than_buy_because_of_transaction_tax():
    assert costs.SELL_COST - costs.BUY_COST == pytest.approx(costs.TW_TRANSACTION_TAX)


# ---------------------------------------------------------------- 回測引擎


def test_weight_decided_today_only_earns_from_tomorrow():
    idx = _idx(4)
    rets = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.10]}, index=idx)
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    res = run_backtest(w, rets)
    # 第 1 天收盤才建倉，第 1 天的 +10% 不能算進去
    assert res.gross.tolist() == pytest.approx([0.0, 0.0, 0.10, 0.10])


def test_buy_and_sell_costs_charged_asymmetrically():
    idx = _idx(3)
    rets = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=idx)
    res = run_backtest(w, rets)
    assert res.cost.iloc[0] == pytest.approx(costs.BUY_COST)
    assert res.cost.iloc[2] == pytest.approx(costs.SELL_COST)


def test_short_position_pays_tax_on_open_and_borrow_fee_while_held():
    idx = _idx(3)
    rets = pd.DataFrame({"A": [0.0, -0.05, 0.0]}, index=idx)
    w = pd.DataFrame({"A": [-1.0, -1.0, -1.0]}, index=idx)
    res = run_backtest(w, rets)
    assert res.cost.iloc[0] == pytest.approx(costs.SELL_COST)  # 放空開倉 = 賣出，要課證交稅
    assert res.gross.iloc[1] == pytest.approx(0.05)             # 下跌賺錢
    assert res.cost.iloc[1] == pytest.approx(costs.SHORT_BORROW_FEE_ANNUAL / 252)


def test_slot_positions_respects_max_names_and_picks_best_scores():
    entry = np.ones((6, 5), dtype=bool)
    exit_ = np.zeros((6, 5), dtype=bool)
    score = np.tile(np.arange(5, dtype=float), (6, 1))
    out = slot_positions(entry, exit_, score, max_names=2, max_hold=3)
    assert (np.count_nonzero(out, axis=1) <= 2).all()
    assert out[0, 0] == pytest.approx(0.5) and out[0, 1] == pytest.approx(0.5)  # 分數最小的先進場
    assert out[0, 2:].sum() == 0


def test_slot_positions_exits_after_max_hold():
    entry = np.zeros((6, 3), dtype=bool)
    entry[0] = True  # 只有第一天有進場訊號
    exit_ = np.zeros((6, 3), dtype=bool)
    out = slot_positions(entry, exit_, np.zeros((6, 3)), max_names=3, max_hold=3)
    assert (out[0:3] > 0).all()
    assert (out[3:] == 0).all()


def test_unused_slots_stay_in_cash():
    entry = np.zeros((2, 4), dtype=bool)
    entry[0, 0] = True
    out = slot_positions(entry, np.zeros((2, 4), dtype=bool), np.zeros((2, 4)), max_names=4, max_hold=5)
    assert out[0].sum() == pytest.approx(0.25)  # 訊號少就少曝險，不硬塞滿


# ---------------------------------------------------------------- Ensemble


def test_cap_never_lets_a_sleeve_exceed_the_limit():
    w = cap_and_redistribute(pd.Series({"a": 3.0, "b": 1.0, "c": 0.0}), 0.4)
    assert w.max() <= 0.4 + 1e-12
    assert w.sum() == pytest.approx(0.8)  # 只有兩個 sleeve 有能力，剩下 20% 留現金


def test_cap_redistributes_to_uncapped_sleeves():
    w = cap_and_redistribute(pd.Series({"a": 6.0, "b": 2.0, "c": 2.0}), 0.4)
    assert w["a"] == pytest.approx(0.4)
    assert w.sum() == pytest.approx(1.0)
    assert w["b"] == pytest.approx(0.3) and w["c"] == pytest.approx(0.3)


def test_all_sleeves_losing_means_all_cash():
    assert cap_and_redistribute(pd.Series({"a": -1.0, "b": -0.5}), 0.4).sum() == 0


def test_ensemble_moves_capital_toward_the_sleeve_that_works():
    idx = _idx(200)
    rng = np.random.default_rng(0)
    good = pd.Series(0.002 + rng.normal(0, 0.005, 200), index=idx)
    bad = pd.Series(-0.002 + rng.normal(0, 0.005, 200), index=idx)
    res = run_ensemble(pd.DataFrame({"good": good, "bad": bad}))
    last = res.sleeve_weights.iloc[-1]
    assert last["bad"] == 0
    assert 0 < last["good"] <= MAX_SLEEVE_WEIGHT + 1e-12


# ---------------------------------------------------------------- 體制與股票池


def test_historical_regime_matches_live_regime_module():
    """歷史向量化版本的任一天，必須跟 tw_trend_core 即時版給出同樣分類。"""
    from core.regime import compute_market_regime

    idx = _idx(25)
    index_px = pd.Series(np.r_[np.full(24, 100.0), 90.0], index=idx)
    z = pd.DataFrame(np.r_[np.zeros((24, 10)), [[-2.5] * 8 + [2.5] * 2]], index=idx)
    hist = regime_series(index_px, z)
    latest = pd.DataFrame({"signal": ["OVERSOLD"] * 8 + ["OVERBOUGHT"] * 2})
    live = compute_market_regime(latest, index_px)
    assert hist.iloc[-1] == live.classification == "RISK_OFF"


def test_regime_filtered_sleeve_opens_nothing_on_risk_off_days():
    idx = _idx(5)
    cols = ["A", "B"]
    z = pd.DataFrame(-3.0, index=idx, columns=cols)
    ctx = {
        "z": z, "er": pd.DataFrame(0.1, index=idx, columns=cols),
        "universe": pd.DataFrame(True, index=idx, columns=cols),
        "regime": pd.Series(["RISK_OFF", "RISK_OFF", "NEUTRAL", "RISK_OFF", "NEUTRAL"], index=idx),
    }
    w = mean_reversion_regime(ctx)
    assert (w.iloc[0:2] == 0).all().all()
    assert (w.iloc[2] > 0).any()


def test_universe_membership_only_takes_effect_after_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    prices = pd.DataFrame({"A": 10.0, "B": 20.0}, index=idx)
    mask = monthly_universe(prices, pd.Series({"A": 1.0, "B": 1.0}), top_n=1)
    jan_end = idx[idx.month == 1][-1]
    assert not mask.loc[:jan_end].any().any()  # 歷史不足一年 → 沒有人合格，而且不會提前生效


def test_regime_block_uses_labels_as_given():
    idx = _idx(60)
    r = pd.Series(0.001, index=idx)
    labels = pd.Series(["RISK_ON"] * 30 + ["RISK_OFF"] * 30, index=idx)
    out = regime_block(r, labels)
    assert out["RISK_ON"]["share_of_days"] == pytest.approx(0.5)
    assert out["NEUTRAL"]["share_of_days"] == 0


# ---------------------------------------------------------------- 盤中資料


def test_intraday_bar_is_dropped_before_close():
    import run_hub

    idx = pd.to_datetime(["2026-11-25", "2026-11-26"])
    prices = pd.DataFrame({"A": [1.0, 1.1]}, index=idx)
    morning = datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc)   # 台北 10:00
    evening = datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc)   # 台北 16:00
    assert len(run_hub.drop_unfinished_session(prices, morning)) == 1
    assert len(run_hub.drop_unfinished_session(prices, evening)) == 2


# ---------------------------------------------------------------- LLM 前瞻追蹤


def test_llm_trades_hit_stop_loss_and_compare_to_benchmark():
    idx = _idx(5)
    prices = pd.DataFrame({"X.TW": [100, 98, 94, 90, 95.0]}, index=idx)
    bench = pd.Series([50, 50, 51, 52, 52.0], index=idx)
    docs = [{"code": "1", "action": "Buy", "entry": 100, "stop_loss": 95, "take_profit": 120,
             "confidence": 0.6, "created_at": idx[0].isoformat()}]
    t = simulate_trades(docs, prices, bench, {"1": "X.TW"})
    assert t.iloc[0]["exit_reason"] == "stop_loss"
    assert t.iloc[0]["return"] == pytest.approx(0.94 - 1 - costs.ROUND_TRIP_COST)
    assert t.iloc[0]["excess_vs_0050"] < t.iloc[0]["return"]  # 同期 0050 上漲，超額更差
    assert summarize(t)["n_trades"] == 1


def test_llm_sell_signal_profits_when_price_falls():
    idx = _idx(3)
    prices = pd.DataFrame({"X.TW": [100, 95, 85.0]}, index=idx)
    docs = [{"code": "1", "action": "Sell", "entry": 100, "stop_loss": 105, "take_profit": 90,
             "confidence": 0.5, "created_at": idx[0].isoformat()}]
    t = simulate_trades(docs, prices, pd.Series(1.0, index=idx), {"1": "X.TW"})
    assert t.iloc[0]["exit_reason"] == "take_profit"
    assert t.iloc[0]["return"] > 0


def test_risk_off_self_check_uses_decision_day_regime_not_lagged_label():
    """評估用落後一日的體制，但硬規則檢查必須用決策當天的體制，否則會誤報違規。"""
    from hub.evaluation import constraint_block

    idx = _idx(3)
    w = pd.DataFrame({"A": [0.0, 0.05, 0.05]}, index=idx)
    entries = (w > 0) & (w.shift(1).fillna(0) <= 0)          # 第 1 天進場
    decision = pd.Series(["RISK_OFF", "NEUTRAL", "NEUTRAL"], index=idx)
    lagged = decision.shift(1)                                 # 第 1 天的落後標籤是 RISK_OFF
    out = constraint_block(w, lagged, [], True, entries, decision_regimes=decision)
    assert out["new_longs_during_risk_off"] == 0
