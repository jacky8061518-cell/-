"""
hub/backtest.py
共用的向量化回測引擎：給定每日目標權重，算出含成本的每日報酬。

時間紀律：第 t 日收盤後決定的權重，從 t+1 日起才開始賺報酬（shift(1)），
交易成本記在 t 日（收盤成交）。任何策略都不可能用到當天收盤之後的資訊。

簡化（誠實標註）：目標權重在兩次換倉之間視為固定，等同每天被動再平衡回目標，
這部分漂移造成的微量換手沒有計成本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hub.costs import BUY_COST, SELL_COST, SHORT_BORROW_FEE_ANNUAL

# 台股漲跌幅限制 10%，新上市前五日無限制；超過 ±40% 的單日報酬幾乎一定是資料錯誤
#（未調整的減資／分割），截斷並計入資料品質事件，而不是讓它污染回測。
RETURN_GLITCH_LIMIT = 0.40


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    gross: pd.Series
    cost: pd.Series
    net: pd.Series
    turnover: pd.Series  # 每日單邊換手率
    gross_exposure: pd.Series
    net_exposure: pd.Series
    holdings: pd.Series


def daily_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """收盤價 → 日報酬；回傳 (報酬, 被截斷的異常值數量)。"""
    rets = prices.pct_change(fill_method=None)
    glitches = int((rets.abs() > RETURN_GLITCH_LIMIT).sum().sum())
    rets = rets.where(rets.abs() <= RETURN_GLITCH_LIMIT, 0.0).fillna(0.0)
    return rets, glitches


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> BacktestResult:
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)

    trades = w.diff()
    trades.iloc[0] = w.iloc[0]
    # 權重增加 = 買進（含回補空單），權重減少 = 賣出（含放空開倉），後者要課證交稅。
    buys = trades.clip(lower=0).sum(axis=1)
    sells = (-trades.clip(upper=0)).sum(axis=1)
    short_notional = (-held.clip(upper=0)).sum(axis=1)
    cost = buys * BUY_COST + sells * SELL_COST + short_notional * SHORT_BORROW_FEE_ANNUAL / 252

    return BacktestResult(
        weights=w,
        gross=gross,
        cost=cost,
        net=gross - cost,
        turnover=(buys + sells) / 2,
        gross_exposure=w.abs().sum(axis=1),
        net_exposure=w.sum(axis=1),
        holdings=(w != 0).sum(axis=1),
    )


def slot_positions(
    entry: np.ndarray,
    exit_: np.ndarray,
    score: np.ndarray,
    max_names: int,
    max_hold: int,
    side: float = 1.0,
) -> np.ndarray:
    """事件驅動的「固定格位」持倉：每檔固定 1/max_names 權重，沒用到的格位留現金。

    刻意不把資金硬塞滿——訊號少的時候就該少曝險，這本身就是風險控管。
    score 越小越優先進場（呼叫端自行決定正負號）。
    """
    t_len, n = entry.shape
    held = np.zeros(n, dtype=bool)
    age = np.zeros(n, dtype=np.int32)
    out = np.zeros((t_len, n))
    unit = side / max_names
    for t in range(t_len):
        leaving = held & (exit_[t] | (age >= max_hold))
        held[leaving] = False
        age[leaving] = 0
        slots = max_names - int(held.sum())
        if slots > 0:
            cand = np.flatnonzero(entry[t] & ~held)
            if cand.size:
                s = score[t, cand]
                s = np.where(np.isnan(s), np.inf, s)
                pick = cand[np.argsort(s, kind="stable")[:slots]]
                held[pick] = True
        age[held] += 1
        out[t, held] = unit
    return out
