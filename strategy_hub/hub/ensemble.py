"""
hub/ensemble.py
Ensemble Agent：讓資金跟隨模型能力，而非預測大盤方向。

把量化公司的「交易員資金配額制度」套到策略 sleeve 上：
1. 每 REBALANCE_EVERY 個交易日檢視一次各 sleeve 過去 LOOKBACK 日的「淨值」報酬
  （已扣成本），算風險調整後績效（Sharpe）。
2. 強者加碼、弱者減碼：權重 ∝ max(Sharpe, 0)。近期賠錢的 sleeve 直接歸零，
   不是「少給一點」——這就是交易員被收回額度。
3. 限制集中度：單一 sleeve 上限 MAX_SLEEVE_WEIGHT，超出部分重新分給其他 sleeve，
   全部都頂到上限還有剩，剩下的就留現金。
4. 風險預算：以 sleeve 共變異數估算組合年化波動，超過 TARGET_VOL 就等比例降槓桿。
5. 所有 sleeve 都失效時，整個組合留現金——不必先猜牛熊，「沒有人在賺錢」本身就是訊號。

決策在 t 日收盤後做，t+1 日才生效；sleeve 間資金搬移另計切換成本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hub.costs import ROUND_TRIP_COST

LOOKBACK = 63            # 約一季
REBALANCE_EVERY = 5      # 每週檢視一次
MAX_SLEEVE_WEIGHT = 0.40
TARGET_VOL = 0.15
# sleeve 之間搬錢，底層等於賣一些、買一些：搬動 1 單位約付半個來回成本。
SWITCH_COST = ROUND_TRIP_COST / 2


def cap_and_redistribute(raw: pd.Series, cap: float) -> pd.Series:
    """水位填充：把超過上限的部分按比例分給未達上限者；全部頂到上限時剩餘留現金。"""
    w = raw.clip(lower=0).astype(float)
    if w.sum() <= 0:
        return w * 0.0
    w = w / w.sum()
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        # 只能分給「還沒頂到上限」的 sleeve——已經封頂的不能再收，否則會在封頂者之間來回倒
        room = (w < cap - 1e-12) & (w > 0)
        if not room.any():
            break  # 剩餘的 excess 留現金
        w[room] += excess * w[room] / w[room].sum()
    return w.clip(upper=cap)


@dataclass
class EnsembleResult:
    sleeve_weights: pd.DataFrame
    net: pd.Series
    switch_cost: pd.Series
    trailing_sharpe: pd.DataFrame


def run_ensemble(sleeve_net: pd.DataFrame, start: pd.Timestamp | None = None) -> EnsembleResult:
    idx = sleeve_net.index
    names = sleeve_net.columns
    weights = pd.DataFrame(0.0, index=idx, columns=names)
    sharpe_log = pd.DataFrame(np.nan, index=idx, columns=names)

    first = LOOKBACK if start is None else max(LOOKBACK, idx.searchsorted(start))
    current = pd.Series(0.0, index=names)
    for t in range(first, len(idx)):
        if (t - first) % REBALANCE_EVERY == 0:
            window = sleeve_net.iloc[t - LOOKBACK + 1 : t + 1]
            mu, sd = window.mean(), window.std()
            sharpe = (mu / sd.replace(0, np.nan) * np.sqrt(252)).fillna(0.0)
            # 從未曝險（全為 0 報酬）的 sleeve 沒有能力可言，不給額度
            sharpe[window.abs().sum() == 0] = 0.0
            sharpe_log.iloc[t] = sharpe
            w = cap_and_redistribute(sharpe, MAX_SLEEVE_WEIGHT)
            if w.sum() > 0:
                cov = window.cov().to_numpy() * 252
                vol = float(np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 0.0)))
                if vol > TARGET_VOL:
                    w = w * (TARGET_VOL / vol)
            current = w
        weights.iloc[t] = current.to_numpy()

    held = weights.shift(1).fillna(0.0)
    switch = weights.diff().abs().sum(axis=1).fillna(0.0) * SWITCH_COST
    net = (held * sleeve_net).sum(axis=1) - switch
    return EnsembleResult(weights, net, switch, sharpe_log.ffill())


def static_equal_weight(sleeve_net: pd.DataFrame) -> pd.Series:
    """對照組：永遠平均分配給所有 sleeve，不看近期表現。用來檢驗『動態配置』本身有沒有價值。"""
    return sleeve_net.mean(axis=1)
