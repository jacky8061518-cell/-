"""
hub/sleeves.py
策略「子帳戶」（sleeve）：每個都是純函式，輸入價格與指標、輸出每日目標權重。
Ensemble Agent 再依各 sleeve 的近期風險調整後績效動態配置資金。

全部都是「規則型」策略，可以乾淨回測到 2016 年——這是刻意的：
LLM 的新聞分析層（tw_trend_core 的 Buy/Hold 訊號）無法做歷史回測，因為模型
訓練資料已經「看過」那段歷史（前視偏誤）。LLM 層只能前瞻追蹤，見 evaluation.py。

| sleeve                 | 對應的既有模組／假說                                  |
|------------------------|-------------------------------------------------------|
| market_timing_0050     | 使用者 top-down 假說：盤面不好就不在場上               |
| mean_reversion         | tw_trend_core 的 Z-Score 超賣 + 震盪制度（原始邏輯）  |
| mean_reversion_regime  | 同上 + regime.py 大盤過濾器（RISK_OFF 不開新多單）    |
| trend_breakout         | 創意電子教訓：超買 + 趨勢中 = 動能延續，不是要回歸    |
| momentum_12_1          | trading-intel backtest/baselines.py 的 Momentum12m1   |
| industry_rotation      | sector-rotation-research 的輪動邏輯，套到台股官方產業 |
| regime_short           | 使用者假說：「大盤不好時甚至該有空單」                |
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from hub.backtest import slot_positions

MAX_NAMES = 20                 # 每檔固定 5%：同時也是個股集中度上限
TREND_EFFICIENCY_THRESHOLD = 0.4  # 與 tw_trend_core/core/factors.py 相同


def _to_frame(arr: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(arr, index=like.index, columns=like.columns)


def mean_reversion(ctx: dict, regime_filter: bool = False) -> pd.DataFrame:
    z, er, uni = ctx["z"], ctx["er"], ctx["universe"]
    entry = (z <= -2) & (er < TREND_EFFICIENCY_THRESHOLD) & uni
    if regime_filter:
        entry = entry & (ctx["regime"] != "RISK_OFF").to_numpy()[:, None]
    exit_ = z >= 0
    arr = slot_positions(entry.to_numpy(), exit_.fillna(False).to_numpy(), z.to_numpy(),
                         max_names=MAX_NAMES, max_hold=10)
    return _to_frame(arr, z)


def mean_reversion_regime(ctx: dict) -> pd.DataFrame:
    return mean_reversion(ctx, regime_filter=True)


def trend_breakout(ctx: dict) -> pd.DataFrame:
    z, er, uni, prices, ma = ctx["z"], ctx["er"], ctx["universe"], ctx["prices"], ctx["ma"]
    entry = (z >= 2) & (er >= TREND_EFFICIENCY_THRESHOLD) & uni
    exit_ = prices < ma  # 跌破 20 日均線即出場（移動停損）
    arr = slot_positions(entry.to_numpy(), exit_.fillna(False).to_numpy(), (-z).to_numpy(),
                         max_names=MAX_NAMES, max_hold=60)
    return _to_frame(arr, z)


def _month_end_positions(index: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(np.arange(len(index)), index=index)
    return s.groupby(index.to_period("M")).last().tolist()


def momentum_12_1(ctx: dict) -> pd.DataFrame:
    prices, uni = ctx["prices"], ctx["universe"]
    score = prices.shift(21) / prices.shift(252) - 1
    w = np.zeros(prices.shape)
    rebal = _month_end_positions(prices.index)
    for i, t in enumerate(rebal):
        s = score.iloc[t].where(uni.iloc[t])
        s = s[s > 0].nlargest(MAX_NAMES)  # 只做正動能，沒有就留現金
        end = rebal[i + 1] if i + 1 < len(rebal) else len(prices.index) - 1
        if len(s):
            w[t:end + 1, prices.columns.get_indexer(s.index)] = 1 / MAX_NAMES
    return _to_frame(w, prices)


def industry_rotation(ctx: dict, top_industries: int = 3, lookback: int = 126) -> pd.DataFrame:
    prices, uni, industry = ctx["prices"], ctx["universe"], ctx["industry"]
    rets = ctx["returns"].where(uni)
    groups = industry.reindex(prices.columns)
    ind_ret = rets.T.groupby(groups).mean().T  # 等權重產業指數日報酬
    ind_mom = (1 + ind_ret.fillna(0)).rolling(lookback).apply(np.prod, raw=True) - 1
    w = np.zeros(prices.shape)
    rebal = _month_end_positions(prices.index)
    per_name_cap = 1 / MAX_NAMES
    for i, t in enumerate(rebal):
        mom = ind_mom.iloc[t].dropna()
        leaders = mom[mom > 0].nlargest(top_industries).index
        end = rebal[i + 1] if i + 1 < len(rebal) else len(prices.index) - 1
        for ind in leaders:
            members = groups.index[(groups == ind) & uni.iloc[t].to_numpy()]
            if len(members) == 0:
                continue
            each = min((1 / top_industries) / len(members), per_name_cap)  # 小產業不准單押
            w[t:end + 1, prices.columns.get_indexer(members)] = each
    return _to_frame(w, prices)


def regime_short(ctx: dict) -> pd.DataFrame:
    z, uni = ctx["z"], ctx["universe"]
    risk_off = (ctx["regime"] == "RISK_OFF").to_numpy()[:, None]
    risk_on = (ctx["regime"] == "RISK_ON").to_numpy()[:, None]
    entry = (z >= 2) & uni & risk_off            # 大盤系統性下殺時仍逆勢超買 → 補跌候選
    exit_ = (z <= 0).fillna(False) | risk_on     # 回到均線或大盤轉強就回補
    arr = slot_positions(entry.to_numpy(), exit_.to_numpy(), (-z).to_numpy(),
                         max_names=10, max_hold=10, side=-0.5)
    return _to_frame(arr, z)


def market_timing_0050(ctx: dict) -> pd.DataFrame:
    """Top-down 最直接的版本：大盤體制不是 RISK_OFF 就全額持有 0050，RISK_OFF 就退到現金。

    只換 0050 一檔，不碰個股——檢驗「先看盤面好不好再決定要不要在場上」這件事本身值多少。
    """
    bench = ctx["benchmark_ticker"]
    w = pd.DataFrame(0.0, index=ctx["regime"].index, columns=[bench])
    w[bench] = (ctx["regime"] != "RISK_OFF").astype(float)
    return w


SLEEVES: dict[str, Callable[[dict], pd.DataFrame]] = {
    "market_timing_0050": market_timing_0050,
    "mean_reversion": mean_reversion,
    "mean_reversion_regime": mean_reversion_regime,
    "trend_breakout": trend_breakout,
    "momentum_12_1": momentum_12_1,
    "industry_rotation": industry_rotation,
    "regime_short": regime_short,
}

SLEEVE_LABELS = {
    "market_timing_0050": "0050＋大盤過濾",
    "mean_reversion": "均值回歸（原始）",
    "mean_reversion_regime": "均值回歸＋大盤過濾",
    "trend_breakout": "趨勢突破",
    "momentum_12_1": "12-1 動能",
    "industry_rotation": "產業輪動",
    "regime_short": "風險規避放空",
}
