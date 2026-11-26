"""
hub/regimes.py
把 tw_trend_core/core/regime.py 的「當天」大盤體制判別，向量化套用到整段歷史，
讓每個策略都能拆成「在 RISK_ON / NEUTRAL / RISK_OFF 各自表現如何」（評估第 03 項）。

門檻常數直接 import 自 regime.py，不複製數字；tests 另外驗證任一天的歷史標籤
與即時版 compute_market_regime 的結果完全一致。

指數來源與即時版相同，都是加權指數（^TWII）。只有在抓不到 ^TWII 時才退回用 0050 代理，
並在營運健康度標示 warn——曾發生 Yahoo 缺 0050 某日資料、被前值填補，導致代理版與即時版
給出不同體制（即時 RISK_OFF、代理 NEUTRAL），所以代理只能當備援。
"""

from __future__ import annotations

import pandas as pd

from hub.data import TW_CORE_ROOT  # noqa: F401  確保 tw_trend_core 在 sys.path

from core.regime import (  # tw_trend_core
    RISK_OFF_OVERSOLD_RATIO,
    RISK_OFF_TAIEX_DEVIATION,
    RISK_ON_OVERSOLD_RATIO,
    TAIEX_MA_LENGTH,
)
from core.quant_engine import ZSCORE_THRESHOLD  # tw_trend_core

REGIMES = ("RISK_ON", "NEUTRAL", "RISK_OFF")


def regime_series(index_prices: pd.Series, z_all: pd.DataFrame) -> pd.Series:
    ma = index_prices.rolling(TAIEX_MA_LENGTH, min_periods=TAIEX_MA_LENGTH).mean()
    deviation = index_prices / ma - 1

    oversold = (z_all <= -ZSCORE_THRESHOLD).sum(axis=1)
    overbought = (z_all >= ZSCORE_THRESHOLD).sum(axis=1)
    total = oversold + overbought
    ratio = (oversold / total).where(total > 0)

    taiex_bearish = deviation <= RISK_OFF_TAIEX_DEVIATION
    breadth_bearish = ratio >= RISK_OFF_OVERSOLD_RATIO
    risk_on = (deviation > 0) & (ratio <= RISK_ON_OVERSOLD_RATIO)

    labels = pd.Series("NEUTRAL", index=index_prices.index)
    labels[risk_on.fillna(False)] = "RISK_ON"
    labels[(taiex_bearish & breadth_bearish).fillna(False)] = "RISK_OFF"
    return labels


def threshold_grid(index_px: pd.Series, z_all: pd.DataFrame, bench_returns: pd.Series, start: pd.Timestamp,
                   devs=(-0.02, -0.03, -0.04, -0.05), ratios=(0.6, 0.7, 0.8)) -> list[dict]:
    """穩健度檢查：把大盤過濾套在 0050 上，掃過一組門檻。

    如果只有正好用的那組門檻有效、旁邊一格就失效，代表是過度擬合的巧合；
    整張表都穩定優於買進持有，才算是真的有用的規則。
    """
    from hub.backtest import run_backtest
    from hub.evaluation import cagr, drawdown, sharpe

    dev = index_px / index_px.rolling(TAIEX_MA_LENGTH, min_periods=TAIEX_MA_LENGTH).mean() - 1
    os_, ob = (z_all <= -ZSCORE_THRESHOLD).sum(axis=1), (z_all >= ZSCORE_THRESHOLD).sum(axis=1)
    ratio = (os_ / (os_ + ob)).where(os_ + ob > 0)
    rets = bench_returns.to_frame("B")
    out = []
    for d in devs:
        for rt in ratios:
            off = ((dev <= d) & (ratio >= rt)).fillna(False)
            r = run_backtest(pd.DataFrame({"B": (~off).astype(float)}, index=rets.index), rets).net[start:]
            out.append({"dev": d, "ratio": rt, "cagr": round(cagr(r), 4), "sharpe": round(sharpe(r), 3),
                        "max_drawdown": round(float(drawdown(r).min()), 4),
                        "share_out": round(float(off[start:].mean()), 4),
                        "is_live": d == RISK_OFF_TAIEX_DEVIATION and rt == RISK_OFF_OVERSOLD_RATIO})
    return out
