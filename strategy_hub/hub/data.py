"""
hub/data.py
資料層：重用 tw_trend_core 的價格快取與股票池（它本身又重用 sector-rotation-research
的共用種子），不另外維護一份價格資料。

股票池的已知偏誤（誠實標註，評估時一律拿「同樣偏誤的等權重股票池」當公平基準）：
- 倖存者偏誤：security-master 只有「目前仍上市」的公司，已下市的不在裡面。
- 股本前視：市值代理 = 歷史股價 × 「目前」發行股數。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TW_CORE_ROOT = REPO_ROOT / "tw_trend_core"
if str(TW_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(TW_CORE_ROOT))

SECURITY_MASTER_PATH = REPO_ROOT / "data" / "databases" / "tw" / "security-master.csv"
BENCHMARK_TICKER = "0050.TW"
MIN_HISTORY_DAYS = 252


def load_security_master() -> pd.DataFrame:
    df = pd.read_csv(SECURITY_MASTER_PATH, dtype={"Code": str})
    return df[df["Asset type"] == "股票"].set_index("Yahoo ticker")


def load_panel() -> pd.DataFrame:
    """全部個股 + 0050 的還原收盤價寬表（日期 × 標的）。"""
    from core.quant_engine import load_price_history  # tw_trend_core

    tickers = load_security_master().index.tolist() + [BENCHMARK_TICKER]
    return load_price_history(tickers)


def monthly_universe(prices: pd.DataFrame, shares: pd.Series, top_n: int = 300) -> pd.DataFrame:
    """每月底依「市值代理」取前 top_n 大、且至少有一年歷史的個股，當月沿用。

    回傳與 prices 同形狀的布林遮罩。只用「當時」的價格排名，不偷看未來股價。
    """
    shares = shares.reindex(prices.columns)
    history = prices.notna().cumsum()
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    rows = []
    for dt in month_ends:
        mcap = prices.loc[dt] * shares
        eligible = mcap[(history.loc[dt] >= MIN_HISTORY_DAYS) & mcap.notna() & (mcap > 0)]
        rows.append((dt, set(eligible.nlargest(top_n).index)))
    # 月底決定的名單從「下一個交易日」起生效
    pos = {dt: i for i, dt in enumerate(prices.index)}
    for i, (dt, names) in enumerate(rows):
        start = pos[dt] + 1
        end = pos[rows[i + 1][0]] + 1 if i + 1 < len(rows) else len(prices.index)
        if start < end and names:
            mask.iloc[start:end, mask.columns.get_indexer(list(names))] = True
    return mask


def indicators(prices: pd.DataFrame, length: int = 20) -> dict[str, pd.DataFrame]:
    """與 tw_trend_core 完全相同定義的 20 期 Z-Score 與 Kaufman 效率比（向量化、整段歷史）。"""
    ma = prices.rolling(length, min_periods=length).mean()
    sd = prices.rolling(length, min_periods=length).std()
    z = (prices - ma) / sd.replace(0, np.nan)
    net = (prices - prices.shift(length)).abs()
    path = prices.diff().abs().rolling(length, min_periods=length).sum()
    er = (net / path.replace(0, np.nan))
    return {"ma": ma, "z": z, "er": er}
