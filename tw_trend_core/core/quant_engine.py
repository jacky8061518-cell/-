"""
core/quant_engine.py
台股全市場量化引擎：針對全部上市 + 上櫃個股，一次性以向量化運算
（而非逐檔迴圈）計算 20 期滾動布林帶與 Z-Score，找出統計異常標的。

價格資料重用 sector-rotation-research 專案（repo 根目錄 src/sector_rotation）
已維護的下載與修補邏輯，但寫入獨立的本地快取檔案，絕不覆寫該專案共用的
data/databases/tw/adjusted-prices.parquet。
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sector_rotation.data import download_adjusted_prices  # noqa: E402

logger = logging.getLogger(__name__)

SHARED_SEED_PATH = REPO_ROOT / "data" / "databases" / "tw" / "adjusted-prices.parquet"
OWN_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "tw_price_cache.parquet"

BB_LENGTH = 20
BB_STD = 2.0
ZSCORE_THRESHOLD = 2.0

# 重新抓取「新鮮資料」時，往回多抓幾天做為緩衝，涵蓋非交易日與資料延遲
REFRESH_BUFFER_DAYS = 10


def load_price_history(
    tickers: list[str],
    cache_path: Path = OWN_CACHE_PATH,
    shared_seed_path: Path = SHARED_SEED_PATH,
) -> pd.DataFrame:
    """載入台股歷史收盤價，並將最新交易日補齊到快取中。

    第一次執行時，從 sector-rotation-research 專案已提交的共用價格檔
    (shared_seed_path) 讀出既有 14 年歷史做為種子（零網路成本），
    僅對外抓取種子檔案之後的「新鮮」區間；之後每次執行只需抓取自己
    快取檔案最後一筆日期起算的近期資料。結果一律寫入 cache_path
    （本專案自己的快取檔案），絕不寫回共用檔案。
    """
    base = pd.DataFrame()
    if cache_path.exists():
        base = pd.read_parquet(cache_path)
        base.index = pd.to_datetime(base.index)
    elif shared_seed_path.exists():
        seed = pd.read_parquet(shared_seed_path)
        seed.index = pd.to_datetime(seed.index)
        base = seed.reindex(columns=[t for t in tickers if t in seed.columns])

    if base.empty:
        start = date(2012, 1, 1)
    else:
        start = (base.index.max() - timedelta(days=REFRESH_BUFFER_DAYS)).date()

    fresh = download_adjusted_prices(
        tickers, start, date.today() + timedelta(days=1), min_observations=1,
    )

    combined = pd.concat([base, fresh]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(columns=tickers).ffill()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    return combined


def compute_latest_zscores(
    prices: pd.DataFrame, length: int = BB_LENGTH, std_mult: float = BB_STD,
) -> pd.DataFrame:
    """對整張寬表（每檔標的一欄）一次性計算 20 期布林帶與 Z-Score，回傳最新一天的結果。

    純函式，不做任何 I/O，方便用合成資料測試。
    """
    mean = prices.rolling(length, min_periods=length).mean()
    std = prices.rolling(length, min_periods=length).std()
    z_score = (prices - mean) / std

    latest = pd.DataFrame({
        "price": prices.iloc[-1],
        "mean": mean.iloc[-1],
        "std": std.iloc[-1],
        "z_score": z_score.iloc[-1],
    })
    latest["upper_band"] = latest["mean"] + std_mult * latest["std"]
    latest["lower_band"] = latest["mean"] - std_mult * latest["std"]
    latest["signal"] = latest["z_score"].apply(_detect_signal)
    latest.index.name = "yahoo_ticker"
    # 只排除歷史不足（mean/std 本身就是 NaN）的標的；價格連續持平導致 std=0、
    # z_score 變成 0/0=NaN 的標的仍保留，交給 _detect_signal 視為 NEUTRAL。
    return latest.dropna(subset=["mean", "std"])


def _detect_signal(z: float) -> str:
    if pd.isna(z):
        return "NEUTRAL"
    if z >= ZSCORE_THRESHOLD:
        return "OVERBOUGHT"
    if z <= -ZSCORE_THRESHOLD:
        return "OVERSOLD"
    return "NEUTRAL"


def get_anomalies(latest: pd.DataFrame, threshold: float = ZSCORE_THRESHOLD) -> pd.DataFrame:
    """從全市場最新結果中篩出 |Z-Score| > threshold 的異常標的，依偏離程度排序。"""
    anomalies = latest[latest["z_score"].abs() > threshold].copy()
    return anomalies.reindex(anomalies["z_score"].abs().sort_values(ascending=False).index)
