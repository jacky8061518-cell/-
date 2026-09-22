"""
core/factors.py
量化訊號強化層：在基礎 Z-Score 之上疊加四個獨立因子，讓「哪些異常標的
值得花時間研究」的排序不再只靠單一價格統計量：

1. 趨勢/震盪制度判別（Kaufman Efficiency Ratio）
   高 Z-Score 若發生在「真的在趨勢中」的標的上，代表的是動能延續訊號，
   不該套用均值回歸框架去建議「等拉回」——這正是創意電子（3443）分析
   第一版誤判的原因，這裡把該教訓量化成可重複計算的指標，而不是每次
   都靠人工新聞查證才能發現。
2. 產業相對 Z-Score
   用 security-master.csv 既有的產業分類，把個股 Z-Score 減去同產業
   全市場中位數，分辨「整個產業都在動（有真實產業事件）」與
   「單一個股異常（可能是雜訊或真的個股利多利空）」。
3. 成交量 Z-Score（量價確認）
   價格異常若沒有成交量放大佐證，可信度較低；此模組另外抓取成交量
   歷史（不動用共用的 adjusted-prices.parquet，該檔只有收盤價）。
4. 流動性篩選
   排除成交金額過小、實際上難以進出的標的，避免異常清單被一堆
   幾乎沒有流動性的雜訊股佔滿。

最後把四個因子合成一個可解釋（非黑箱）的 Edge Score（0-100），
取代單純依 |Z-Score| 排序。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# 參數
# ---------------------------------------------------------------------------

TREND_LENGTH = 20
TREND_EFFICIENCY_THRESHOLD = 0.4  # Kaufman ER 超過此值視為「趨勢中」

VOLUME_LOOKBACK_DAYS = 40  # 只需近期資料算 20 期量能 Z-Score，不需要 14 年歷史
VOLUME_BATCH_SIZE = 100

MIN_DAILY_TURNOVER_TWD = 3_000_000       # 低於此日均成交金額視為流動性不足，排除
LIQUIDITY_FULL_SCORE_TWD = 50_000_000    # 達到此日均成交金額給滿分流動性因子

EDGE_SCORE_WEIGHTS = {"extremity": 0.5, "volume_confirmation": 0.3, "liquidity": 0.2}


# ---------------------------------------------------------------------------
# 1. 趨勢／震盪制度判別（Kaufman Efficiency Ratio）
# ---------------------------------------------------------------------------

def compute_trend_efficiency(prices: pd.DataFrame, length: int = TREND_LENGTH) -> pd.Series:
    """Kaufman Efficiency Ratio：淨變動 ÷ 總路徑長度，介於 0（原地震盪、無淨進展）
    到 1（單邊直線趨勢）之間。純函式、全向量化，不對每檔標的跑迴圈或線性迴歸。
    """
    if len(prices) <= length:
        return pd.Series(float("nan"), index=prices.columns)

    net_change = (prices.iloc[-1] - prices.iloc[-1 - length]).abs()
    path_length = prices.diff().abs().iloc[-length:].sum()
    er = (net_change / path_length).replace([float("inf"), float("-inf")], float("nan"))
    return er


def classify_regime(
    prices: pd.DataFrame, length: int = TREND_LENGTH, threshold: float = TREND_EFFICIENCY_THRESHOLD,
) -> pd.DataFrame:
    """回傳每檔標的的效率比與制度標籤：TRENDING_UP / TRENDING_DOWN / RANGE_BOUND。

    效率比缺值（例如路徑長度為 0，價格完全沒有波動）保守視為 RANGE_BOUND，
    因為此時沒有任何證據支持「這是趨勢」。
    """
    efficiency = compute_trend_efficiency(prices, length)
    direction_up = (prices.iloc[-1] - prices.iloc[-1 - length]) > 0

    def _label(ticker: str) -> str:
        er = efficiency.get(ticker)
        if pd.isna(er) or er < threshold:
            return "RANGE_BOUND"
        return "TRENDING_UP" if direction_up.get(ticker, False) else "TRENDING_DOWN"

    return pd.DataFrame({
        "trend_efficiency": efficiency,
        "regime": [_label(t) for t in efficiency.index],
    })


# ---------------------------------------------------------------------------
# 2. 產業相對 Z-Score
# ---------------------------------------------------------------------------

def compute_industry_median_zscores(full_latest: pd.DataFrame, industry_by_ticker: dict[str, str]) -> pd.Series:
    """用「全市場」（非僅異常清單）的 Z-Score 計算各產業的中位數，
    避免用已經篩選過的異常子集反過來估計產業基準造成偏誤。
    """
    industries = pd.Series(industry_by_ticker, name="industry")
    combined = full_latest[["z_score"]].join(industries, how="inner")
    return combined.groupby("industry")["z_score"].median()


def attach_industry_relative_zscore(
    anomalies: pd.DataFrame, industry_median_by_group: pd.Series, industry_by_ticker: dict[str, str],
) -> pd.DataFrame:
    """在異常標的清單上附加所屬產業的中位數 Z-Score，以及個股相對於產業的偏離。

    industry_relative_z 接近 0：這檔標的跟整個產業同步動，很可能是產業層級的事件。
    industry_relative_z 遠離 0：這檔標的明顯比同業更異常，是真正的個股層級訊號。
    """
    anomalies = anomalies.copy()
    anomalies["industry_median_z"] = anomalies["industry"].map(industry_median_by_group)
    anomalies["industry_relative_z"] = anomalies["z_score"] - anomalies["industry_median_z"]
    return anomalies


# ---------------------------------------------------------------------------
# 3. 成交量 Z-Score（量價確認）
# ---------------------------------------------------------------------------

def _fetch_volume_batch(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers, start=start, end=end, progress=False,
        group_by="column", threads=min(16, len(tickers)), timeout=20,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Volume" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        return raw["Volume"].copy()
    if "Volume" not in raw:
        return pd.DataFrame()
    result = raw[["Volume"]].copy()
    result.columns = [tickers[0]]
    return result


def load_volume_history(
    tickers: list[str], lookback_days: int = VOLUME_LOOKBACK_DAYS, batch_size: int = VOLUME_BATCH_SIZE,
) -> pd.DataFrame:
    """抓取近期成交量歷史（只需近 40 天即可算 20 期量能 Z-Score，不需要像收盤價
    那樣回溯 14 年，因此不重用共用的 adjusted-prices.parquet 種子機制，
    每次直接抓取短窗口）。"""
    start = date.today() - timedelta(days=lookback_days)
    end = date.today() + timedelta(days=1)

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    frames = [_fetch_volume_batch(batch, start, end) for batch in batches]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    volumes = pd.concat(frames, axis=1)
    return volumes.reindex(columns=tickers)


def compute_volume_zscore(volumes: pd.DataFrame, length: int = TREND_LENGTH) -> pd.DataFrame:
    """量能的滾動 Z-Score：今日成交量偏離近期平均成交量幾個標準差。"""
    mean = volumes.rolling(length, min_periods=length).mean()
    std = volumes.rolling(length, min_periods=length).std()
    z = (volumes - mean) / std

    latest = pd.DataFrame({
        "volume": volumes.iloc[-1],
        "avg_volume": mean.iloc[-1],
        "volume_z_score": z.iloc[-1],
    })
    latest.index.name = "yahoo_ticker"
    return latest


# ---------------------------------------------------------------------------
# 4. 流動性
# ---------------------------------------------------------------------------

def compute_liquidity(avg_volume: pd.Series, latest_price: pd.Series) -> pd.Series:
    """粗估日均成交金額（新台幣） = 近期平均成交量 × 現價。"""
    return avg_volume * latest_price


# ---------------------------------------------------------------------------
# 5. 合成 Edge Score
# ---------------------------------------------------------------------------

def compute_edge_score(enriched: pd.DataFrame) -> pd.Series:
    """把統計極端度、量能確認、流動性合成一個透明可拆解的 0-100 分數，
    取代單純依 |Z-Score| 排序。刻意不把「趨勢/震盪制度」與「產業相對 Z」
    折進分數本身——這兩者是「該怎麼解讀訊號」的標籤，不是「訊號強不強」
    的量測，混在同一個數字裡會讓分數變得不透明。
    """
    extremity = (enriched["z_score"].abs() / 4).clip(upper=1)
    volume_confirmation = (enriched["volume_z_score"] / 3).clip(lower=0, upper=1).fillna(0)
    liquidity = (enriched["avg_daily_turnover"] / LIQUIDITY_FULL_SCORE_TWD).clip(upper=1).fillna(0)

    w = EDGE_SCORE_WEIGHTS
    score = 100 * (w["extremity"] * extremity + w["volume_confirmation"] * volume_confirmation + w["liquidity"] * liquidity)
    return score.round(1)


def apply_liquidity_filter(
    enriched: pd.DataFrame, min_turnover: float = MIN_DAILY_TURNOVER_TWD,
) -> pd.DataFrame:
    """排除日均成交金額低於門檻（預設新台幣 300 萬）的標的——這些即使統計上
    異常，實際上也難以用有意義的部位進出，保留只會讓清單充滿雜訊股。"""
    return enriched[enriched["avg_daily_turnover"].fillna(0) >= min_turnover]
