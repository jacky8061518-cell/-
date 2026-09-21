"""
core/quant_engine.py
量化引擎：負責抓取市場數據、計算布林帶與 Z-Score，並偵測統計異常訊號。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta  # noqa: F401  # 技術指標擴充套件，供未來加入更多指標使用
except ImportError:  # pragma: no cover - pandas_ta 對 Python 版本要求較嚴格，允許降級運作
    ta = None

logger = logging.getLogger(__name__)

# 預設監控標的清單
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL"]

# 布林帶參數：20 週期移動平均線、2 倍標準差
BB_LENGTH = 20
BB_STD = 2.0

# Z-Score 超過此門檻視為統計上的異常訊號
ZSCORE_THRESHOLD = 2.0


@dataclass
class MarketSnapshot:
    """單一標的的最新量化快照。"""

    symbol: str
    price: float
    mean: float
    upper_band: float
    lower_band: float
    std_dev: float
    z_score: float
    signal: str  # "OVERBOUGHT" / "OVERSOLD" / "NEUTRAL"
    history: pd.DataFrame = field(repr=False)


def fetch_hourly_data(symbol: str, period: str = "60d") -> pd.DataFrame:
    """抓取指定標的的一小時 K 線數據。"""
    df = yf.Ticker(symbol).history(period=period, interval="1h")
    if df.empty:
        raise ValueError(f"無法取得 {symbol} 的數據，請確認代碼是否正確。")
    return df.rename(columns=str.lower)


def compute_bollinger_and_zscore(
    df: pd.DataFrame, length: int = BB_LENGTH, std_mult: float = BB_STD
) -> pd.DataFrame:
    """計算布林帶（20 週期均值、上下軌）與 Z-Score，回傳附加欄位後的資料表。"""
    if len(df) < length:
        raise RuntimeError("布林帶計算失敗，歷史數據長度不足。")

    df = df.copy()
    df["bb_mean"] = df["close"].rolling(window=length).mean()
    df["bb_std"] = df["close"].rolling(window=length).std()
    df["bb_upper"] = df["bb_mean"] + std_mult * df["bb_std"]
    df["bb_lower"] = df["bb_mean"] - std_mult * df["bb_std"]
    df["z_score"] = (df["close"] - df["bb_mean"]) / df["bb_std"]
    return df


def detect_signal(z_score: float) -> str:
    """根據 Z-Score 判斷是否觸發超買（Overbought）或超賣（Oversold）異常訊號。"""
    if pd.isna(z_score):
        return "NEUTRAL"
    if z_score >= ZSCORE_THRESHOLD:
        return "OVERBOUGHT"
    if z_score <= -ZSCORE_THRESHOLD:
        return "OVERSOLD"
    return "NEUTRAL"


def scan_symbol(symbol: str) -> MarketSnapshot:
    """對單一標的執行完整的抓取 + 指標計算流程，回傳最新快照。"""
    df = fetch_hourly_data(symbol)
    df = compute_bollinger_and_zscore(df)
    latest = df.iloc[-1]
    z_score = float(latest["z_score"])

    return MarketSnapshot(
        symbol=symbol,
        price=float(latest["close"]),
        mean=float(latest["bb_mean"]),
        upper_band=float(latest["bb_upper"]),
        lower_band=float(latest["bb_lower"]),
        std_dev=float(latest["bb_std"]),
        z_score=z_score,
        signal=detect_signal(z_score),
        history=df,
    )


def scan_market(symbols: list[str] | None = None) -> dict[str, MarketSnapshot]:
    """掃描多個標的，個別標的失敗不影響其餘標的的掃描結果。"""
    symbols = symbols or DEFAULT_SYMBOLS
    snapshots: dict[str, MarketSnapshot] = {}
    for symbol in symbols:
        try:
            snapshots[symbol] = scan_symbol(symbol)
        except Exception as exc:  # noqa: BLE001  # 單一標的失敗不應中斷整體掃描
            logger.warning("掃描 %s 時發生錯誤：%s", symbol, exc)
    return snapshots


def get_anomalies(snapshots: dict[str, MarketSnapshot]) -> list[MarketSnapshot]:
    """從掃描結果中篩選出觸發異常訊號（超買 / 超賣）的標的。"""
    return [snap for snap in snapshots.values() if snap.signal != "NEUTRAL"]
