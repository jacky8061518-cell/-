"""
core/regime.py
大盤體制判別（top-down 市場過濾器）：在個股層級的 Buy/Sell 建議之前，
先問一個更根本的問題——現在整個大盤是不是處於系統性下殺？

動機：對 2026-09-21~11-25 已發出訊號的回測顯示，在台股史詩級崩盤期間，
即使個股層級查到扎實的「再評價」證據，Buy 訊號平均虧損仍達約 -11%。
問題不是個股分析查證不夠仔細，是系統完全沒有「大盤正在系統性下殺時，
先別進場」的機制——會不斷對超跌股喊買進，等於持續接落下的刀，而刀還在落。

本模組用兩個完全透明、可重現、不需要額外付費資料源的訊號組合判斷體制，
刻意不做成黑箱評分：
1. 大盤趨勢：加權指數（^TWII）現價相對 20 日均線的乖離。
2. 市場寬度：全市場異常標的中「超賣」占比——這個資料本來就是
   daily_scan.py 全市場掃描的副產品；超賣占比極端偏高本身就是
   系統性去槓桿／恐慌性賣壓的強烈訊號（例如崩盤週某次掃描
   892 檔異常中 890 檔超賣）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

TAIEX_TICKER = "^TWII"
TAIEX_MA_LENGTH = 20

# 門檻刻意取整數、有討論空間，但絕對透明可重現——不是機器學習調出來的黑箱數字。
RISK_OFF_TAIEX_DEVIATION = -0.03   # 加權指數跌破20日均線超過3%
RISK_OFF_OVERSOLD_RATIO = 0.70     # 異常標的中超賣占比達70%以上
RISK_ON_OVERSOLD_RATIO = 0.50      # 超賣占比低於50%才算風險偏好正常


@dataclass(frozen=True)
class MarketRegime:
    classification: str  # "RISK_ON" | "NEUTRAL" | "RISK_OFF"
    taiex_price: float | None
    taiex_ma20: float | None
    taiex_deviation: float | None  # (price - ma20) / ma20
    oversold_count: int
    overbought_count: int
    oversold_ratio: float | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "taiex_price": self.taiex_price,
            "taiex_ma20": self.taiex_ma20,
            "taiex_deviation": self.taiex_deviation,
            "oversold_count": self.oversold_count,
            "overbought_count": self.overbought_count,
            "oversold_ratio": self.oversold_ratio,
            "reason": self.reason,
        }


def fetch_taiex_history(lookback_days: int = 60) -> pd.Series:
    """抓取加權指數（^TWII）近期收盤價，只為算20日均線，不需要長歷史。"""
    import yfinance as yf

    start = date.today() - timedelta(days=lookback_days)
    end = date.today() + timedelta(days=1)
    raw = yf.download(TAIEX_TICKER, start=start, end=end, progress=False, timeout=20)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def compute_market_regime(
    full_latest: pd.DataFrame, taiex_history: pd.Series | None = None,
) -> MarketRegime:
    """根據大盤趨勢與全市場超賣/超買寬度，判斷目前是否為系統性風險規避環境。

    full_latest：daily_scan.py 全市場 Z-Score 計算結果
    （quant_engine.compute_latest_zscores 的輸出），用其中的 signal 欄位
    （OVERBOUGHT/OVERSOLD/NEUTRAL）算市場寬度，不需要額外資料。
    """
    oversold_count = int((full_latest["signal"] == "OVERSOLD").sum())
    overbought_count = int((full_latest["signal"] == "OVERBOUGHT").sum())
    total_extreme = oversold_count + overbought_count
    oversold_ratio = (oversold_count / total_extreme) if total_extreme > 0 else None

    taiex_price = taiex_ma20 = taiex_deviation = None
    if taiex_history is not None and len(taiex_history) >= TAIEX_MA_LENGTH:
        taiex_price = float(taiex_history.iloc[-1])
        taiex_ma20 = float(taiex_history.rolling(TAIEX_MA_LENGTH).mean().iloc[-1])
        taiex_deviation = (taiex_price - taiex_ma20) / taiex_ma20

    taiex_bearish = taiex_deviation is not None and taiex_deviation <= RISK_OFF_TAIEX_DEVIATION
    breadth_bearish = oversold_ratio is not None and oversold_ratio >= RISK_OFF_OVERSOLD_RATIO

    if taiex_bearish and breadth_bearish:
        classification = "RISK_OFF"
        reason = (
            f"加權指數跌破20日均線{abs(taiex_deviation) * 100:.1f}%，"
            f"且全市場異常標的中{oversold_ratio * 100:.0f}%為超賣——"
            "大盤趨勢與市場寬度同時確認系統性賣壓，暫停新增買進訊號。"
        )
    elif breadth_bearish and taiex_deviation is None:
        # 抓不到大盤指數時（例如 API 限流），寬度單獨極端偏空也視為風險規避，保守起見。
        classification = "RISK_OFF"
        reason = f"加權指數資料缺失，但全市場{oversold_ratio * 100:.0f}%異常標的為超賣，單獨已達風險規避門檻。"
    elif (
        taiex_deviation is not None and taiex_deviation > 0
        and oversold_ratio is not None and oversold_ratio <= RISK_ON_OVERSOLD_RATIO
    ):
        classification = "RISK_ON"
        reason = "加權指數站上20日均線，且市場寬度未明顯偏空，屬正常風險偏好環境。"
    else:
        classification = "NEUTRAL"
        reason = "大盤趨勢或市場寬度未達極端門檻，維持正常分析流程但保持謹慎。"

    return MarketRegime(
        classification=classification,
        taiex_price=taiex_price,
        taiex_ma20=taiex_ma20,
        taiex_deviation=taiex_deviation,
        oversold_count=oversold_count,
        overbought_count=overbought_count,
        oversold_ratio=oversold_ratio,
        reason=reason,
    )


def apply_regime_policy(enriched: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
    """把體制判別套用到已經算好 Edge Score 的異常清單上。

    刻意不直接刪除任何標的（保留透明度，分析者仍看得到全貌），而是明確標記：
    - buy_eligible=False：RISK_OFF 時，原本的做多候選（OVERSOLD）被標記為
      「禁止建議買進」，不論個股層級查到多扎實的再評價證據。
    - short_candidate=True：RISK_OFF 時，OVERBOUGHT 標的被標記為「值得研究
      放空/賣出機會」——大盤系統性下殺時，逆勢還超買的標的更可能是補跌
      候選，而不是該追高搶反彈的對象。
    """
    enriched = enriched.copy()
    is_risk_off = regime.classification == "RISK_OFF"
    enriched["buy_eligible"] = ~(is_risk_off & (enriched["signal"] == "OVERSOLD"))
    enriched["short_candidate"] = is_risk_off & (enriched["signal"] == "OVERBOUGHT")
    return enriched
