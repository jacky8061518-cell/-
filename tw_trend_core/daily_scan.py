"""
daily_scan.py
台股全市場每日掃描腳本（純量化，不需要 Anthropic API Key）：
對全部上市 + 上櫃個股（約 1,900 多檔）一次性計算 Z-Score，
印出所有 |Z-Score| > 2 的異常標的，並寫入 data/latest_tw_scan.json，
供「手動智慧模式」交給 Claude Code 分析（做法與 ai_trend_core/daily_scan.py 相同）。

註：全市場掃描需要抓取近期價格資料，執行時間視網路狀況通常需要數分鐘。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.quant_engine import ZSCORE_THRESHOLD, compute_latest_zscores, get_anomalies, load_price_history
from core.universe import TwStock, load_stock_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tw_trend_core")

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "latest_tw_scan.json"


def build_anomaly_payload(anomalies_df, stocks_by_ticker: dict[str, TwStock]) -> list[dict[str, Any]]:
    """把異常標的的計算結果轉成適合複製貼給 Claude Code 分析的結構化資料。"""
    payload = []
    for ticker, row in anomalies_df.iterrows():
        stock = stocks_by_ticker.get(ticker)
        payload.append({
            "code": stock.code if stock else ticker,
            "yahoo_ticker": ticker,
            "name": stock.name if stock else ticker,
            "market": stock.market if stock else None,
            "industry": stock.industry if stock else None,
            "price": round(float(row["price"]), 2),
            "mean": round(float(row["mean"]), 2),
            "upper_band": round(float(row["upper_band"]), 2),
            "lower_band": round(float(row["lower_band"]), 2),
            "std_dev": round(float(row["std"]), 4),
            "z_score": round(float(row["z_score"]), 4),
            "signal": row["signal"],
        })
    return payload


def main() -> None:
    started = time.monotonic()
    stocks = load_stock_universe()
    tickers = [s.yahoo_ticker for s in stocks]
    stocks_by_ticker = {s.yahoo_ticker: s for s in stocks}
    logger.info("台股全市場掃描開始，共 %d 檔上市 + 上櫃個股（可能需要數分鐘）...", len(tickers))

    prices = load_price_history(tickers)
    latest = compute_latest_zscores(prices)
    logger.info(
        "價格資料截至 %s，成功計算 Z-Score 的標的共 %d 檔（其餘可能為新股，歷史不足 20 個交易日）。",
        prices.index[-1].date(), len(latest),
    )

    anomalies = get_anomalies(latest, threshold=ZSCORE_THRESHOLD)
    payload = build_anomaly_payload(anomalies, stocks_by_ticker)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "price_date": str(prices.index[-1].date()),
                "universe_size": len(tickers),
                "computed_size": len(latest),
                "zscore_threshold": ZSCORE_THRESHOLD,
                "anomalies": payload,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - started
    print(f"\n=== 台股全市場掃描完成（耗時 {elapsed:.0f} 秒）===")
    print(f"價格資料日期：{prices.index[-1].date()}")
    print(f"掃描標的數：{len(tickers)}　成功計算：{len(latest)}")
    print(f"|Z-Score| > {ZSCORE_THRESHOLD} 異常標的數：{len(payload)}\n")

    if not payload:
        print("（本次掃描無異常標的，無需進一步分析。）")
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"\n請把上面這段 JSON 複製貼給 Claude Code，或請它直接讀取 {OUTPUT_PATH}，"
        "由它扮演首席策略官挑選其中幾檔進行分析、查詢新聞。"
        "（標的數可能較多，建議先請 Claude Code 依 |Z-Score| 或你關心的產業篩選出幾檔即可，"
        "不需要對全部異常標的都做完整新聞分析。）"
    )


if __name__ == "__main__":
    main()
