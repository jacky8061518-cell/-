"""
daily_scan.py
台股全市場每日掃描腳本（純量化，不需要 Anthropic API Key）：
對全部上市 + 上櫃個股（約 1,900 多檔）一次性計算 Z-Score，篩出
|Z-Score| > 2 的異常標的後，疊加四個強化因子重新排序：

1. 趨勢/震盪制度（Kaufman Efficiency Ratio）：這檔是真的在趨勢中，
   還是原地震盪？前者高 Z-Score 代表動能延續，不該套用均值回歸邏輯。
2. 產業相對 Z-Score：這是整個產業在動，還是這檔個股特別異常？
3. 成交量 Z-Score：價格異常有沒有放量佐證？
4. 流動性：日均成交金額過低的標的直接排除，避免清單被雜訊股佔滿。

最後合成一個 0-100 的 Edge Score 取代單純依 |Z-Score| 排序，寫入
data/latest_tw_scan.json，供「手動智慧模式」交給 Claude Code 分析
（做法與 ai_trend_core/daily_scan.py 相同）。

註：全市場掃描需要抓取近期價格與成交量資料，執行時間視網路狀況通常需要數分鐘。
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

from core.factors import (
    apply_liquidity_filter,
    attach_industry_relative_zscore,
    classify_regime,
    compute_edge_score,
    compute_industry_median_zscores,
    compute_liquidity,
    compute_volume_zscore,
    load_volume_history,
)
from core.quant_engine import ZSCORE_THRESHOLD, compute_latest_zscores, get_anomalies, load_price_history
from core.universe import TwStock, load_stock_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tw_trend_core")

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "latest_tw_scan.json"


def enrich_anomalies(anomalies_df, full_latest_df, prices, industry_by_ticker: dict[str, str]):
    """對已篩出的異常標的疊加趨勢制度、產業相對 Z-Score、成交量與流動性因子，
    計算 Edge Score 並套用流動性篩選。回傳 (enriched_df, filtered_out_count)。
    """
    enriched = anomalies_df.copy()
    enriched["industry"] = [industry_by_ticker.get(t) for t in enriched.index]

    industry_median = compute_industry_median_zscores(full_latest_df, industry_by_ticker)
    enriched = attach_industry_relative_zscore(enriched, industry_median, industry_by_ticker)

    anomaly_tickers = enriched.index.tolist()
    regime_df = classify_regime(prices[anomaly_tickers])
    enriched = enriched.join(regime_df)

    volumes = load_volume_history(anomaly_tickers)
    if volumes.empty:
        logger.warning("成交量資料抓取失敗（可能是網路問題），本次跳過量能因子與流動性篩選。")
        enriched["volume_z_score"] = float("nan")
        enriched["avg_daily_turnover"] = float("nan")
        enriched["edge_score"] = compute_edge_score(enriched)
        return enriched.sort_values("edge_score", ascending=False), 0

    volume_z = compute_volume_zscore(volumes)
    enriched = enriched.join(volume_z[["volume_z_score", "avg_volume"]])
    enriched["avg_daily_turnover"] = compute_liquidity(enriched["avg_volume"], enriched["price"])
    enriched["edge_score"] = compute_edge_score(enriched)

    if enriched["avg_daily_turnover"].notna().sum() == 0:
        logger.warning("所有標的的成交量資料皆缺失，本次跳過流動性篩選以避免誤刪全部異常標的。")
        return enriched.sort_values("edge_score", ascending=False), 0

    before = len(enriched)
    enriched = apply_liquidity_filter(enriched)
    filtered_out = before - len(enriched)
    return enriched.sort_values("edge_score", ascending=False), filtered_out


def build_anomaly_payload(anomalies_df, stocks_by_ticker: dict[str, TwStock]) -> list[dict[str, Any]]:
    """把（已強化排序的）異常標的轉成適合複製貼給 Claude Code 分析的結構化資料。"""
    payload = []
    for ticker, row in anomalies_df.iterrows():
        stock = stocks_by_ticker.get(ticker)

        def _round_or_none(value, digits=2):
            return None if value is None or (isinstance(value, float) and value != value) else round(float(value), digits)

        payload.append({
            "code": stock.code if stock else ticker,
            "yahoo_ticker": ticker,
            "name": stock.name if stock else ticker,
            "market": stock.market if stock else None,
            "industry": stock.industry if stock else None,
            "price": _round_or_none(row["price"]),
            "mean": _round_or_none(row["mean"]),
            "upper_band": _round_or_none(row["upper_band"]),
            "lower_band": _round_or_none(row["lower_band"]),
            "std_dev": _round_or_none(row["std"], 4),
            "z_score": _round_or_none(row["z_score"], 4),
            "signal": row["signal"],
            "regime": row.get("regime"),
            "trend_efficiency": _round_or_none(row.get("trend_efficiency"), 3),
            "industry_median_z": _round_or_none(row.get("industry_median_z"), 3),
            "industry_relative_z": _round_or_none(row.get("industry_relative_z"), 3),
            "volume_z_score": _round_or_none(row.get("volume_z_score"), 3),
            "avg_daily_turnover": _round_or_none(row.get("avg_daily_turnover"), 0),
            "edge_score": _round_or_none(row.get("edge_score"), 1),
        })
    return payload


def main() -> None:
    started = time.monotonic()
    stocks = load_stock_universe()
    tickers = [s.yahoo_ticker for s in stocks]
    stocks_by_ticker = {s.yahoo_ticker: s for s in stocks}
    industry_by_ticker = {s.yahoo_ticker: s.industry for s in stocks}
    logger.info("台股全市場掃描開始，共 %d 檔上市 + 上櫃個股（可能需要數分鐘）...", len(tickers))

    prices = load_price_history(tickers)
    latest = compute_latest_zscores(prices)
    logger.info(
        "價格資料截至 %s，成功計算 Z-Score 的標的共 %d 檔（其餘可能為新股，歷史不足 20 個交易日）。",
        prices.index[-1].date(), len(latest),
    )

    anomalies = get_anomalies(latest, threshold=ZSCORE_THRESHOLD)
    filtered_out = 0
    if not anomalies.empty:
        logger.info("計算趨勢制度、產業相對 Z-Score、成交量與流動性因子（僅對 %d 檔異常標的）...", len(anomalies))
        anomalies, filtered_out = enrich_anomalies(anomalies, latest, prices, industry_by_ticker)

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
                "liquidity_filtered_count": filtered_out,
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
    print(f"|Z-Score| > {ZSCORE_THRESHOLD} 異常標的數：{len(payload)}（另有 {filtered_out} 檔因流動性不足被排除）\n")

    if not payload:
        print("（本次掃描無異常標的，無需進一步分析。）")
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"\n已依 Edge Score（綜合統計極端度、量能確認、流動性）由高到低排序。"
        f"請把上面這段 JSON 複製貼給 Claude Code，或請它直接讀取 {OUTPUT_PATH}，"
        "由它扮演首席策略官挑選其中幾檔進行分析、查詢新聞。"
        "（regime 為 TRENDING_UP/TRENDING_DOWN 的標的代表動能延續訊號，"
        "不宜套用「等回檔至均值」的均值回歸邏輯；RANGE_BOUND 才是傳統均值回歸的合理場景。）"
    )


if __name__ == "__main__":
    main()
