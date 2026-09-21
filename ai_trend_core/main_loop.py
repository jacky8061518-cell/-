"""
main_loop.py
AI Trend Core 24/7 循環執行腳本：
每 15 分鐘掃描一次市場，若發現異常訊號則觸發 AI 智能體深度分析，並將結果存入資料庫。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.parsing import summarize_reasoning
from core import database, notifier
from core.quant_engine import DEFAULT_SYMBOLS, MarketSnapshot, get_anomalies, scan_market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai_trend_core")

SCAN_INTERVAL_SECONDS = 15 * 60

# 信心評分超過此門檻時，主動透過 Telegram 推播警報
TELEGRAM_CONFIDENCE_THRESHOLD = 0.8


def _handle_anomaly(snapshot: MarketSnapshot) -> None:
    """對單一異常標的啟動 AI 智能體分析，並將結果寫入資料庫。"""
    from agents.trading_crew import analyze_opportunity  # 延遲載入，避免無異常時仍初始化 LLM

    logger.info("偵測到異常：%s（%s），啟動 AI 智能體分析...", snapshot.symbol, snapshot.signal)
    try:
        result = analyze_opportunity(snapshot.symbol, snapshot)
    except Exception:
        logger.exception("分析 %s 時發生錯誤，略過此標的。", snapshot.symbol)
        return

    database.insert_signal(
        symbol=snapshot.symbol,
        signal_type=snapshot.signal,
        price=snapshot.price,
        z_score=snapshot.z_score,
        upper_band=snapshot.upper_band,
        lower_band=snapshot.lower_band,
        mean_price=snapshot.mean,
        action=result.get("action"),
        entry=result.get("entry"),
        take_profit=result.get("take_profit"),
        stop_loss=result.get("stop_loss"),
        confidence=result.get("confidence"),
        reasoning=result.get("final_report"),
        sentiment_summary=result.get("sentiment_report"),
        raw_market_data={
            "mean": snapshot.mean,
            "upper_band": snapshot.upper_band,
            "lower_band": snapshot.lower_band,
            "std_dev": snapshot.std_dev,
        },
    )
    logger.info(
        "已儲存 %s 的 AI 建議：Action=%s Confidence=%s",
        snapshot.symbol, result.get("action"), result.get("confidence"),
    )

    confidence = result.get("confidence")
    if confidence is not None and confidence > TELEGRAM_CONFIDENCE_THRESHOLD:
        message = notifier.format_signal_alert(
            symbol=snapshot.symbol,
            price=snapshot.price,
            z_score=snapshot.z_score,
            action=result.get("action") or "N/A",
            confidence=confidence,
            take_profit=result.get("take_profit"),
            stop_loss=result.get("stop_loss"),
            reasoning_summary=summarize_reasoning(result.get("final_report")),
        )
        if notifier.send_telegram_message(message):
            logger.info("已透過 Telegram 推播 %s 的高信心訊號。", snapshot.symbol)


def run_once(symbols: list[str] | None = None) -> None:
    """執行一次完整的「全市場掃描 + 異常深度分析」流程。"""
    symbols = symbols or DEFAULT_SYMBOLS
    logger.info("開始掃描 %d 個標的：%s", len(symbols), ", ".join(symbols))

    snapshots = scan_market(symbols)
    for symbol, snap in snapshots.items():
        logger.info(
            "%-8s 價格=%.2f  Z-Score=%+.2f  訊號=%s",
            symbol, snap.price, snap.z_score, snap.signal,
        )

    anomalies = get_anomalies(snapshots)
    if not anomalies:
        logger.info("本次掃描未發現異常訊號。")
        return

    for snapshot in anomalies:
        _handle_anomaly(snapshot)


def main() -> None:
    database.init_db()
    logger.info("AI Trend Core 掃描迴圈啟動，每 %d 分鐘執行一次。", SCAN_INTERVAL_SECONDS // 60)
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("本輪掃描發生未預期錯誤。")
        logger.info("本輪結束，休眠 %d 分鐘...", SCAN_INTERVAL_SECONDS // 60)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
