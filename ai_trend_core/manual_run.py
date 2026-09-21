"""
manual_run.py
手動單次執行腳本：只掃描一次全市場、若發現異常則啟動 AI 智能體分析，
將完整報告顯示在終端機並寫入資料庫，執行完畢即結束（不像 main_loop.py
會無限循環），適合每天自行決定何時查看報告的使用方式。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import database
from core.quant_engine import DEFAULT_SYMBOLS, MarketSnapshot, get_anomalies, scan_market
from main_loop import handle_anomaly, logger


def _print_board(snapshots: dict[str, MarketSnapshot]) -> None:
    print("\n=== 市場掃描結果 ===")
    for symbol, snap in snapshots.items():
        print(f"{symbol:<8} 價格={snap.price:>12,.2f}  Z-Score={snap.z_score:+.2f}  訊號={snap.signal}")


def _print_report(snapshot: MarketSnapshot, result: dict[str, Any]) -> None:
    confidence = result.get("confidence")
    print(f"\n{'=' * 60}")
    print(f"🤖 AI 分析報告：{snapshot.symbol}（{snapshot.signal}，Z-Score={snapshot.z_score:+.2f}）")
    print(f"{'=' * 60}")
    print(f"建議動作：{result.get('action') or 'N/A'}")
    print(f"進場：{result.get('entry')}　停利：{result.get('take_profit')}　停損：{result.get('stop_loss')}")
    print(f"信心評分：{confidence:.0%}" if confidence is not None else "信心評分：N/A")
    print(f"{'-' * 60}")
    print(result.get("final_report") or "（無完整推理內容）")


def main(symbols: list[str] | None = None) -> None:
    database.init_db()
    logger.info("手動觸發單次掃描（不進入 24/7 循環）...")

    snapshots = scan_market(symbols or DEFAULT_SYMBOLS)
    _print_board(snapshots)

    anomalies = get_anomalies(snapshots)
    if not anomalies:
        print("\n本次掃描未發現異常訊號，無需啟動 AI 智能體分析。")
        return

    print(f"\n偵測到 {len(anomalies)} 個異常標的，開始逐一啟動 AI 智能體分析...")
    analyzed_count = 0
    for snapshot in anomalies:
        result = handle_anomaly(snapshot)
        if result is None:
            continue
        _print_report(snapshot, result)
        analyzed_count += 1

    print(f"\n本次共分析 {analyzed_count} / {len(anomalies)} 個異常標的，結果已寫入 data/trading_signals.db。")


if __name__ == "__main__":
    main()
