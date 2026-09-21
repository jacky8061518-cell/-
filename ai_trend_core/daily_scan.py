"""
daily_scan.py
每日手動掃描腳本（純量化，不需要 Anthropic API Key，只需要 yfinance）：
掃描全市場、印出所有標的的 Z-Score，並將 |Z-Score| > 2 的異常標的完整數據
寫入 data/latest_scan.json，同時印在終端機。

「手動智慧模式」工作流程：
1. 執行本腳本。
2. 把印出的 JSON 複製貼給 Claude Code，或請 Claude Code 直接讀取
   data/latest_scan.json。
3. 由 Claude Code（使用你的 Claude Pro 額度，而非額外付費的 API）扮演
   「首席策略官」，搭配網路搜尋分析新聞情緒，給出交易建議。
4. 請 Claude Code 把分析結果寫入 core/database.py 的 trading_signals.db，
   以便追蹤歷史勝率。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.quant_engine import DEFAULT_SYMBOLS, MarketSnapshot, ZSCORE_THRESHOLD, get_anomalies, scan_market

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "latest_scan.json"


def build_anomaly_payload(anomalies: list[MarketSnapshot]) -> list[dict[str, Any]]:
    """把異常標的的快照轉成適合複製貼給 Claude Code 分析的結構化資料。"""
    return [
        {
            "symbol": snap.symbol,
            "price": round(snap.price, 2),
            "mean": round(snap.mean, 2),
            "upper_band": round(snap.upper_band, 2),
            "lower_band": round(snap.lower_band, 2),
            "std_dev": round(snap.std_dev, 4),
            "z_score": round(snap.z_score, 4),
            "signal": snap.signal,
        }
        for snap in anomalies
    ]


def main(symbols: list[str] | None = None) -> None:
    symbols = symbols or DEFAULT_SYMBOLS
    print(f"正在掃描 {len(symbols)} 個標的：{', '.join(symbols)}\n")

    snapshots = scan_market(symbols)

    print("=== 全部標的掃描結果 ===")
    for symbol, snap in snapshots.items():
        print(f"{symbol:<8} 價格={snap.price:>12,.2f}  Z-Score={snap.z_score:+.2f}  訊號={snap.signal}")

    anomalies = get_anomalies(snapshots)
    anomaly_payload = build_anomaly_payload(anomalies)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "zscore_threshold": ZSCORE_THRESHOLD,
                "anomalies": anomaly_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n=== |Z-Score| > {ZSCORE_THRESHOLD} 的異常標的（共 {len(anomaly_payload)} 個）===")
    if not anomaly_payload:
        print("（本次掃描無異常標的，無需進一步分析。）")
        return

    print(json.dumps(anomaly_payload, ensure_ascii=False, indent=2))
    print(
        f"\n請把上面這段 JSON 複製貼給 Claude Code，或請它直接讀取 {OUTPUT_PATH}，"
        "由它扮演首席策略官進行分析、查詢新聞，並將結果寫入 trading_signals.db。"
    )


if __name__ == "__main__":
    main()
