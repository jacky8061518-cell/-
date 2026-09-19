"""趨勢核心選股器：掃一批真實台股，讓多個規則型「偵測 agent」各自標記。

使用者參考了一個外部的 TW 趨勢篩選器網站，想要「很多 agent 在偵測」的畫面。
這支腳本先用確定性規則模擬多個獨立偵測器（而不是 LLM），理由跟 CLAUDE.md
第一條一致——掃描層要的是快、便宜、可解釋，LLM 不該進這條路徑；LLM 分析
留給已經篩出來的候選標的（見 scripts/daily_run.py 的 agent 分析層）。

每個「偵測 agent」都是一條獨立、可解釋的規則，對應到 src/trading_intel
既有的量化函式，不是另外發明的黑盒分數：

- TrendAgent：classify_trend（短長期均線相對位置）
- BreakoutAgent：現價是否為近 60 日新高／新低
- VolumeAgent：今日成交量相對 20 日均量的比值
- MomentumAgent：12-1 動量分數（跳過最近 21 日的滾動報酬）
- VolatilityAgent：classify_volatility（EWMA 年化波動率分層）
- DeviationAgent：現價乖離 60 日均線的百分比（乖離過大視為均值回歸候選）

用法：``uv run python scripts/screener_scan.py > /tmp/screener.json``
"""

from __future__ import annotations

import json

import numpy as np

from trading_intel.core.clock import utc_now
from trading_intel.ingestion.yahoo import fetch_chart, parse_chart
from trading_intel.models.regime import classify_trend, classify_volatility

#: 涵蓋主要產業的台股觀察名單。Yahoo 代號慣例：上市 .TW，上櫃 .TWO。
WATCHLIST: dict[str, tuple[str, str]] = {
    "2330.TW": ("台積電", "半導體"),
    "2454.TW": ("聯發科", "半導體"),
    "3711.TW": ("日月光投控", "半導體"),
    "2303.TW": ("聯電", "半導體"),
    "3034.TW": ("聯詠", "半導體"),
    "2317.TW": ("鴻海", "電子代工"),
    "2382.TW": ("廣達", "電子代工"),
    "2357.TW": ("華碩", "電子代工"),
    "2308.TW": ("台達電", "電子零組件"),
    "2327.TW": ("國巨", "電子零組件"),
    "3008.TW": ("大立光", "光學元件"),
    "3406.TW": ("玉晶光", "光學元件"),
    "2412.TW": ("中華電", "電信"),
    "3045.TW": ("台灣大", "電信"),
    "2881.TW": ("富邦金", "金融"),
    "2882.TW": ("國泰金", "金融"),
    "2891.TW": ("中信金", "金融"),
    "2886.TW": ("兆豐金", "金融"),
    "1301.TW": ("台塑", "傳產"),
    "1303.TW": ("南亞", "傳產"),
    "2002.TW": ("中鋼", "傳產"),
    "2603.TW": ("長榮", "航運"),
    "2609.TW": ("陽明", "航運"),
    "2615.TW": ("萬海", "航運"),
    "1216.TW": ("統一", "食品"),
    "2912.TW": ("統一超", "零售"),
    "2207.TW": ("和泰車", "汽車"),
    "6505.TW": ("台塑化", "能源"),
    "4904.TW": ("遠傳", "電信"),
    "2379.TW": ("瑞昱", "半導體"),
}

BREAKOUT_WINDOW = 60
VOLUME_LOOKBACK = 20
MOMENTUM_LOOKBACK = 180
MOMENTUM_SKIP = 21
DEVIATION_MA_WINDOW = 60
DEVIATION_ALERT_PCT = 0.15  # 乖離超過 15% 視為過大
VOLUME_SURGE_RATIO = 1.8  # 成交量超過均量 1.8 倍視為異常


def scan_one(ticker: str, name: str, sector: str) -> dict[str, object] | None:
    payload, _ = fetch_chart(ticker, range_="1y", interval="1d")
    bars = parse_chart(payload, ticker)
    if len(bars) < 65:
        return None

    closes = np.array([float(b.close) for b in bars])
    volumes = np.array([float(b.volume) for b in bars])
    dates = [b.trade_date for b in bars]
    returns = np.diff(closes) / closes[:-1]

    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    day_change = last_close / prev_close - 1

    trend = classify_trend(closes).value
    volatility_state = classify_volatility(returns).value

    window = closes[-BREAKOUT_WINDOW:]
    is_new_high = bool(last_close >= window.max())
    is_new_low = bool(last_close <= window.min())

    avg_volume = float(volumes[-VOLUME_LOOKBACK - 1 : -1].mean())
    volume_ratio = float(volumes[-1] / avg_volume) if avg_volume > 0 else 0.0

    momentum_score: float | None = None
    if len(closes) > MOMENTUM_SKIP + MOMENTUM_LOOKBACK:
        end = -1 - MOMENTUM_SKIP
        start = end - MOMENTUM_LOOKBACK
        momentum_score = float(closes[end] / closes[start] - 1)

    ma60 = float(closes[-DEVIATION_MA_WINDOW:].mean())
    deviation_pct = float(last_close / ma60 - 1) if ma60 > 0 else 0.0

    detections: list[dict[str, str]] = []
    if trend == "TRENDING_UP":
        detections.append({"agent": "TrendAgent", "label": "趨勢向上確認", "severity": "info"})
    elif trend == "TRENDING_DOWN":
        detections.append({"agent": "TrendAgent", "label": "趨勢向下確認", "severity": "warn"})
    if is_new_high:
        detections.append(
            {"agent": "BreakoutAgent", "label": f"{BREAKOUT_WINDOW} 日新高", "severity": "good"}
        )
    if is_new_low:
        detections.append(
            {"agent": "BreakoutAgent", "label": f"{BREAKOUT_WINDOW} 日新低", "severity": "crit"}
        )
    if volume_ratio >= VOLUME_SURGE_RATIO:
        detections.append(
            {
                "agent": "VolumeAgent",
                "label": f"量能異常放大 {volume_ratio:.1f}x",
                "severity": "warn",
            }
        )
    if momentum_score is not None and momentum_score > 0.15:
        detections.append({"agent": "MomentumAgent", "label": "動能轉強", "severity": "good"})
    elif momentum_score is not None and momentum_score < -0.15:
        detections.append({"agent": "MomentumAgent", "label": "動能轉弱", "severity": "crit"})
    if volatility_state in {"HIGH", "EXTREME"}:
        detections.append(
            {
                "agent": "VolatilityAgent",
                "label": f"波動狀態：{volatility_state}",
                "severity": "warn",
            }
        )
    elif volatility_state == "LOW":
        detections.append(
            {"agent": "VolatilityAgent", "label": "波動壓縮（低波動盤整）", "severity": "info"}
        )
    if abs(deviation_pct) >= DEVIATION_ALERT_PCT:
        direction = "正乖離過大" if deviation_pct > 0 else "負乖離過大"
        detections.append(
            {
                "agent": "DeviationAgent",
                "label": f"{direction} {deviation_pct:+.1%}（乖離 60 日均線）",
                "severity": "warn",
            }
        )

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "last_close": last_close,
        "day_change": day_change,
        "trend": trend,
        "volatility_state": volatility_state,
        "volume_ratio": volume_ratio,
        "momentum_score": momentum_score,
        "deviation_pct": deviation_pct,
        "is_new_high": is_new_high,
        "is_new_low": is_new_low,
        "detections": detections,
        "closes_tail": [float(c) for c in closes[-90:]],
        "dates_tail": [d.isoformat() for d in dates[-90:]],
    }


def main() -> dict[str, object]:
    asof = utc_now()
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for ticker, (name, sector) in WATCHLIST.items():
        try:
            result = scan_one(ticker, name, sector)
        except Exception as exc:
            errors.append({"ticker": ticker, "detail": str(exc)})
            continue
        if result is not None:
            rows.append(result)

    agent_counts: dict[str, int] = {}
    for row in rows:
        detections = row["detections"]
        if isinstance(detections, list):
            for d in detections:
                agent_counts[d["agent"]] = agent_counts.get(d["agent"], 0) + 1

    return {
        "generated_at": asof.isoformat(),
        "universe_size": len(WATCHLIST),
        "scanned": len(rows),
        "errors": errors,
        "agent_counts": agent_counts,
        "rows": rows,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            main(),
            ensure_ascii=False,
            indent=2,
            default=lambda o: None if isinstance(o, float) and (o != o) else o,
        )
    )
