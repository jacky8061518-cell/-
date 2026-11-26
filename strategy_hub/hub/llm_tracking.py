"""
hub/llm_tracking.py
LLM 分析層（tw_trend_core 每日排程寫入的 Buy/Hold/Sell 訊號）的「前瞻」績效追蹤。

為什麼跟規則型 sleeve 分開：LLM 的訓練資料已經看過歷史，拿它回測過去等於偷看答案，
所以 LLM 層只能從訊號發出那天往後追蹤。

每筆可交易訊號（Buy/Sell 且有進場價）依當時給的停損／停利往後走：
- 只有收盤價，以「收盤價觸及」近似停損停利（崩盤時盤中低點更低，實際虧損可能更重）
- 同時計算同一段持有期間 0050 的報酬，衡量「這筆決策有沒有勝過什麼都不做只買大盤」
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hub.costs import ROUND_TRIP_COST


def load_signal_docs(directory: Path) -> list[dict]:
    docs = []
    for f in sorted(Path(directory).rglob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d = d.get("data", d)
        d.setdefault("code", f.stem.split("_")[-1])
        docs.append(d)
    return docs


def simulate_trades(docs: list[dict], prices: pd.DataFrame, bench: pd.Series,
                    code_to_ticker: dict[str, str]) -> pd.DataFrame:
    rows = []
    for s in docs:
        action = s.get("action")
        if action not in ("Buy", "Sell") or s.get("entry") in (None, 0):
            continue
        ticker = code_to_ticker.get(str(s.get("code")))
        if ticker not in prices.columns:
            continue
        created = pd.Timestamp(s["created_at"]).tz_localize(None).normalize()
        path = prices[ticker].dropna()
        path = path[path.index >= created]
        if path.empty:
            continue
        side = 1.0 if action == "Buy" else -1.0
        entry = float(s["entry"])
        sl, tp = s.get("stop_loss"), s.get("take_profit")
        exit_px, exit_dt, reason = float(path.iloc[-1]), path.index[-1], "open"
        for dt, px in path.items():
            if sl is not None and (px - float(sl)) * side <= 0:
                exit_px, exit_dt, reason = px, dt, "stop_loss"
                break
            if tp is not None and (px - float(tp)) * side >= 0:
                exit_px, exit_dt, reason = px, dt, "take_profit"
                break
        ret = side * (exit_px / entry - 1) - (ROUND_TRIP_COST if reason != "open" else ROUND_TRIP_COST / 2)
        b = bench[(bench.index >= created) & (bench.index <= exit_dt)].dropna()
        bench_ret = float(b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else 0.0
        rows.append({
            "code": str(s.get("code")), "name": s.get("name"), "action": action,
            "confidence": s.get("confidence"), "created": created.date().isoformat(),
            "exit_date": exit_dt.date().isoformat(), "exit_reason": reason,
            "return": ret, "bench_return": bench_ret, "excess_vs_0050": ret - bench_ret,
            "regime_blocked": "體制過濾器" in (s.get("reasoning") or ""),
        })
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    buckets = pd.cut(trades["confidence"].astype(float), [0, 0.4, 0.5, 0.6, 1.0],
                     labels=["<=0.4", "0.4-0.5", "0.5-0.6", ">0.6"])
    by_conf = trades.groupby(buckets, observed=False)["return"].agg(["count", "mean"])
    return {
        "n_trades": int(len(trades)),
        "mean_return": float(trades["return"].mean()),
        "median_return": float(trades["return"].median()),
        "win_rate": float((trades["return"] > 0).mean()),
        "mean_excess_vs_0050": float(trades["excess_vs_0050"].mean()),
        "beat_0050_rate": float((trades["excess_vs_0050"] > 0).mean()),
        "stop_loss_rate": float((trades["exit_reason"] == "stop_loss").mean()),
        "by_confidence": {str(k): {"count": int(v["count"]),
                                   "mean_return": None if pd.isna(v["mean"]) else float(v["mean"])}
                          for k, v in by_conf.iterrows()},
        "confidence_return_corr": float(trades["confidence"].astype(float).corr(trades["return"]))
        if len(trades) > 2 else None,
        "first_signal": trades["created"].min(),
        "last_signal": trades["created"].max(),
    }
