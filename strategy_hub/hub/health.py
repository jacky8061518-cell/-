"""
hub/health.py
跨專案營運監控：把 repo 裡所有模組的「現在還活著嗎、資料新不新鮮」放在同一個地方。

狀態只有三種：ok / warn / fail，每一項都附上判斷依據，不給看不懂的燈號。

外部狀態（排程最後一次執行結果、Artifact 資料庫中的最新掃描時間）無法從這支腳本
直接查到，由呼叫端（每日排程）以 ops_status JSON 傳入；沒傳就標示為 unknown，
而不是假裝一切正常。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hub.data import REPO_ROOT

STALE_WARN_DAYS = 3   # 交易日
STALE_FAIL_DAYS = 10


def _bdays_between(a: date, b: date) -> int:
    return int(np.busday_count(a, b)) if a < b else 0


def _freshness(component: str, last: date | None, today: date, what: str) -> dict:
    if last is None:
        return {"component": component, "status": "fail", "detail": f"找不到{what}"}
    lag = _bdays_between(last, today)
    status = "ok" if lag <= STALE_WARN_DAYS else "warn" if lag <= STALE_FAIL_DAYS else "fail"
    return {"component": component, "status": status,
            "detail": f"{what}最後日期 {last.isoformat()}（落後 {lag} 個交易日）", "last": last.isoformat()}


def check_price_panel(prices: pd.DataFrame, universe_today: pd.Series, glitches_30d: int, today: date) -> list[dict]:
    out = [_freshness("TW 價格面板（tw_trend_core 快取）", prices.index[-1].date(), today, "價格資料")]
    raw_last = prices.iloc[-1][universe_today[universe_today].index]
    missing = float(raw_last.isna().mean()) if len(raw_last) else 1.0
    out.append({
        "component": "資料品質：股票池缺值",
        "status": "ok" if missing < 0.02 else "warn" if missing < 0.10 else "fail",
        "detail": f"今日股票池 {len(raw_last)} 檔中 {missing:.1%} 無報價",
    })
    out.append({
        "component": "資料品質：異常報酬截斷",
        "status": "ok" if glitches_30d == 0 else "warn",
        "detail": f"近 30 日有 {glitches_30d} 筆單日報酬超過 ±40% 被視為資料錯誤截斷",
    })
    return out


def check_stale_bars(prices: pd.DataFrame, universe_today: pd.Series, bench: pd.Series,
                     index_source: str) -> list[dict]:
    """價格面板經過前值填補，缺資料不會顯示成缺值——要看「收盤價跟前一天一模一樣」才抓得到。"""
    out = []
    names = universe_today[universe_today].index
    unchanged = float((prices[names].iloc[-1] == prices[names].iloc[-2]).mean()) if len(names) else 0.0
    out.append({
        "component": "資料品質：疑似前值填補",
        "status": "ok" if unchanged < 0.15 else "warn" if unchanged < 0.40 else "fail",
        "detail": f"股票池最新收盤與前一日完全相同的比例 {unchanged:.1%}（大型股正常應很低）",
    })
    if bench.iloc[-1] == bench.iloc[-2]:
        out.append({"component": "資料品質：0050 基準", "status": "warn",
                    "detail": f"0050 最新收盤與前一日完全相同（{bench.index[-1].date()}），疑似資料源缺漏被前值填補"})
    out.append({"component": "大盤體制指數來源",
                "status": "ok" if index_source == "^TWII" else "warn",
                "detail": "加權指數 ^TWII（與即時版一致）" if index_source == "^TWII"
                else "抓不到 ^TWII，暫以 0050 代理——與即時版可能不一致"})
    return out


def check_shared_seed(today: date) -> list[dict]:
    tw = REPO_ROOT / "data" / "databases" / "tw"
    out = []
    try:
        seed = pd.read_parquet(tw / "adjusted-prices.parquet")
        out.append(_freshness("sector-rotation-research 共用價格種子", pd.to_datetime(seed.index.max()).date(),
                              today, "共用種子"))
    except Exception as exc:  # noqa: BLE001
        out.append({"component": "sector-rotation-research 共用價格種子", "status": "fail", "detail": str(exc)})
    try:
        flows = pd.read_parquet(tw / "institutional-flows.parquet", columns=["Date"])
        out.append(_freshness("sector-rotation-research 三大法人資料", pd.to_datetime(flows["Date"]).max().date(),
                              today, "法人買賣超"))
    except Exception as exc:  # noqa: BLE001
        out.append({"component": "sector-rotation-research 三大法人資料", "status": "fail", "detail": str(exc)})
    return out


def check_trading_intel(today: date) -> list[dict]:
    runs = sorted((REPO_ROOT / "trading-intel" / "runs").glob("*.json"))
    if not runs:
        return [{"component": "trading-intel 每日執行", "status": "fail", "detail": "沒有任何執行紀錄"}]
    latest = json.loads(runs[-1].read_text(encoding="utf-8"))
    out = [_freshness("trading-intel 每日執行", date.fromisoformat(runs[-1].stem), today, "執行紀錄")]
    agents = latest.get("agents", {})
    if agents.get("status") == "skipped":
        out.append({"component": "trading-intel LLM 代理人層", "status": "warn",
                    "detail": agents.get("reason", "代理人層未執行")})
    pretrade = latest.get("pretrade", {})
    if pretrade and not pretrade.get("approved", True):
        out.append({"component": "trading-intel 交易前風控", "status": "warn",
                    "detail": f"最後一次執行的交易前檢查未通過，違反 {len(pretrade.get('breached_limits', []))} 項限制"})
    return out


def check_ops_status(ops: dict | None) -> list[dict]:
    if not ops:
        return [{"component": "tw_trend_core 每日排程", "status": "unknown",
                 "detail": "本次未提供排程狀態（ops_status），無法判斷"}]
    out = []
    run = ops.get("tw_routine_last_run") or {}
    st = run.get("status", "")
    out.append({
        "component": "tw_trend_core 每日排程",
        "status": "ok" if st.endswith("SUCCEEDED") else "fail" if st else "unknown",
        "detail": f"最後執行 {run.get('fired_at', '?')}，結果 {st or '未知'}",
    })
    for item in ops.get("extra_checks", []):
        out.append(item)
    return out


def static_notes() -> list[dict]:
    return [{"component": "ai_trend_core（美股／加密）", "status": "paused",
             "detail": "依使用者指示暫停，先專注台股；程式碼保留未刪除"}]


def overall(checks: list[dict]) -> str:
    states = {c["status"] for c in checks}
    return "fail" if "fail" in states else "warn" if "warn" in states else "ok"
