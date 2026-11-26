"""
run_hub.py
Strategy Hub 每日主程式：整合 repo 內所有專案的策略與營運狀態，產出一份報告。

    python strategy_hub/run_hub.py [--ops-status ops.json] [--llm-signals-dir DIR]

輸出 strategy_hub/data/hub_report.json（成績單、Ensemble 現況、營運健康度）與
strategy_hub/data/hub_curves.json（週頻淨值曲線，給儀表板畫圖用）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

HUB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(HUB_ROOT))

from hub import health  # noqa: E402
from hub.backtest import daily_returns, run_backtest  # noqa: E402
from hub.data import BENCHMARK_TICKER, indicators, load_panel, load_security_master, monthly_universe  # noqa: E402
from hub.ensemble import (LOOKBACK, MAX_SLEEVE_WEIGHT, REBALANCE_EVERY, TARGET_VOL,  # noqa: E402
                          run_ensemble, static_equal_weight)
from hub.evaluation import CONCENTRATION_CAP, scorecard  # noqa: E402
from hub.llm_tracking import load_signal_docs, simulate_trades, summarize  # noqa: E402
from hub.regimes import regime_series, threshold_grid  # noqa: E402
from hub.sleeves import SLEEVE_LABELS, SLEEVES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("strategy_hub")

EVAL_START = pd.Timestamp("2016-01-04")
UNIVERSE_SIZE = 300
# 已知的融券限制期間（2026-11-23 起金管會平盤以下禁空、融券保證金提高；結束日未定，視為持續中）
SHORT_BANS = [("2026-11-23", "2099-12-31")]
OUT_DIR = HUB_ROOT / "data"
TAIPEI = ZoneInfo("Asia/Taipei")
INDEX_GAPS: list[str] = []


def weekly_equity(r: pd.Series) -> list[list]:
    eq = (1 + r).cumprod()
    wk = eq.resample("W-FRI").last().dropna()
    return [[d.date().isoformat(), round(float(v), 4)] for d, v in wk.items()]


def top_holdings(weights: pd.DataFrame, master: pd.DataFrame, n: int = 10) -> list[dict]:
    last = weights.iloc[-1]
    last = last[last != 0].sort_values(key=abs, ascending=False).head(n)
    return [{"ticker": t, "code": str(master.at[t, "Code"]) if t in master.index else t.split(".")[0],
             "name": master.at[t, "Name"] if t in master.index else ("元大台灣50" if t == BENCHMARK_TICKER else t),
             "weight": round(float(w), 4)} for t, w in last.items()]


def drop_unfinished_session(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """台股 13:30 收盤；台北時間 14:30 前抓到的「今天」那一列是盤中價，不是收盤價，必須丟掉。"""
    now = now or datetime.now(timezone.utc)
    taipei = now.astimezone(TAIPEI)
    if prices.index[-1].date() == taipei.date() and taipei.hour * 60 + taipei.minute < 14 * 60 + 30:
        log.info("最後一列 %s 為盤中資料（台北 %s），不納入。", prices.index[-1].date(), taipei.strftime("%H:%M"))
        return prices.iloc[:-1]
    return prices


def load_regime_index(dates: pd.DatetimeIndex, fallback: pd.Series) -> tuple[pd.Series, str]:
    """體制用的指數：優先用加權指數 ^TWII（與 tw_trend_core 即時版同一來源）。"""
    from core.regime import fetch_taiex_history  # tw_trend_core

    try:
        twii = fetch_taiex_history(lookback_days=(date.today() - dates[0].date()).days + 30)
        twii.index = pd.to_datetime(twii.index).tz_localize(None).normalize()
        if len(twii) > 500:
            raw = twii.reindex(dates)
            filled = [d.date().isoformat() for d in raw.index[-10:][raw.iloc[-10:].isna()]]
            if filled:
                INDEX_GAPS.extend(filled)
                log.warning("^TWII 缺少 %s 的資料，以前值填補", filled)
            return raw.ffill(limit=3), "^TWII"
    except Exception:  # noqa: BLE001
        log.warning("抓不到 ^TWII，體制改用 0050 代理", exc_info=True)
    return fallback, "0050.TW"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops-status", type=Path)
    ap.add_argument("--llm-signals-dir", type=Path)
    args = ap.parse_args()
    started = time.monotonic()
    today = date.today()

    master = load_security_master()
    prices_all = drop_unfinished_session(load_panel())
    bench_px = prices_all[BENCHMARK_TICKER]
    prices = prices_all[[c for c in prices_all.columns if c in master.index]]
    log.info("價格面板 %s ~ %s，%d 檔個股", prices.index[0].date(), prices.index[-1].date(), prices.shape[1])

    returns, _ = daily_returns(prices)
    glitches_30d = daily_returns(prices.iloc[-31:])[1]
    bench_ret = daily_returns(bench_px.to_frame())[0][BENCHMARK_TICKER]
    bt_returns = returns.assign(**{BENCHMARK_TICKER: bench_ret})  # 0050 只給 market_timing sleeve 用
    ind = indicators(prices)
    universe = monthly_universe(prices, master["Issued shares"], UNIVERSE_SIZE)
    index_px, index_source = load_regime_index(prices.index, bench_px)
    regimes = regime_series(index_px, ind["z"])

    ctx = {"prices": prices, "returns": returns, "universe": universe, "regime": regimes,
           "industry": master["Industry"], "benchmark_ticker": BENCHMARK_TICKER, **ind}

    sleeve_results = {}
    for name, fn in SLEEVES.items():
        log.info("回測 sleeve：%s", name)
        sleeve_results[name] = run_backtest(fn(ctx), bt_returns)

    ew_weights = universe.astype(float).div(universe.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0.0)
    ew = run_backtest(ew_weights, returns)

    sleeve_net = pd.DataFrame({k: v.net for k, v in sleeve_results.items()})
    ens = run_ensemble(sleeve_net, start=EVAL_START)
    static = static_equal_weight(sleeve_net)

    sl = slice(EVAL_START, None)
    # 評估用「前一日收盤已知」的體制切分報酬，避免循環論證（見 evaluation.regime_block）
    reg = regimes.shift(1)[sl]
    bench_eval, ew_eval = bench_ret[sl], ew.net[sl]
    common = dict(regimes=reg, bench=bench_eval, equal_weight=ew_eval, short_bans=SHORT_BANS)

    cards = []
    for name, res in sleeve_results.items():
        w = res.weights[sl]
        entries = (w > 0) & (w.shift(1).fillna(0) <= 0)
        cards.append(scorecard(
            name, SLEEVE_LABELS[name], res.net[sl], gross=res.gross[sl], cost=res.cost[sl],
            turnover=res.turnover[sl], gross_exposure=res.gross_exposure[sl], holdings=res.holdings[sl],
            weights=w, entries=entries, regime_filtered=(name == "mean_reversion_regime"),
            decision_regimes=regimes[sl],
            cap=1.0 if name == "market_timing_0050" else CONCENTRATION_CAP,  # 0050 本身是 50 檔分散的 ETF
            **common))
    expo = pd.DataFrame({k: v.net_exposure for k, v in sleeve_results.items()})
    gexpo = pd.DataFrame({k: v.gross_exposure for k, v in sleeve_results.items()})
    eff_net = (ens.sleeve_weights.shift(1) * expo).sum(axis=1)[sl]
    eff_gross = (ens.sleeve_weights.shift(1) * gexpo).sum(axis=1)[sl]
    cards.append(scorecard("ensemble", "Ensemble Agent（動態配置）", ens.net[sl], kind="ensemble",
                           weights=ens.sleeve_weights[sl], cap=MAX_SLEEVE_WEIGHT,
                           extra={"exposure": {"avg_net_market_exposure": round(float(eff_net.mean()), 4),
                                               "avg_gross_market_exposure": round(float(eff_gross.mean()), 4),
                                               "current_net_market_exposure": round(float(eff_net.iloc[-1]), 4),
                                               "switch_cost_drag_annual": round(float(ens.switch_cost[sl].mean() * 252), 4)}},
                           **{**common, "short_bans": []}))
    cards.append(scorecard("static_equal", "Sleeve 等權重（不調整）", static[sl], kind="baseline", **common))
    cards.append(scorecard("equal_weight_universe", "股票池等權重", ew_eval, gross=ew.gross[sl], cost=ew.cost[sl],
                           turnover=ew.turnover[sl], gross_exposure=ew.gross_exposure[sl],
                           holdings=ew.holdings[sl], kind="baseline", **common))
    cards.append(scorecard("bench_0050", "0050 買進持有", bench_eval, kind="baseline", **common))

    # Ensemble 現況：強者加碼、弱者減碼
    sw = ens.sleeve_weights
    rebal_days = sw.index[sw.diff().abs().sum(axis=1) > 1e-9]
    prev = sw.loc[rebal_days[-2]] if len(rebal_days) >= 2 else sw.iloc[-1] * 0
    ensemble_now = {
        "as_of": sw.index[-1].date().isoformat(),
        "last_rebalance": rebal_days[-1].date().isoformat() if len(rebal_days) else None,
        "cash_weight": round(float(1 - sw.iloc[-1].sum()), 4),
        "params": {"lookback_days": LOOKBACK, "rebalance_every_days": REBALANCE_EVERY,
                   "max_sleeve_weight": MAX_SLEEVE_WEIGHT, "target_vol": TARGET_VOL},
        "sleeves": [{
            "id": k, "label": SLEEVE_LABELS[k],
            "weight": round(float(sw.iloc[-1][k]), 4),
            "prev_weight": round(float(prev[k]), 4),
            "trailing_sharpe": round(float(ens.trailing_sharpe.iloc[-1][k]), 3),
            "holdings": top_holdings(sleeve_results[k].weights, master),
        } for k in SLEEVES],
    }

    z_last = ind["z"].iloc[-1]
    oversold, overbought = int((z_last <= -2).sum()), int((z_last >= 2).sum())
    dev = float(index_px.iloc[-1] / index_px.rolling(20).mean().iloc[-1] - 1)
    regime_now = {
        "date": prices.index[-1].date().isoformat(), "classification": regimes.iloc[-1],
        "index_source": index_source, "index_level": round(float(index_px.iloc[-1]), 2), "index_deviation_ma20": round(dev, 4),
        "oversold": oversold, "overbought": overbought,
        "oversold_ratio": round(oversold / (oversold + overbought), 4) if oversold + overbought else None,
        "history_share": {k: round(float((reg == k).mean()), 4) for k in ("RISK_ON", "NEUTRAL", "RISK_OFF")},
        "robustness_grid": threshold_grid(index_px, ind["z"], bench_ret, EVAL_START),
        "robustness_ex2026": None,
    }
    ex = slice(EVAL_START, pd.Timestamp("2025-12-31"))
    from hub.evaluation import cagr as _cagr, sharpe as _sharpe
    regime_now["robustness_ex2026"] = {
        "filter": {"cagr": round(_cagr(sleeve_results["market_timing_0050"].net[ex]), 4),
                   "sharpe": round(_sharpe(sleeve_results["market_timing_0050"].net[ex]), 3)},
        "bench_0050": {"cagr": round(_cagr(bench_ret[ex]), 4), "sharpe": round(_sharpe(bench_ret[ex]), 3)},
    }

    llm = {"n_trades": 0, "note": "未提供 --llm-signals-dir"}
    if args.llm_signals_dir and args.llm_signals_dir.exists():
        code_to_ticker = {str(c): t for t, c in master["Code"].items()}
        trades = simulate_trades(load_signal_docs(args.llm_signals_dir), prices, bench_px, code_to_ticker)
        llm = summarize(trades)
        llm["trades"] = trades.sort_values("created").to_dict(orient="records")

    ops = json.loads(args.ops_status.read_text(encoding="utf-8")) if args.ops_status else None
    checks = (health.check_price_panel(prices, universe.iloc[-1], glitches_30d, today)
              + health.check_stale_bars(prices, universe.iloc[-1], bench_px, index_source)
              + health.check_shared_seed(today) + health.check_trading_intel(today)
              + health.check_ops_status(ops) + health.static_notes())
    for c in cards:
        cons = c.get("constraints", {})
        if cons.get("new_longs_during_risk_off"):
            checks.append({"component": f"風控自我檢查：{c['label']}", "status": "fail",
                           "detail": f"RISK_OFF 期間出現 {cons['new_longs_during_risk_off']} 筆新多單，硬規則失效"})
        if cons.get("days_concentration_breach"):
            checks.append({"component": f"集中度：{c['label']}", "status": "warn",
                           "detail": f"{cons['days_concentration_breach']} 天單一個股權重超過 5%"})
    if INDEX_GAPS:
        checks.append({"component": "資料品質：加權指數缺日", "status": "warn",
                       "detail": f"Yahoo 缺少 ^TWII {', '.join(INDEX_GAPS)} 的收盤，已用前一日值填補；當日體制判斷可能失真"})
    short_now = (sleeve_results["regime_short"].weights.iloc[-1] < 0).any()
    if short_now:
        checks.append({"component": "限空令：風險規避放空 sleeve", "status": "warn",
                       "detail": "目前持有空單，但 2026-11-23 起平盤以下禁空、融券保證金提高——實盤可能無法建立或成本更高"})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "price_date": prices.index[-1].date().isoformat(),
        "eval_start": EVAL_START.date().isoformat(),
        "universe_size": UNIVERSE_SIZE,
        "regime_now": regime_now,
        "ensemble_now": ensemble_now,
        "scorecards": cards,
        "llm_layer": llm,
        "health": {"overall": health.overall(checks), "checks": checks},
        "caveats": [
            "股票池只含目前仍上市公司（倖存者偏誤），市值代理用目前股數；公平比較請看「vs 股票池等權重」。",
            f"歷史體制指數來源：{index_source}（與 tw_trend_core 即時版相同；抓不到時才用 0050 代理）。",
            "成本：手續費與滑價買賣各 19.25bps、證交稅賣出 30bps、借券費年化 2%，沿用 trading-intel 設定。",
            "LLM 訊號只能前瞻追蹤，不能回測（模型訓練資料已看過歷史）。",
            "Sharpe 以無風險利率 0 計算。",
        ],
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    curves = {k: weekly_equity(v.net[sl]) for k, v in sleeve_results.items()}
    curves.update({"ensemble": weekly_equity(ens.net[sl]), "static_equal": weekly_equity(static[sl]),
                   "equal_weight_universe": weekly_equity(ew_eval), "bench_0050": weekly_equity(bench_eval)})
    alloc = ens.sleeve_weights[sl].resample("W-FRI").last().dropna(how="all")  # 整週休市會產生空列
    curves_doc = {"price_date": report["price_date"], "equity": curves,
                  "ensemble_allocation": {"dates": [d.date().isoformat() for d in alloc.index],
                                          **{k: [round(float(x), 3) for x in alloc[k]] for k in alloc.columns}}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hub_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "hub_curves.json").write_text(json.dumps(curves_doc, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Strategy Hub 完成（{report['runtime_seconds']} 秒），價格日期 {report['price_date']} ===")
    print(f"大盤體制：{regime_now['classification']}　營運健康：{report['health']['overall']}")
    print(f"{'策略':<22}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'成本拖累':>9}{'RISK_OFF日均':>12}")
    for c in cards:
        p, r = c["performance"], c["regime"]["RISK_OFF"]
        cd = (c.get("costs") or {}).get("cost_drag_annual")
        fmt = lambda x, pct=True: "—" if x is None else (f"{x:.1%}" if pct else f"{x:.2f}")  # noqa: E731
        print(f"{c['label']:<20}{fmt(p['cagr']):>8}{fmt(p['sharpe'], False):>8}{fmt(p['max_drawdown']):>8}"
              f"{fmt(cd):>9}{fmt(r['mean_daily_return']):>12}")
    print("\nEnsemble 目前配置：", {s["id"]: s["weight"] for s in ensemble_now["sleeves"]},
          "現金", ensemble_now["cash_weight"])


if __name__ == "__main__":
    main()
