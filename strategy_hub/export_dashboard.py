"""
export_dashboard.py
把 run_hub.py 的輸出轉成 Strategy Hub 儀表板的資料檔（每個檔一個 dataset）。

    python strategy_hub/export_dashboard.py

每個檔都是 {rows, meta}：rows 是儀表板讀的資料列，meta 附上產生它的程式碼與資料日期，
讀者在儀表板的 Sources 裡可以看到每個數字是怎麼算出來的。
輸出在 strategy_hub/data/dash/<dataset_id>.json。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parent
DATA = HUB_ROOT / "data"
OUT = DATA / "dash"


def _code(rel: str) -> str:
    return (HUB_ROOT / rel).read_text(encoding="utf-8")


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HUB_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    report = json.loads((DATA / "hub_report.json").read_text(encoding="utf-8"))
    curves = json.loads((DATA / "hub_curves.json").read_text(encoding="utf-8"))
    commit = _commit()
    base_meta = [
        {"label": "資料日期", "value": report["price_date"]},
        {"label": "產生時間", "value": report["generated_at"]},
        {"label": "評估起點", "value": report["eval_start"]},
        {"label": "程式", "value": f"strategy_hub/run_hub.py → export_dashboard.py（commit {commit}）"},
    ]

    def write(name: str, rows: list[dict], code_files: list[str]) -> None:
        meta = base_meta + [{"label": f, "value": _code(f), "type": "code", "lang": "python"} for f in code_files]
        OUT.mkdir(parents=True, exist_ok=True)
        # allow_nan=False：NaN 不是合法 JSON，儀表板會整個讀不進來，寧可在這裡就失敗
        (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False,
                                                     allow_nan=False, separators=(",", ":")), encoding="utf-8")

    labels = {c["id"]: c["label"] for c in report["scorecards"]}

    rows = []
    for c in report["scorecards"]:
        p, co, rg = c["performance"], c.get("costs") or {}, c["regime"]
        v0, ve, cs = c["vs_0050"], c["vs_equal_weight"], c.get("constraints") or {}
        rows.append({
            "id": c["id"], "label": c["label"], "kind": c["kind"],
            "cagr": p["cagr"], "ann_vol": p["ann_vol"], "sharpe": p["sharpe"],
            "max_drawdown": p["max_drawdown"], "current_drawdown": p["current_drawdown"],
            "ret_1y": p["ret_1y"], "ret_3m": p["ret_3m"], "ret_ytd": p["ret_ytd"],
            "gross_cagr": co.get("gross_cagr"), "cost_drag": co.get("cost_drag_annual"),
            "turnover": co.get("turnover_annual"), "avg_exposure": co.get("avg_gross_exposure"),
            "sharpe_on": rg["RISK_ON"]["sharpe"], "sharpe_neutral": rg["NEUTRAL"]["sharpe"],
            "sharpe_off": rg["RISK_OFF"]["sharpe"],
            "mdr_on": rg["RISK_ON"]["mean_daily_return"], "mdr_neutral": rg["NEUTRAL"]["mean_daily_return"],
            "mdr_off": rg["RISK_OFF"]["mean_daily_return"],
            "pct_positive_years": rg["pct_positive_years"], "worst_year": rg["worst_year"],
            "excess_vs_0050": v0["excess_cagr"], "ir_vs_0050": v0["information_ratio"],
            "hit_rate_vs_0050": v0["monthly_hit_rate"], "beta": v0["beta"],
            "ir_vs_ew": ve["information_ratio"],
            "cap": cs.get("cap"), "max_weight": cs.get("max_single_name_weight"),
            "breach_days": cs.get("days_concentration_breach"),
            "short_ban_days": cs.get("days_short_during_ban"),
            "new_longs_risk_off": cs.get("new_longs_during_risk_off"),
        })
    write("scorecards", rows, ["hub/evaluation.py", "hub/sleeves.py"])

    write("yearly", [{"series": c["id"], "label": c["label"], "year": y, "ret": r}
                     for c in report["scorecards"] for y, r in c["regime"]["yearly"].items()],
          ["hub/evaluation.py"])

    write("equity", [{"date": d, "series": k, "equity": v}
                     for k, pts in curves["equity"].items() for d, v in pts],
          ["hub/backtest.py", "hub/costs.py"])

    en = report["ensemble_now"]
    alloc = []
    for s in en["sleeves"]:
        ch = s["weight"] - s["prev_weight"]
        action = ("歸零" if s["weight"] == 0 and s["prev_weight"] > 0 else "加碼" if ch > 0.005
                  else "減碼" if ch < -0.005 else "不變")
        alloc.append({"id": s["id"], "label": s["label"], "weight": s["weight"], "prev_weight": s["prev_weight"],
                      "change": round(ch, 4), "trailing_sharpe": s["trailing_sharpe"], "action": action})
    alloc.append({"id": "cash", "label": "現金", "weight": en["cash_weight"], "prev_weight": None,
                  "change": None, "trailing_sharpe": None, "action": None})
    write("allocation", alloc, ["hub/ensemble.py"])

    ah = curves["ensemble_allocation"]
    write("allocation_history", [{"date": d, "sleeve": k, "weight": ah[k][i]}
                                 for k in ah if k != "dates" for i, d in enumerate(ah["dates"])],
          ["hub/ensemble.py"])

    write("holdings", [{"sleeve": s["id"], "sleeve_label": s["label"], "code": h["code"], "name": h["name"],
                        "weight": h["weight"]} for s in en["sleeves"] for h in s["holdings"]],
          ["hub/sleeves.py"])

    rn = report["regime_now"]
    write("regime", [{
        "date": rn["date"], "classification": rn["classification"], "index_source": rn["index_source"],
        "index_level": rn["index_level"], "deviation": rn["index_deviation_ma20"],
        "oversold": rn["oversold"], "overbought": rn["overbought"], "oversold_ratio": rn["oversold_ratio"],
        "share_on": rn["history_share"]["RISK_ON"], "share_neutral": rn["history_share"]["NEUTRAL"],
        "share_off": rn["history_share"]["RISK_OFF"],
        "ex2026_filter_cagr": rn["robustness_ex2026"]["filter"]["cagr"],
        "ex2026_filter_sharpe": rn["robustness_ex2026"]["filter"]["sharpe"],
        "ex2026_bench_cagr": rn["robustness_ex2026"]["bench_0050"]["cagr"],
        "ex2026_bench_sharpe": rn["robustness_ex2026"]["bench_0050"]["sharpe"],
        "ensemble_last_rebalance": en["last_rebalance"],
    }], ["hub/regimes.py"])
    write("robustness", [{**g, "key": f"{g['dev']}|{g['ratio']}"} for g in rn["robustness_grid"]],
          ["hub/regimes.py"])

    write("health", [{"component": c["component"], "status": c["status"], "detail": c["detail"]}
                     for c in report["health"]["checks"]], ["hub/health.py"])

    llm = report["llm_layer"]
    summary = {k: llm.get(k) for k in ("n_trades", "mean_return", "median_return", "win_rate",
                                       "mean_excess_vs_0050", "beat_0050_rate", "stop_loss_rate",
                                       "confidence_return_corr", "first_signal", "last_signal")}
    write("llm_summary", [summary], ["hub/llm_tracking.py"])
    write("llm_confidence", [{"bucket": k, "count": v["count"], "mean_return": v["mean_return"]}
                             for k, v in (llm.get("by_confidence") or {}).items()], ["hub/llm_tracking.py"])
    write("llm_trades", llm.get("trades") or [], ["hub/llm_tracking.py"])

    print("exported:", sorted(p.name for p in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
