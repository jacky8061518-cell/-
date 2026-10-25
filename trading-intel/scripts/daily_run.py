"""每日掃描主流程：抓真實股價、跑品質閘門與量化層、視情況呼叫真正的 LLM agent。

這是 SPEC 說的「AI 24 小時自己掃描市場」實際落地的第一版可執行流程，
串起資料層 → 品質層 → 特徵/量化層 → agent 分析層 → 風控層 → 報告輸出。

**目前仍缺兩塊，故意不假裝已經做到**：
1. 沒有真正的新聞來源，所以 NewsAnalystAgent 這一步略過——它需要一份文件才能分析。
2. TWSE 官方資料在雲端主機會被擋，此腳本只用 Yahoo Finance 抓價格（ADR 0007）。

``ANTHROPIC_API_KEY`` 環境變數存在時，HypothesisAgent／RedTeamAgent／
LibrarianAgent 會真的呼叫 Claude；不存在時這三步會誠實跳過並記錄原因，
而不是用假資料冒充。這對應 agent 契約本來就有的「證據不足時 abstain」精神——
沒有金鑰就是一種「無法執行」的狀態，應該誠實回報，不是躲過去。

用法：``uv run python scripts/daily_run.py``
輸出：``runs/<日期>.json``（機器可讀）與 ``runs/<日期>.md``（人看的報告）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from trading_intel.core.clock import utc_now
from trading_intel.core.enums import Market
from trading_intel.core.ids import EntityId, make_entity_id
from trading_intel.core.settings import AgentBudget, load_settings
from trading_intel.features.statistics import check_stationarity, ewma_volatility, ou_half_life
from trading_intel.ingestion.yahoo import fetch_chart, parse_chart
from trading_intel.models.regime import classify_trend, classify_volatility
from trading_intel.normalize.corporate_actions import RawPrice
from trading_intel.normalize.quality import check_raw_sanity, check_sanity
from trading_intel.portfolio.construction import build_portfolio
from trading_intel.risk.pretrade import PortfolioState, check_pretrade

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
HYPOTHESES_STORE = PROJECT_ROOT / "data" / "hypotheses.json"

#: 觀察名單。Yahoo 代號慣例：上市 .TW，上櫃 .TWO。(名稱, 產業)
WATCHLIST: dict[str, tuple[str, str]] = {
    "2330.TW": ("台積電", "半導體"),
    "2454.TW": ("聯發科", "半導體"),
    "2317.TW": ("鴻海", "電子代工"),
    "2308.TW": ("台達電", "電子零組件"),
    "3008.TW": ("大立光", "光學元件"),
}

TOP_N_CANDIDATES = 2
LOOKBACK_DAYS = 180
SKIP_DAYS = 21


@dataclass
class TickerReport:
    ticker: str
    name: str
    sector: str
    entity_id: str
    quality_alerts: list[str] = field(default_factory=list)
    tradable: bool = True
    last_close: float | None = None
    ewma_volatility: float | None = None
    stationarity_verdict: str | None = None
    ou_half_life_days: float | None = None
    trend: str | None = None
    volatility_state: str | None = None
    momentum_score: float | None = None


def fetch_ticker(ticker: str) -> list[tuple[date, Decimal, int]]:
    """抓即時資料，回傳 (交易日, 收盤價, 成交量)，網路失敗時讓例外往上拋——
    這是即時流程，不是回測，不該吞掉錯誤假裝抓到資料。"""
    payload, _ = fetch_chart(ticker, range_="1y", interval="1d")
    bars = parse_chart(payload, ticker)
    return [(bar.trade_date, bar.close, bar.volume) for bar in bars]


def analyze_ticker(ticker: str, name: str, sector: str, asof: datetime) -> TickerReport:
    entity_id: EntityId = make_entity_id(Market.TW, ticker.split(".")[0])
    report = TickerReport(ticker=ticker, name=name, sector=sector, entity_id=str(entity_id))
    raw_rows = fetch_ticker(ticker)
    if len(raw_rows) < 60:
        report.quality_alerts.append("資料筆數不足 60 筆，狀態設為 NO_TRADE")
        report.tradable = False
        return report

    settings = load_settings(env="dev")
    sanity_alerts = check_raw_sanity(raw_rows, entity_id=entity_id, asof=asof)
    prices = [
        RawPrice(entity_id=entity_id, trade_date=d, close=c, volume=v, ingest_time=asof)
        for d, c, v in raw_rows
        if c > 0 and v >= 0
    ]
    sequence_alerts = check_sanity(
        prices, entity_id=entity_id, market=Market.TW, asof=asof, limits=settings.quality
    )
    all_alerts = [*sanity_alerts, *sequence_alerts]
    report.quality_alerts = [a.detail for a in all_alerts]
    if any(a.severity.value == "P0" for a in all_alerts):
        report.tradable = False

    closes = np.array([float(c) for _, c, _ in raw_rows])
    returns = np.diff(closes) / closes[:-1]
    report.last_close = float(closes[-1])
    report.ewma_volatility = ewma_volatility(returns, span=20)
    stationarity = check_stationarity(returns)
    report.stationarity_verdict = stationarity.verdict.value
    half_life = ou_half_life(returns, alpha=0.05)
    report.ou_half_life_days = None if np.isnan(half_life) else float(half_life)
    report.trend = classify_trend(closes).value
    report.volatility_state = classify_volatility(returns).value

    if len(closes) > SKIP_DAYS + LOOKBACK_DAYS:
        end = -1 - SKIP_DAYS
        start = end - LOOKBACK_DAYS
        report.momentum_score = float(closes[end] / closes[start] - 1)
    elif len(closes) > SKIP_DAYS + 20:
        report.momentum_score = float(closes[-1 - SKIP_DAYS] / closes[0] - 1)

    if not report.tradable:
        report.momentum_score = None

    return report


def run_agents(
    reports: list[TickerReport], candidates: list[TickerReport], asof: datetime
) -> dict[str, object]:
    """有 ANTHROPIC_API_KEY 才真的呼叫 Claude；沒有就誠實記錄跳過原因。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "status": "skipped",
            "reason": "未設定 ANTHROPIC_API_KEY，本次僅執行量化層，agent 分析層未執行",
        }

    from trading_intel.agents.analysts import HypothesisAgent, LibrarianAgent, RedTeamAgent
    from trading_intel.agents.base import AnthropicClient

    budget = AgentBudget(
        daily_token_budget=200_000, max_calls_per_hour=50, timeout_seconds=60.0, max_retries=2
    )
    client = AnthropicClient()

    anomalies = [
        f"{r.name}（{r.ticker}）波動率狀態為 {r.volatility_state}，趨勢判讀為 {r.trend}"
        for r in reports
        if r.tradable and r.volatility_state in {"HIGH", "EXTREME"}
    ]
    redteam_findings: list[dict[str, object]] = []
    result: dict[str, object] = {
        "status": "ran",
        "hypothesis": None,
        "redteam": redteam_findings,
        "librarian": None,
    }

    if anomalies:
        hyp_agent = HypothesisAgent(client=client, budget=budget)
        hyp_output = hyp_agent.propose(
            anomalies,
            available_features=["ewma_volatility", "ou_half_life_days", "momentum_score", "trend"],
        )
        if not hyp_output.abstain and hyp_output.payload is not None:
            result["hypothesis"] = {
                "statement": hyp_output.payload.statement,
                "falsification_condition": hyp_output.payload.falsification_condition,
                "confidence": hyp_output.payload.confidence,
            }
            existing = _load_hypotheses()
            lib_agent = LibrarianAgent(client=client, budget=budget)
            lib_output = lib_agent.check_duplicate(hyp_output.payload.statement, existing=existing)
            if not lib_output.abstain and lib_output.payload is not None:
                result["librarian"] = {
                    "verdict": lib_output.payload.verdict.value,
                    "reasoning_digest": lib_output.payload.reasoning_digest,
                }
                if lib_output.payload.verdict.value == "NOVEL":
                    _append_hypothesis(hyp_output.payload.statement, asof)
        else:
            result["hypothesis"] = {"abstain": True, "reason": hyp_output.reasoning_digest}

    redteam_agent = RedTeamAgent(client=client, budget=budget)
    for candidate in candidates:
        rt_output = redteam_agent.attack(
            f"{candidate.name}（{candidate.ticker}）12-1 動量候選訊號",
            backtest_summary={
                "動量分數": f"{candidate.momentum_score:.3f}"
                if candidate.momentum_score
                else "N/A",
                "年化波動率": f"{candidate.ewma_volatility:.1%}"
                if candidate.ewma_volatility
                else "N/A",
                "定態檢定": candidate.stationarity_verdict or "N/A",
            },
        )
        if not rt_output.abstain and rt_output.payload is not None:
            redteam_findings.append(
                {
                    "ticker": candidate.ticker,
                    "findings": [
                        {"vector": f.vector.value, "severity": f.severity.value, "detail": f.detail}
                        for f in rt_output.payload.findings
                    ],
                }
            )
        else:
            redteam_findings.append(
                {"ticker": candidate.ticker, "abstain": True, "reason": rt_output.reasoning_digest}
            )
    return result


def _load_hypotheses() -> dict[str, str]:
    if not HYPOTHESES_STORE.exists():
        return {}
    data = json.loads(HYPOTHESES_STORE.read_text(encoding="utf-8"))
    return {str(k): str(v["statement"]) for k, v in data.items()}


def _append_hypothesis(statement: str, asof: datetime) -> None:
    HYPOTHESES_STORE.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(HYPOTHESES_STORE.read_text(encoding="utf-8"))
        if HYPOTHESES_STORE.exists()
        else {}
    )
    new_id = f"hyp-{asof.strftime('%Y%m%d')}-{len(data) + 1}"
    data[new_id] = {"statement": statement, "recorded_at": asof.isoformat()}
    HYPOTHESES_STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pretrade_check(candidates: list[TickerReport]) -> dict[str, object]:
    """把候選訊號交給真正的組合建構函式配權，再過風控事前檢查。

    早期版本直接把候選標的等權重各分一半，這樣不管候選是誰，單一標的權重
    永遠是 50%，永遠超過 5% 的單一標的上限——那不是在檢查風險，只是在
    重複講一遍寫死的配權方式。改成先呼叫 build_portfolio 用真正的限額
    （單一標的上限、產業上限、波動率目標化）把候選分數轉成合法權重，
    風控檢查才有意義：如果還是被否決，代表限額本身跟現有候選數量衝突，
    是值得注意的真訊號，不是配權邏輯先天就會觸發的假警報。
    """
    settings = load_settings(env="dev")
    if not candidates:
        return {"approved": True, "breaches": [], "weights": {}, "note": "無候選標的，無需檢查"}

    scores = {
        make_entity_id(Market.TW, c.ticker.split(".")[0]): (c.momentum_score or 0.0)
        for c in candidates
    }
    sectors = {make_entity_id(Market.TW, c.ticker.split(".")[0]): c.sector for c in candidates}
    forecast_vol = float(
        np.mean([c.ewma_volatility for c in candidates if c.ewma_volatility is not None]) or 0.2
    )
    construction = build_portfolio(
        scores,
        sectors=sectors,
        forecast_annual_volatility=forecast_vol,
        limits=settings.risk,
        settings=settings.portfolio,
    )

    state = PortfolioState(
        weights={},
        sectors=sectors,
        adv_values={eid: Decimal("100000000") for eid in scores},
        equity=Decimal("10000000"),
    )
    verdict, breaches = check_pretrade(construction.weights, state, settings.risk)
    return {
        "approved": verdict.approved,
        "breached_limits": list(verdict.breached_limits),
        "breaches": [{"code": b.code.value, "detail": b.detail} for b in breaches],
        "weights": {str(k): v for k, v in construction.weights.items()},
        "exposure_scalar": construction.exposure_scalar,
    }


def build_markdown(
    reports: list[TickerReport],
    candidates: list[TickerReport],
    agents: dict[str, object],
    pretrade: dict[str, object],
    asof: datetime,
) -> str:
    lines = [f"# 每日掃描報告　{asof.date().isoformat()}", ""]
    lines.append("## 觀察名單狀態")
    for r in reports:
        flag = "🔴 NO_TRADE" if not r.tradable else "🟢"
        lines.append(
            f"- {flag} **{r.name}**（{r.ticker}）收盤 {r.last_close}　"
            f"趨勢 {r.trend}　波動 {r.volatility_state}　"
            f"動量分數 {r.momentum_score if r.momentum_score is not None else 'N/A'}"
        )
        for alert in r.quality_alerts:
            lines.append(f"  - ⚠ {alert}")
    lines.append("")
    lines.append("## 候選訊號（12-1 動量排名前 " + str(TOP_N_CANDIDATES) + "）")
    for c in candidates:
        lines.append(
            f"- {c.name}（{c.ticker}）動量分數 {c.momentum_score:.3f}"
            if c.momentum_score is not None
            else f"- {c.name}"
        )
    lines.append("")
    lines.append("## Agent 分析層")
    lines.append(f"狀態：{agents.get('status')}")
    if agents.get("status") == "skipped":
        lines.append(f"原因：{agents.get('reason')}")
    else:
        if agents.get("hypothesis"):
            lines.append(f"- 假設：{agents['hypothesis']}")
        if agents.get("librarian"):
            lines.append(f"- 知識庫比對：{agents['librarian']}")
        redteam_list = agents.get("redteam")
        if isinstance(redteam_list, list):
            for rt in redteam_list:
                lines.append(f"- RedTeam（{rt.get('ticker')}）：{rt}")
    lines.append("")
    lines.append("## 組合建構與風控事前檢查")
    weights = pretrade.get("weights")
    if isinstance(weights, dict) and weights:
        lines.append("配權（依真實限額配置後）：")
        for entity_id, weight in weights.items():
            lines.append(f"- {entity_id}：{weight:.2%}")
    lines.append(f"通過：{pretrade['approved']}")
    if not pretrade["approved"]:
        lines.append(f"違反：{pretrade['breached_limits']}")
    return "\n".join(lines)


def main() -> None:
    asof = utc_now()
    reports = [
        analyze_ticker(ticker, name, sector, asof) for ticker, (name, sector) in WATCHLIST.items()
    ]

    tradable = [r for r in reports if r.tradable and r.momentum_score is not None]
    candidates = sorted(tradable, key=lambda r: r.momentum_score or 0, reverse=True)[
        :TOP_N_CANDIDATES
    ]

    agents_result = run_agents(reports, candidates, asof)
    pretrade_result = run_pretrade_check(candidates)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = asof.date().isoformat()
    output = {
        "generated_at": asof.isoformat(),
        "watchlist": [r.__dict__ for r in reports],
        "candidates": [c.ticker for c in candidates],
        "agents": agents_result,
        "pretrade": pretrade_result,
    }
    (RUNS_DIR / f"{stamp}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RUNS_DIR / f"{stamp}.md").write_text(
        build_markdown(reports, candidates, agents_result, pretrade_result, asof), encoding="utf-8"
    )
    print(f"完成，輸出於 runs/{stamp}.json 與 runs/{stamp}.md")


if __name__ == "__main__":
    main()
