"""Daily brief generation.

A brief that runs past one page does not get read, so everything here is
ruthless about length: three overnight items, the regime, the shortlist, and
the health line. Detail lives in the console; the brief is the thing a trader
reads in ninety seconds before the open.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .contracts import DeskState, Position, Severity


def pre_market_brief(
    state: DeskState,
    positions: list[Position] | None = None,
    last_prices: pd.Series | None = None,
) -> str:
    """Markdown brief for the session opening."""
    positions = positions or []
    lines: list[str] = []
    stamp = pd.Timestamp(state.as_of).strftime("%Y-%m-%d")
    lines.append(f"## 盤前簡報 — {stamp}")
    lines.append("")

    lines.append(f"### 一、市場狀態：{state.regime.label if state.regime else '未判定'}")
    if state.regime:
        lines.append(state.regime.detail)
        lines.append("")
        lines.append(_regime_implication(state.regime.label))
    lines.append("")

    lines.append("### 二、今日訊號")
    actionable = [s for s in state.signals if s.severity <= Severity.P1_ACTION]
    watchlist = [s for s in state.signals if s.severity == Severity.P2_ATTENTION]
    if not state.signals:
        lines.append("- 無訊號通過門檻。訊號少不是系統故障，是市場今天沒有值得下注的機會。")
    else:
        for signal in actionable:
            lines.append(
                f"- **{signal.direction_label} {signal.symbol} {signal.name}**"
                f"（信心 {signal.conviction:.0%}，建議 {signal.risk.suggested_weight_pct:.2f}%）"
                f"｜失效條件：{signal.invalidation.split('；或')[0]}"
            )
        if watchlist:
            names = "、".join(f"{s.symbol} {s.name}" for s in watchlist[:5])
            lines.append(f"- 觀察名單（未達行動門檻）：{names}")
    if state.suppressed_signals:
        lines.append(
            f"- 另有 {state.suppressed_signals} 條候選訊號因每日訊號預算被壓下，"
            "可於訊號流頁面查看。"
        )
    lines.append("")

    if state.narratives:
        lines.append("### 三、市場主軸")
        for narrative in state.narratives[:3]:
            lines.append(f"- {narrative['summary']}")
        lines.append("")

    lines.append("### 四、現有部位")
    if not positions:
        lines.append("- 目前無持倉紀錄。")
    else:
        for position in positions:
            price = float(last_prices.get(position.symbol, float("nan"))) if last_prices is not None else float("nan")
            pnl = position.unrealised_pct(price) if price == price else float("nan")
            distance = position.stop_distance_pct(price) if price == price else None
            detail = f"- {position.symbol} {position.name}｜權重 {position.weight_pct:.2f}%"
            if pnl == pnl:
                detail += f"｜未實現 {pnl:+.2%}"
            if distance is not None:
                detail += f"｜距停損 {distance:.1%}"
            lines.append(detail)
    lines.append("")

    lines.append("### 五、系統健康")
    lines.append(_health_line(state))
    return "\n".join(lines)


def post_market_brief(
    state: DeskState,
    evaluation: pd.DataFrame | None = None,
) -> str:
    """Markdown brief for after the close, centred on what went wrong."""
    lines: list[str] = []
    stamp = pd.Timestamp(state.as_of).strftime("%Y-%m-%d")
    lines.append(f"## 盤後覆盤 — {stamp}")
    lines.append("")

    lines.append("### 一、今日訊號成效")
    if evaluation is None or evaluation.empty:
        lines.append("- 尚無足夠的前瞻資料可評估（訊號需經過持有期間才能計分）。")
    else:
        hit_rate = float(evaluation["hit"].mean())
        mean_return = float(evaluation["forward_return"].mean())
        lines.append(
            f"- 已評分訊號 {len(evaluation)} 條，命中率 {hit_rate:.0%}，"
            f"平均報酬 {mean_return:+.2%}"
        )
        worst = evaluation.nsmallest(3, "forward_return")
        for _, row in worst.iterrows():
            lines.append(
                f"- 最差：{row['symbol']} {row.get('name', '')} "
                f"{row['forward_return']:+.2%}（信心當時 {row['conviction']:.0%}）"
            )
        lines.append("")
        lines.append(
            "- 檢查重點：高信心卻大幅虧損的訊號，要判斷是資料錯誤、特徵漂移，"
            "還是 regime 已經改變。三者的處置方式完全不同。"
        )
    lines.append("")

    lines.append("### 二、系統事件")
    degradations = list(getattr(state.board, "degradations", []))
    if not degradations:
        lines.append("- 無降級事件。")
    for reason in degradations:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("### 三、明日待辦")
    lines.append("- 確認過期資料來源是否恢復")
    lines.append("- 檢視被否決訊號的理由分佈，判斷門檻是否需要調整")
    if state.breaker is not None and state.breaker.tripped:
        lines.append(f"- **熔斷處理**：{state.breaker.action}（恢復方式：{state.breaker.recovery}）")
    return "\n".join(lines)


def _regime_implication(label: str) -> str:
    return {
        "順風": "多方訊號可給予正常至略高的權重；空方訊號需要更強的證據。",
        "中性": "維持基準權重，優先選擇證據交叉驗證充分的標的。",
        "震盪": "波動放大，所有部位規模自動縮減；避免追價。",
        "逆風": "多方訊號權重自動打折，優先保護既有部位；空方訊號權重略增。",
    }.get(label, "環境判定不足，維持保守。")


def _health_line(state: DeskState) -> str:
    failed = [run.agent for run in state.agent_runs if not run.healthy]
    stale = [
        source for source, meta in state.data_freshness.items() if meta.get("stale")
    ]
    parts = [f"執行耗時 {state.elapsed_seconds:.1f} 秒", f"掃描 {state.universe_size} 檔"]
    if failed:
        parts.append(f"**失效 Agent：{'、'.join(failed)}**")
    if stale:
        parts.append(f"**過期來源：{'、'.join(stale)}**")
    if not failed and not stale:
        parts.append("所有 Agent 與資料來源正常")
    return "- " + "｜".join(parts)
