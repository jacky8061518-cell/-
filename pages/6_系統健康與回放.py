"""System health and decision replay.

Two jobs: show whether the machinery is working, and let a trader reconstruct
any decision without asking an engineer. The replay is the reason the whole
pipeline is deterministic.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trading_desk.ui import (
    page_setup,
    empty_state,
    get_store,
    render_freshness_table,
    run_from_settings,
    sidebar_controls,
)

page_setup("系統健康與回放")
settings = sidebar_controls()

st.title("系統健康與回放")
st.caption("沒有 metrics 的元件不准上線；沒有回放能力的決策不該被信任。")

state = run_from_settings(settings)

# --- Agent health ---------------------------------------------------------

st.subheader("Agent 執行狀態")
runs = pd.DataFrame(
    [
        {
            "Agent": run.agent,
            "版本": run.version,
            "狀態": {"ok": "正常", "slow": "逾時", "failed": "失效"}[run.status],
            "耗時(ms)": round(run.duration_ms, 1),
            "產出": run.findings,
            "降級原因": run.degraded_reason or "—",
            "錯誤": run.error or "—",
        }
        for run in state.agent_runs
    ]
)
st.dataframe(runs, hide_index=True, use_container_width=True)

failed = [run for run in state.agent_runs if not run.healthy]
if failed:
    st.error(
        f"{len(failed)} 個 Agent 未正常完成。流水線仍產出訊號，但相關訊號的信心已被打折——"
        "系統是降級運作，不是全有或全無。"
    )
else:
    st.success("所有 Agent 正常完成。")

st.caption(
    "資料層 fail-open（標記品質後繼續），交易層 fail-closed（停）。"
    "單一 Agent 失效不會讓整條流水線停擺，但風控無回應時一定停止下單。"
)

st.divider()

# --- Data freshness -------------------------------------------------------

st.subheader("資料新鮮度")
freshness = render_freshness_table(state)
if freshness.empty:
    empty_state("本次執行沒有註冊任何資料來源。")
else:
    st.dataframe(freshness, hide_index=True, use_container_width=True)
st.caption(
    "「未啟用」與「過期」是不同狀態。前者是這套部署沒有這項能力，"
    "後者是應該有資料卻沒進來——只有後者該打折訊號信心。"
)

if state.board is not None and state.board.degradations:
    st.subheader("降級事件")
    for reason in state.board.degradations:
        st.warning(reason)

st.divider()

# --- Replay ---------------------------------------------------------------

st.subheader("決策回放")
st.markdown(
    "輸入任一 `decision_id`，重建當時的完整決策鏈。"
    "同樣的輸入必須產生同樣的輸出——不一致本身就是嚴重的 bug 警訊。"
)

store = get_store()
current = {signal.decision_id: signal for signal in state.signals}
options = ["（選擇本次執行的決策）"] + [
    f"{signal.decision_id}｜{signal.symbol} {signal.name}" for signal in state.signals
]
choice = st.selectbox("本次執行的決策", options)
manual = st.text_input("或直接輸入 decision_id", placeholder="dec_...")

decision_id = manual.strip() or (choice.split("｜")[0] if choice != options[0] else "")

if decision_id:
    signal = current.get(decision_id)
    if signal is not None:
        st.success("在本次執行中找到該決策，以下為即時重建的決策鏈。")
        chain = {
            "決策識別碼": signal.decision_id,
            "訊號識別碼": signal.signal_id,
            "決策時點": str(signal.created_at),
            "標的": f"{signal.symbol} {signal.name}",
            "方向": signal.direction,
            "信心": signal.conviction,
            "信心組成": (
                state.board.view(signal.symbol).notes.get("conviction_breakdown")
                if state.board
                else None
            ),
            "證據": [
                {
                    "類型": item.kind,
                    "方向": item.direction,
                    "權重": item.weight,
                    "數值": item.value,
                    "來源 Agent": item.agent,
                    "說明": item.detail,
                }
                for item in signal.evidence
            ],
            "反面證據": [
                {"嚴重度": item.severity, "說明": item.detail, "來源": item.agent}
                for item in signal.counter_evidence
            ],
            "風險裁決": {
                "建議權重%": signal.risk.suggested_weight_pct,
                "綁定限制": signal.risk.binding_constraint,
                "停損": signal.risk.stop_level,
                "最大虧損%NAV": signal.risk.max_loss_pct_nav,
            },
            "模型來源": signal.model_provenance,
        }
        st.json(chain)
    else:
        recorded = store.find_decision(decision_id)
        if recorded:
            st.info(f"該決策不在本次執行中，以下為執行日誌中的 {len(recorded)} 筆紀錄。")
            for row in recorded:
                st.json(row)
        else:
            st.warning(
                "找不到該決策。可能是尚未寫入日誌（請至「每日簡報」頁面寫入），"
                "或 decision_id 有誤。"
            )

st.divider()

# --- Run log --------------------------------------------------------------

st.subheader("執行日誌")
history = store.runs()
if history.empty:
    empty_state("尚無執行紀錄。", "append-only 日誌會記錄每次執行的環境判定、訊號數與降級事件。")
else:
    st.dataframe(history.tail(50), hide_index=True, use_container_width=True)

with st.expander("本次執行的原始統計"):
    st.json(
        {
            "run_id": state.run_id,
            "as_of": str(state.as_of),
            "universe_size": state.universe_size,
            "elapsed_seconds": state.elapsed_seconds,
            "suppressed_signals": state.suppressed_signals,
            "market": {
                key: str(value)[:200]
                for key, value in state.market.items()
                if key not in {"news_annotations", "flow_groups"}
            },
        }
    )
