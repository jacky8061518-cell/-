"""Trading desk entry point.

The home page answers one question: what does the desk want me to look at right
now. Everything else is a click away in the sidebar.
"""

from __future__ import annotations

import streamlit as st

from trading_desk.contracts import Severity
from trading_desk.ui import (
    page_setup,
    empty_state,
    get_store,
    render_header,
    render_signal_card,
    run_from_settings,
    sidebar_controls,
)

page_setup("總覽")

settings = sidebar_controls()

st.title("🛰️ 自主市場情報交易台")
st.caption(
    "多個 Agent 持續掃描市場，統計層負責篩選、語意層負責解讀、風控層負責定量。"
    "系統產出的是帶證據、帶信心、帶風險預算的建議；下不下單由交易員決定。"
)

with st.spinner("執行 Agent 流水線…"):
    state = run_from_settings(settings)

render_header(state)

if state.breaker is not None and state.breaker.tripped:
    st.error(
        f"**熔斷觸發（{state.breaker.level}）**：{state.breaker.reason}　"
        f"｜動作：{state.breaker.action}　｜恢復：{state.breaker.recovery}"
    )

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("需要決定的事")
    actionable = [
        signal
        for signal in state.signals
        if signal.severity <= Severity.P1_ACTION
    ]
    if not actionable:
        empty_state(
            "今日沒有達到行動門檻的訊號。",
            "訊號少不代表系統故障。門檻存在的目的就是讓多數日子安靜。",
        )
    store = get_store()
    for signal in actionable:
        render_signal_card(signal, store=store, expanded=len(actionable) <= 2, key_prefix="home_")

    if state.suppressed_signals:
        st.caption(
            f"另有 {state.suppressed_signals} 條候選訊號因每日訊號預算被壓下，"
            "可在「訊號流」頁面查看完整清單。"
        )

with right:
    st.subheader("今日市場主軸")
    if not state.narratives:
        empty_state("尚未形成足夠規模的主題群聚。")
    for narrative in state.narratives[:4]:
        with st.container(border=True):
            st.markdown(f"**{narrative['theme']}**")
            st.caption(narrative["summary"])
            leaders = "、".join(
                f"{item['symbol']} {item['name']}" for item in narrative["leaders"][:4]
            )
            st.markdown(f'<span class="desk-note">代表標的：{leaders}</span>', unsafe_allow_html=True)

    st.subheader("系統狀態")
    failed = [run.agent for run in state.agent_runs if not run.healthy]
    if failed:
        st.error(f"失效 Agent：{'、'.join(failed)}")
    elif state.data_freshness and any(
        meta.get("stale") for meta in state.data_freshness.values()
    ):
        st.warning("部分資料來源過期，相關訊號信心已自動打折。")
    else:
        st.success("所有 Agent 與資料來源正常。")
    if not state.market.get("news_enabled", False):
        st.info("語意層未啟用：目前僅使用價格、資金流與統計證據。")

st.divider()
st.caption(
    f"執行識別碼 `{state.run_id}`　｜　決策時點 {state.as_of:%Y-%m-%d}　"
    f"｜　本頁所有數字都可在「系統健康與回放」頁面重現。"
)
