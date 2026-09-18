"""Market intelligence: regime, narrative clusters, and the semantic layer's output."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from trading_desk.ui import page_setup, empty_state, run_from_settings, sidebar_controls

page_setup("情報與敘事")
settings = sidebar_controls()

st.title("情報與敘事")
st.caption(
    "交易員需要的不是 500 條新聞，而是知道今天市場在講什麼故事，"
    "以及自己的部位站在故事的哪一邊。"
)

state = run_from_settings(settings)

# --- Regime ---------------------------------------------------------------

st.subheader("市場狀態判定")
regime = state.regime
columns = st.columns(5)
columns[0].metric("綜合判定", regime.label)
columns[1].metric("趨勢", regime.trend)
columns[2].metric("波動", regime.volatility)
columns[3].metric("廣度", f"{regime.breadth:.0%}" if regime.breadth == regime.breadth else "—")
columns[4].metric("風險偏好", regime.risk_appetite)
st.info(regime.detail)
st.caption(
    "多數策略失效不是因為模型爛，是因為 regime 換了而模型不知道。"
    "所有訊號的信心分數都會依此判定自動調整。"
)

st.divider()

# --- Narratives -----------------------------------------------------------

st.subheader("市場主軸")
if not state.narratives:
    empty_state("尚未形成足夠規模的主題群聚。", "主題需要同產業至少 3 檔標的同時被觸發才會成立。")
else:
    frame = pd.DataFrame(
        [
            {
                "主題": narrative["industry"],
                "檔數": narrative["size"],
                "淨方向": narrative["net_direction"],
                "傾向": narrative["tone"],
                "主要驅動": narrative["drivers"],
            }
            for narrative in state.narratives
        ]
    )
    figure = px.bar(
        frame, x="淨方向", y="主題", orientation="h", color="淨方向",
        color_continuous_scale=["#EF4444", "#64748B", "#22C55E"], range_color=[-1, 1],
        hover_data=["檔數", "主要驅動"],
    )
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(260, 46 * len(frame)), margin=dict(l=10, r=10, t=30, b=10),
        title="各主題的淨方向（-1 全空、+1 全多；游標移上可看被觸發檔數）",
    )
    st.plotly_chart(figure, use_container_width=True)

    for narrative in state.narratives:
        with st.container(border=True):
            st.markdown(f"**{narrative['theme']}**")
            st.write(narrative["summary"])
            leaders = "、".join(
                f"{item['symbol']} {item['name']}" for item in narrative["leaders"]
            )
            st.caption(f"代表標的：{leaders}")

st.divider()

# --- Semantic layer -------------------------------------------------------

st.subheader("語意層")
if not state.market.get("news_enabled", False):
    st.info(
        "語意層未啟用。目前的證據僅來自價格、統計與資金流。"
        "在側邊欄開啟新聞情緒分析後，系統會針對統計層篩出的候選標的抓取新聞。"
    )
    st.caption(
        "未啟用時系統把它記為「沒有這項能力」而不是「來源過期」——"
        "沒訂閱的 feed 不該讓所有訊號被打折。"
    )
else:
    columns = st.columns(3)
    columns[0].metric("抓取則數", state.market.get("news_items", 0))
    columns[1].metric("去重後", state.market.get("news_unique", 0))
    columns[2].metric(
        "標註被拒", len(state.market.get("news_rejected", [])),
        help="引用片段不是原文逐字子字串的標註會被丟棄，不會進入決策。",
    )

    annotations = state.market.get("news_annotations", [])
    if annotations:
        frame = pd.DataFrame(
            [
                {
                    "標的": item.symbol,
                    "事件類型": item.event_type,
                    "極性": item.polarity,
                    "重要性": item.materiality,
                    "新穎度": item.novelty,
                    "期間": item.horizon,
                    "引用原文": item.evidence_span,
                    "標註者": item.annotator,
                }
                for item in annotations
            ]
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption(
            "「引用原文」欄位是防幻覺機制：模型必須指出它據以判斷的逐字片段，"
            "程式端驗證該片段確實出現在原文中，否則整筆標註丟棄。"
        )
    rejected = state.market.get("news_rejected", [])
    if rejected:
        with st.expander(f"被拒絕的標註（{len(rejected)}）"):
            for reason in rejected:
                st.write(f"- {reason}")
