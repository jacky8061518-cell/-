"""Post-mortem: did the signals work, and does conviction mean anything.

This page exists to make the system's own failures visible. A desk that only
displays its ideas and never scores them will keep making the same mistake for
months without anyone noticing.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from trading_desk.contracts import Severity
from trading_desk.store import evaluate_signals, performance_by_conviction
from trading_desk.ui import (
    cached_market,
    page_setup,
    empty_state,
    get_store,
    sidebar_controls,
)

page_setup("訊號成效")
settings = sidebar_controls()

st.title("訊號成效與交易員回饋")
st.caption("這一頁是系統的學習迴路。它會誠實顯示自己錯得多離譜。")

store = get_store()
data = cached_market(settings["min_market_cap"] * 1e9, settings["industries"])

runs = store.runs()
signals = store.signals()

if signals.empty:
    empty_state(
        "尚無已記錄的訊號。",
        "請先在「每日簡報」頁面按下「將本次執行寫入紀錄」，或等排程執行累積資料。",
    )
    st.stop()

horizon = st.slider("評估期間（交易日）", 1, 20, 5)
evaluation = evaluate_signals(store, data.prices, horizon_days=horizon)

columns = st.columns(4)
columns[0].metric("已記錄執行", len(runs))
columns[1].metric("已記錄訊號", len(signals))
columns[2].metric("可評分訊號", len(evaluation))

if evaluation.empty:
    columns[3].metric("命中率", "—")
    empty_state(
        "已記錄的訊號還沒有經過完整的持有期間，無法評分。",
        "這是正確行為：不能用還沒發生的資料評估訊號。",
    )
    st.stop()

hit_rate = float(evaluation["hit"].mean())
columns[3].metric("命中率", f"{hit_rate:.0%}", help="目標 > 55%")

sample_warning = len(evaluation) < 100
if sample_warning:
    st.warning(
        f"樣本僅 {len(evaluation)} 條。少於 100 次交易在統計上什麼都證明不了，"
        "以下數字只能當作流程是否運作的檢查，不能當作策略有效的證據。"
    )

st.divider()
st.subheader("信心分數是否真的有意義")
banded = performance_by_conviction(evaluation)
if banded.empty:
    empty_state("樣本不足以分層。")
else:
    banded = banded.assign(
        hit_rate=lambda frame: frame["hit_rate"] * 100,
        mean_return=lambda frame: frame["mean_return"] * 100,
        worst=lambda frame: frame["worst"] * 100,
    )
    st.dataframe(
        banded, hide_index=True, use_container_width=True,
        column_config={
            "hit_rate": st.column_config.NumberColumn("命中率", format="%.0f%%"),
            "mean_return": st.column_config.NumberColumn("平均報酬", format="%.2f%%"),
            "worst": st.column_config.NumberColumn("最差", format="%.2f%%"),
            "signals": st.column_config.NumberColumn("訊號數"),
            "band": st.column_config.TextColumn("信心區間"),
        },
    )
    st.caption(
        "如果這張表是平的——高信心和低信心的命中率沒有差別——那信心分數只是裝飾，"
        "決策層必須重做，而不是繼續拿它去定量部位。"
    )

st.subheader("報酬分佈")
figure = px.histogram(
    evaluation, x="forward_return", nbins=30, color="direction",
    color_discrete_map={"long": "#22C55E", "short": "#EF4444"},
)
figure.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
figure.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=320, margin=dict(l=10, r=10, t=30, b=10),
    title=f"{horizon} 日前瞻報酬（已依方向調整）",
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("最差的訊號")
worst = evaluation.nsmallest(5, "forward_return").assign(
    conviction=lambda frame: frame["conviction"] * 100,
    forward_return=lambda frame: frame["forward_return"] * 100,
)
st.dataframe(
    worst[["as_of", "symbol", "name", "direction", "conviction", "forward_return"]],
    hide_index=True, use_container_width=True,
    column_config={
        "conviction": st.column_config.ProgressColumn("信心", min_value=0.0, max_value=100.0, format="%.0f%%"),
        "forward_return": st.column_config.NumberColumn("前瞻報酬", format="%.2f%%"),
    },
)
st.caption(
    "高信心卻大幅虧損的訊號要歸因：是資料錯誤、特徵漂移，還是 regime 改變？"
    "三者的處置方式完全不同，而只有第三種是可以接受的。"
)

st.divider()
st.subheader("交易員回饋")
feedback = store.feedback()
if feedback.empty:
    empty_state("尚無回饋紀錄。", "在訊號卡上按「採用／忽略／否決」即可累積。")
else:
    columns = st.columns(3)
    actions = feedback["action"].value_counts()
    columns[0].metric("採用", int(actions.get("adopted", 0)))
    columns[1].metric("忽略", int(actions.get("ignored", 0)))
    columns[2].metric("否決", int(actions.get("vetoed", 0)))

    p1 = feedback[feedback["severity"] == int(Severity.P1_ACTION)]
    if not p1.empty:
        rate = float((p1["action"] == "adopted").mean())
        st.metric("P1 訊號採用率", f"{rate:.0%}", help="低於 50% 代表 P1 定義太鬆，必須調緊")

    vetoes = feedback[feedback["action"] == "vetoed"]
    if not vetoes.empty:
        st.markdown("**否決理由分佈**")
        st.bar_chart(vetoes["reason_code"].value_counts())
        st.caption(
            "否決理由是把人的判斷變成系統資產的唯一機制。"
            "「資料有誤」佔比升高代表要修資料管線，而不是調模型。"
        )
