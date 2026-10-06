"""Signal flow: the full list, filterable, with the complete evidence chain."""

from __future__ import annotations

import pandas as pd
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

page_setup("訊號流")
settings = sidebar_controls()

st.title("訊號流")
st.caption("每條訊號都必須回答三個問題：依據哪些證據、信心多少、錯了會賠多少。答不出來的不會出現在這裡。")

state = run_from_settings(settings)
render_header(state)

if not state.signals:
    empty_state("目前沒有訊號通過門檻。", "可在側邊欄調低最低信心或減少最少證據類別數。")
    st.stop()

st.divider()

frame = pd.DataFrame(
    [
        {
            "警示": signal.severity.label,
            "方向": signal.direction_label,
            "代號": signal.symbol,
            "名稱": signal.name,
            "信心": signal.conviction * 100,
            "建議權重%": signal.risk.suggested_weight_pct,
            "最大虧損%NAV": signal.risk.max_loss_pct_nav,
            "證據數": len(signal.evidence),
            "反證數": len(signal.counter_evidence),
            "證據類別": "、".join(sorted({item.kind for item in signal.evidence})),
            "綁定限制": signal.risk.binding_constraint,
        }
        for signal in state.signals
    ]
)

controls = st.columns(4)
directions = controls[0].multiselect("方向", ["做多", "做空"], default=["做多", "做空"])
severities = controls[1].multiselect(
    "警示級別",
    sorted({signal.severity.label for signal in state.signals}),
    default=sorted({signal.severity.label for signal in state.signals}),
)
kinds = sorted({item.kind for signal in state.signals for item in signal.evidence})
required = controls[2].multiselect("必須包含證據類別", kinds)
tradeable_only = controls[3].checkbox("僅顯示風控已放行", value=False)

selected = [
    signal
    for signal in state.signals
    if signal.direction_label in directions
    and signal.severity.label in severities
    and (not required or required <= {item.kind for item in signal.evidence})
    and (not tradeable_only or signal.risk.is_tradeable)
]

st.dataframe(
    frame[frame["代號"].isin({signal.symbol for signal in selected})],
    hide_index=True,
    use_container_width=True,
    column_config={
        "信心": st.column_config.ProgressColumn("信心", min_value=0.0, max_value=100.0, format="%.0f%%"),
        "建議權重%": st.column_config.NumberColumn(format="%.2f%%"),
        "最大虧損%NAV": st.column_config.NumberColumn(format="%.3f%%"),
    },
)

st.divider()
st.subheader(f"訊號卡（{len(selected)} 條）")
store = get_store()
for signal in selected:
    render_signal_card(signal, store=store, key_prefix="flow_")

if state.suppressed_signals:
    st.caption(
        f"另有 {state.suppressed_signals} 條候選訊號被每日訊號預算壓下。"
        "若要放行更多，請調高側邊欄的訊號預算——但先想清楚交易員一天能認真看幾條。"
    )
