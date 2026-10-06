"""Model registry: the control plane's veto, with a human at the last step.

Every button here goes through the same gates a script would hit. If the
registry refuses a promotion, this page shows the refusal rather than working
around it — an interface that could approve what the CLI rejects would make the
whole control plane decorative.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_platform.console import registry
from quant_platform.control.registry import (
    ALLOWED,
    MIN_DAYS_IN_STAGE,
    REQUIRES_HUMAN_APPROVAL,
    PromotionRefused,
    Stage,
)
from trading_desk.ui import page_setup

page_setup("模型 Registry")

st.title("模型 Registry")
st.caption(
    "模型走一條固定的路，一次一步，每一步都有進入條件。"
    "這裡沒有覆寫按鈕：要爭論的是閘門的參數，在審查時爭論，不是在個案上爭論。"
)

store = registry()
records = store.all()

# --- Lifecycle map --------------------------------------------------------

with st.container(border=True):
    st.markdown("**生命週期**")
    st.code(
        "research → validated → shadow → paper → canary → production\n"
        "                                                     ↓\n"
        "                                                  retired",
        language="text",
    )
    left, right = st.columns(2)
    left.markdown(
        "**停留時間下限**\n\n"
        + "\n".join(f"- {stage.label}：{days} 天" for stage, days in MIN_DAYS_IN_STAGE.items())
    )
    right.markdown(
        "**需具名人工核准**\n\n"
        + "\n".join(f"- {stage.label}" for stage in REQUIRES_HUMAN_APPROVAL)
        + "\n\n系統可以自己搜尋、驗證、寫報告，但按下動用資金那個按鈕的是人。"
    )
    st.caption(
        "停留時間不是官僚程序。它是唯一能看見「模型成本」與「實際成交成本」落差的方法，"
        "而那個落差沒有任何回測顯示得出來。"
    )

st.divider()

if not records:
    st.info("registry 目前是空的。")
    st.caption("到「因子研究」頁面執行完整驗證後，即可將模型登錄進來。")
    st.stop()

# --- Overview -------------------------------------------------------------

frame = pd.DataFrame(
    [
        {
            "模型": record.model_id,
            "因子": record.factor_id,
            "階段": record.stage.label,
            "驗證": "通過" if record.validated else "未通過",
            "Sharpe": record.validation.get("metrics", {}).get("sharpe"),
            "DSR": record.validation.get("metrics", {}).get("deflated_sharpe"),
            "停留天數": round(record.days_in_stage(), 1),
            "可動用資金": "是" if record.stage.trades_real_money else "否",
        }
        for record in records.values()
    ]
)
st.subheader(f"已登錄模型（{len(frame)}）")
st.dataframe(
    frame, hide_index=True, use_container_width=True,
    column_config={
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "DSR": st.column_config.NumberColumn("Deflated Sharpe", format="%.3f"),
        "停留天數": st.column_config.NumberColumn(format="%.1f"),
    },
)

deployable = store.deployable()
if deployable:
    st.warning(f"目前有 {len(deployable)} 個模型處於可動用資金的階段："
               + "、".join(record.model_id for record in deployable))
else:
    st.success("目前沒有任何模型處於可動用資金的階段。")

st.divider()

# --- Single model ---------------------------------------------------------

st.subheader("模型詳情與晉級")
model_id = st.selectbox("選擇模型", sorted(records))
record = records[model_id]

columns = st.columns(4)
columns[0].metric("目前階段", record.stage.label)
columns[1].metric("停留天數", f"{record.days_in_stage():.1f}")
columns[2].metric("驗證結果", "通過" if record.validated else "未通過")
columns[3].metric("可動用資金", "是" if record.stage.trades_real_money else "否")

st.markdown(f"**判定**　{record.validation.get('verdict', '（無紀錄）')}")
if record.notes:
    st.caption(f"備註：{record.notes}")

hard = record.validation.get("hard_failures", [])
if hard:
    st.error("未通過的硬性門檻：" + "、".join(hard))
soft = record.validation.get("soft_failures", [])
if soft:
    st.warning("警示項目：" + "、".join(soft))

with st.expander("驗證明細"):
    gates = record.validation.get("gates", [])
    if gates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "檢驗": gate["name"],
                        "結果": "通過" if gate["passed"] else "未通過",
                        "門檻": gate["threshold"],
                        "類型": "硬性" if gate["hard"] else "警示",
                    }
                    for gate in gates
                ]
            ),
            hide_index=True, use_container_width=True,
        )
    st.json(record.spec)

with st.expander("轉換紀錄（append-only）"):
    st.dataframe(pd.DataFrame(record.history), hide_index=True, use_container_width=True)
    st.caption("永不原地修改。虧損之後被問的問題永遠是「我們當時知道什麼、什麼時候知道的」。")

# --- Promotion ------------------------------------------------------------

legal = sorted(ALLOWED[record.stage], key=lambda stage: stage.value)
if not legal:
    st.info(f"{record.stage.label} 是終態，沒有下一步。")
    st.stop()

st.markdown("**晉級**")
target = st.selectbox(
    "下一階段", legal, format_func=lambda stage: stage.label,
    help="只列出合法的下一步。跳級本身就是繞過閘門。",
)
needs_approver = target in REQUIRES_HUMAN_APPROVAL
approver = st.text_input(
    "核准人",
    placeholder="必填：動用資金需要具名負責人" if needs_approver else "選填",
)
reason = st.text_input("理由", placeholder="例如：paper 階段追蹤誤差在容忍範圍內")

if st.button(f"晉級至 {target.label}", type="primary"):
    try:
        store.promote(model_id, target, approver=approver or None, reason=reason)
        st.success(f"{model_id} 已進入 {target.label}。")
        st.rerun()
    except PromotionRefused as error:
        st.error(f"**晉級被拒絕**：{error}")
        st.caption(
            "這不是錯誤訊息，是閘門在運作。要通過，得先滿足條件——"
            "而不是找一個能繞過條件的介面。"
        )
