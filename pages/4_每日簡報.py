"""Daily brief: the one-page read before the open and the review after the close."""

from __future__ import annotations

import streamlit as st

from trading_desk.brief import post_market_brief, pre_market_brief
from trading_desk.store import evaluate_signals
from trading_desk.ui import (
    cached_market,
    page_setup,
    get_store,
    run_from_settings,
    sidebar_controls,
)

page_setup("每日簡報")
settings = sidebar_controls()

st.title("每日簡報")
st.caption("超過一頁的簡報不會被讀完，所以這裡刻意寫得短。細節在其他頁面。")

state = run_from_settings(settings)
store = get_store()
data = cached_market(settings["min_market_cap"] * 1e9, settings["industries"])

pre_tab, post_tab = st.tabs(["盤前簡報", "盤後覆盤"])

with pre_tab:
    brief = pre_market_brief(state, positions=[], last_prices=data.prices.ffill().iloc[-1])
    st.markdown(brief)
    st.download_button(
        "下載盤前簡報（Markdown）",
        brief,
        file_name=f"pre-market-{state.as_of:%Y%m%d}.md",
        mime="text/markdown",
    )

with post_tab:
    evaluation = evaluate_signals(store, data.prices, horizon_days=5)
    review = post_market_brief(state, evaluation)
    st.markdown(review)
    st.download_button(
        "下載盤後覆盤（Markdown）",
        review,
        file_name=f"post-market-{state.as_of:%Y%m%d}.md",
        mime="text/markdown",
    )

st.divider()
if st.button("將本次執行寫入紀錄", help="寫入 append-only 執行日誌，供日後覆盤與回放"):
    store.record_run(state)
    st.success(f"已寫入執行 {state.run_id}，共 {len(state.signals)} 條訊號。")
