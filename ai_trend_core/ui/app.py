"""
ui/app.py
AI Trend Core 可視化儀表板：即時看板、信號牆、圖表分析與 AI 思考過程展示。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import database
from core.quant_engine import DEFAULT_SYMBOLS, scan_market

st.set_page_config(page_title="AI Trend Core", layout="wide", page_icon="📈")
database.init_db()

st.title("📈 AI Trend Core ｜ 自主市場掃描與交易決策平台")
st.caption("即時量化訊號 × AI 智能體團隊分析")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    symbols = st.multiselect("監控標的", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS)
    refresh = st.button("🔄 立即重新掃描")

if "snapshots" not in st.session_state or refresh:
    with st.spinner("正在抓取市場數據並計算指標..."):
        st.session_state["snapshots"] = scan_market(symbols or DEFAULT_SYMBOLS)

snapshots = st.session_state["snapshots"]


def risk_level(z_score: float) -> str:
    """依 Z-Score 絕對值換算出簡易風險等級標籤。"""
    abs_z = abs(z_score)
    if abs_z >= 2.5:
        return "🔴 高風險"
    if abs_z >= 2.0:
        return "🟠 中高風險"
    if abs_z >= 1.0:
        return "🟡 中等風險"
    return "🟢 正常"


st.subheader("📊 即時看板")
board_rows = [
    {
        "標的": symbol,
        "價格": round(snap.price, 2),
        "均值": round(snap.mean, 2),
        "Z-Score": round(snap.z_score, 2),
        "訊號": snap.signal,
        "風險等級": risk_level(snap.z_score),
    }
    for symbol, snap in snapshots.items()
]
if board_rows:
    st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)
else:
    st.info("尚無掃描資料，請於左側選擇標的並重新掃描。")

st.subheader("🧱 信號牆（Signal Wall）")
recent_signals = database.get_recent_signals(limit=20)
if not recent_signals:
    st.info("目前資料庫中尚無 AI 交易訊號，請先執行 main_loop.py 進行背景掃描。")
else:
    for sig in recent_signals:
        confidence = sig.get("confidence") or 0
        if confidence >= 0.75:
            color = "#1DB954"  # 高信心：綠色
        elif confidence >= 0.5:
            color = "#F5A623"  # 中信心：橙色
        else:
            color = "#E74C3C"  # 低信心：紅色

        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 10px 16px;
                        margin-bottom: 10px; background-color: #1a1d24; border-radius: 6px;">
                <b>{sig['symbol']}</b>
                動作：<b>{sig.get('action') or 'N/A'}</b>
                信心評分：<b>{confidence:.0%}</b>
                <span style="color:gray;">{sig['created_at']}</span><br/>
                進場：{sig.get('entry')}　停利：{sig.get('take_profit')}　停損：{sig.get('stop_loss')}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.subheader("📉 圖表分析")
selected_symbol = st.selectbox("選擇標的檢視走勢圖", list(snapshots.keys()) if snapshots else [])
if selected_symbol:
    history = snapshots[selected_symbol].history.tail(120)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=history.index, open=history["open"], high=history["high"],
            low=history["low"], close=history["close"], name="價格",
        )
    )
    fig.add_trace(go.Scatter(x=history.index, y=history["bb_mean"], name="20 期均線",
                              line=dict(color="yellow", width=1)))
    fig.add_trace(go.Scatter(x=history.index, y=history["bb_upper"], name="布林上軌",
                              line=dict(color="red", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=history.index, y=history["bb_lower"], name="布林下軌",
                              line=dict(color="green", width=1, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
        title=f"{selected_symbol} 價格走勢與布林帶",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧠 AI 思考過程")
if recent_signals:
    latest = recent_signals[0]
    with st.expander(f"查看 {latest['symbol']} 最新的 AI 推理過程", expanded=False):
        st.markdown("**情緒分析師報告：**")
        st.write(latest.get("sentiment_summary") or "無資料")
        st.markdown("**首席策略官最終建議：**")
        st.write(latest.get("reasoning") or "無資料")
else:
    st.info("尚無 AI 推理紀錄。")
