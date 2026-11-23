"""Risk dashboard: budget usage, cluster exposure, and distance to each breaker."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_desk.contracts import Position
from trading_desk.risk import RiskEngine, RiskLimits, build_clusters, portfolio_risk_summary
from trading_desk.ui import page_setup, cached_market, run_from_settings, sidebar_controls

page_setup("風險面板")
settings = sidebar_controls()

st.title("風險面板")
st.caption(
    "風控是策略的上級，不是策略的一部分。這一頁顯示的每個限額，策略都無法自行放寬。"
)

state = run_from_settings(settings)
limits = RiskLimits(
    max_weight_per_name=settings["max_weight_per_name"],
    target_portfolio_vol=settings["target_portfolio_vol"],
    min_conviction=settings["min_conviction"],
)
engine = RiskEngine(limits=limits)

# --- Breaker status -------------------------------------------------------

st.subheader("帳戶層熔斷")
breaker = state.breaker
if breaker.tripped:
    st.error(f"**{breaker.level} 觸發**：{breaker.reason}｜動作：{breaker.action}｜恢復：{breaker.recovery}")
else:
    st.success(breaker.reason)

thresholds = [
    ("當日虧損", settings["daily_pnl_pct"], -2.0, -4.0),
    ("目前回撤", settings["drawdown_pct"], -10.0, -20.0),
]
columns = st.columns(len(thresholds))
for column, (label, current, warn, halt) in zip(columns, thresholds):
    used = min(1.0, abs(current) / abs(halt)) if halt else 0.0
    column.metric(label, f"{current:+.1f}%", delta=f"距停機 {abs(halt - current):.1f}pp", delta_color="off")
    column.progress(used, text=f"警戒 {warn:g}%　停機 {halt:g}%")

st.caption("恢復比觸發困難是刻意的：自動觸發、人工恢復，只有明確可逆的技術性狀況才自動恢復。")

st.divider()

# --- Proposed book --------------------------------------------------------

st.subheader("今日建議部位的組合風險")
tradeable = [signal for signal in state.signals if signal.risk.is_tradeable]
if not tradeable:
    st.info("目前沒有風控放行的訊號，組合風險為零。")
    st.stop()

data = cached_market(settings["min_market_cap"] * 1e9, settings["industries"])
returns = (
    data.prices[[signal.symbol for signal in tradeable if signal.symbol in data.prices.columns]]
    .ffill()
    .pct_change(fill_method=None)
    .tail(120)
)
clusters = build_clusters(returns)
positions = [
    Position(
        symbol=signal.symbol,
        name=signal.name,
        direction=signal.direction,
        weight_pct=signal.risk.suggested_weight_pct,
        entry_price=signal.risk.target_level or 0.0,
        entry_date=signal.created_at,
        stop_level=signal.risk.stop_level,
        target_level=signal.risk.target_level,
        signal_id=signal.signal_id,
        cluster=clusters.get(signal.symbol, signal.symbol),
    )
    for signal in tradeable
]
summary = portfolio_risk_summary(positions, data.prices.ffill().iloc[-1], clusters)

metrics = st.columns(4)
metrics[0].metric("總曝險", f"{summary['gross_exposure']:.1f}%", help=f"上限 {limits.max_gross_exposure:.0%}")
metrics[1].metric("淨曝險", f"{summary['net_exposure']:+.1f}%", help=f"上限 {limits.max_net_exposure:.0%}")
metrics[2].metric(
    "有效持倉數", summary["effective_positions"],
    help="考慮相關性後真正獨立的賭注數。分散在 20 檔一起跌的股票，不是分散。",
)
metrics[3].metric("持倉檔數", summary["positions"], help=f"上限 {limits.max_positions}")

cluster_frame = pd.DataFrame(
    [
        {"相關性叢集": name, "曝險%": value, "上限%": limits.max_weight_per_cluster * 100}
        for name, value in summary["cluster_exposure"].items()
    ]
)
if not cluster_frame.empty:
    figure = go.Figure()
    figure.add_bar(
        x=cluster_frame["曝險%"], y=cluster_frame["相關性叢集"],
        orientation="h", marker_color="#38BDF8", name="曝險",
    )
    figure.add_vline(
        x=limits.max_weight_per_cluster * 100, line_dash="dash", line_color="#F97316",
        annotation_text="叢集上限",
    )
    figure.update_layout(
        height=max(240, 42 * len(cluster_frame)), margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="相關性叢集曝險（限額綁在叢集，不是綁在個股）",
    )
    st.plotly_chart(figure, use_container_width=True)

st.divider()
st.subheader("每筆建議的定量依據")
sizing = pd.DataFrame(
    [
        {
            "代號": signal.symbol,
            "名稱": signal.name,
            "方向": signal.direction_label,
            "信心": signal.conviction * 100,
            "叢集": clusters.get(signal.symbol, signal.symbol),
            "建議權重%": signal.risk.suggested_weight_pct,
            "停損": signal.risk.stop_level,
            "最大虧損%NAV": signal.risk.max_loss_pct_nav,
            "綁定限制": signal.risk.binding_constraint,
        }
        for signal in tradeable
    ]
)
st.dataframe(
    sizing, hide_index=True, use_container_width=True,
    column_config={
        "信心": st.column_config.ProgressColumn("信心", min_value=0.0, max_value=100.0, format="%.0f%%"),
    },
)
st.caption(
    "「綁定限制」欄位回答的是：為什麼這筆只給了這個權重。"
    "沒有這一欄，交易員就無法判斷該調整信心還是該調整限額。"
)

with st.expander("定量公式"):
    st.markdown(
        f"""
```
單一部位波動預算 = 組合波動目標 {limits.target_portfolio_vol:.0%} ÷ √最大持倉數 {limits.max_positions}
波動率目標權重   = 單一部位波動預算 ÷ 標的年化波動
信心縮放         = 波動率目標權重 × max(0, 2×信心 − 1)
分數凱利上限     = {limits.kelly_fraction:g} × (2×信心 − 1) ÷ 標的年化波動²
單一標的上限     = {limits.max_weight_per_name:.0%}
單筆虧損上限     = {limits.max_loss_per_trade_nav:.0%} NAV ÷ 停損距離
最終權重         = 以上取最小值
```
除以 √最大持倉數是必要的：把每檔都定量到完整的組合波動目標，
二十檔上滿之後整個帳戶的風險會是目標的數倍。

分數凱利而非全凱利，是因為全凱利假設你知道真實勝率。你不知道。
"""
    )
