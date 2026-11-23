"""Factor research workbench.

Where a hypothesis gets tested. Every control here changes a real assumption in
the backtest, and the page shows what that assumption is worth: switch the price
limits off and watch the Sharpe move, triple the costs and watch the edge go.

Nothing on this page can approve anything. It runs experiments and reports the
gate's verdict; promotion happens in the registry, under the registry's rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_platform.console import (
    build_universe,
    industries,
    model_id_for,
    registry,
    run_experiment,
    run_validation,
)
from quant_platform.research.costs import CostModel, cost_report
from quant_platform.research.factors import FACTORS
from trading_desk.ui import page_setup

page_setup("因子研究")

NAMES = {"mean_reversion": "均值回歸", "momentum": "動能", "low_volatility": "低波動"}

st.title("因子研究台")
st.caption(
    "假說 → 因子 → point-in-time 回測 → 成本 → 驗證閘門。"
    "這一層決定一個想法能不能動用資金；訊號看板只是它的輸出端。"
)

# --- Controls -------------------------------------------------------------

with st.sidebar:
    st.markdown("### 實驗設定")

    factor_key = st.selectbox(
        "因子", list(FACTORS), format_func=lambda k: f"{NAMES.get(k, k)}（{k}）"
    )
    factor = FACTORS[factor_key]

    with st.expander("樣本", expanded=True):
        min_cap = st.slider("最小市值（十億元）", 0.0, 50.0, 5.0, 1.0)
        picked = tuple(st.multiselect("產業（留空為全市場）", industries()))
        start_year = st.slider("起始年份", 2013, 2024, 2013)

    with st.expander("策略參數", expanded=True):
        holding = st.select_slider("持有期間（交易日）", [5, 10, 21, 42, 63], value=21)
        quantile = st.select_slider("分位", [0.05, 0.1, 0.2, 0.3], value=0.1,
                                    format_func=lambda q: f"{q:.0%}")
        long_short = st.radio("方向", ["多空", "只做多"], horizontal=True) == "多空"

    with st.expander("市場摩擦", expanded=True):
        respect_limits = st.checkbox(
            "套用漲跌停限制", value=True,
            help="漲停買不到、跌停賣不掉也放空不了。關掉可以量出這條規則值多少 Sharpe。",
        )
        discount = st.slider("手續費折扣", 0.2, 1.0, 0.6, 0.05,
                             help="0.6 = 6 折，法定費率 0.1425% 的常見零售折扣")
        slippage = st.slider("額外滑價（bps）", 0.0, 20.0, 3.0, 1.0)
        participation = st.slider("成交量參與率", 0.005, 0.10, 0.01, 0.005,
                                  format="%.3f",
                                  help="資料庫沒有成交量，這是假設值而非量測值")
        cost_multiple = st.select_slider("成本倍數", [1.0, 2.0, 3.0, 5.0], value=1.0)

    with st.expander("驗證"):
        trials = st.number_input(
            "多重檢定嘗試次數", 1, 500, 50,
            help="誠實填寫。填小會讓 Deflated Sharpe 過度樂觀——那是在騙自己。",
        )
        null_draws = st.slider("隨機化對照次數", 4, 16, 8)

args = dict(
    factor_key=factor_key, holding_days=holding, quantile=quantile,
    long_short=long_short, respect_limits=respect_limits,
    min_market_cap=min_cap * 1e9, industries_selected=picked,
    commission_discount=discount, slippage_bps=slippage,
    participation=participation, start_year=start_year,
)

# --- Hypothesis -----------------------------------------------------------

with st.container(border=True):
    st.markdown(f"**假說**　{factor.claim}")
    st.markdown(f"**預期失效條件**　{factor.fails_when}")
    st.caption(
        "先寫下假說與失效條件，再看數字。順序顛倒過來就是過度配適的開始："
        "先看到結果再編故事，任何結果都編得出理由。"
    )

universe_size = len(build_universe(min_cap * 1e9, picked))
st.caption(f"樣本：{universe_size} 檔｜起始 {start_year} 年｜"
           f"{'多空各' if long_short else '只做多'} {quantile:.0%} 分位｜{holding} 日持有")

if universe_size < 40:
    st.error(f"universe 只有 {universe_size} 檔，樣本太小，無法做橫斷面排序。請放寬市值或產業條件。")
    st.stop()

# --- Backtest -------------------------------------------------------------

with st.spinner(f"回測 {universe_size} 檔…"):
    run = run_experiment(**args, signal_delay_days=0, cost_multiple=cost_multiple)

if "error" in run:
    st.error(run["error"])
    st.stop()

summary = run["summary"]
columns = st.columns(5)
columns[0].metric("Sharpe", f"{run['sharpe']:.2f}")
columns[1].metric("年化報酬", f"{summary['annual_return']:+.2%}")
columns[2].metric("淨報酬／期", f"{summary['net_return_per_period']:+.3%}",
                  delta=f"毛 {summary['gross_return_per_period']:+.3%}", delta_color="off")
columns[3].metric("最大回撤", f"{summary['max_drawdown']:.1%}")
columns[4].metric("換手率", f"{run['turnover']:.0%}",
                  delta=f"成本 {summary['cost_per_period']:.3%}/期", delta_color="off")

if respect_limits and run["blocked_per_period"] > 0:
    st.caption(
        f"每期平均有 {run['blocked_per_period']:.1f} 檔因漲跌停無法成交而剔除。"
        "關掉側邊欄的漲跌停限制可以看出這條規則值多少績效。"
    )

figure = go.Figure()
figure.add_scatter(x=run["equity"].index, y=run["equity"].values, name="策略（淨值）",
                   line=dict(color="#38BDF8", width=2))
figure.add_scatter(x=run["benchmark_equity"].index, y=run["benchmark_equity"].values,
                   name="0050.TW", line=dict(color="#94A3B8", width=1.5, dash="dot"))
figure.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=340, margin=dict(l=10, r=10, t=36, b=10),
    title="累積淨值（扣成本後）", yaxis_title="成長倍數",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
)
st.plotly_chart(figure, use_container_width=True)

if long_short:
    st.caption(
        "多空市場中性策略的比較基準是現金，不是 0050。"
        "在年化 +22% 的大多頭裡拿指數當基準，測到的是市場而不是策略——"
        "0050 曲線在此僅供對照。"
    )

tab_year, tab_cost = st.tabs(["逐年表現", "成本拆解"])

with tab_year:
    yearly = run["by_year"]
    if yearly.empty:
        st.info("樣本期間不足以分年。")
    else:
        display = yearly.assign(
            total_return=lambda f: f["total_return"] * 100,
            mean_return=lambda f: f["mean_return"] * 100,
            hit_rate=lambda f: f["hit_rate"] * 100,
            worst=lambda f: f["worst"] * 100,
            benchmark=lambda f: f["benchmark"] * 100,
        )
        st.dataframe(
            display, hide_index=True, use_container_width=True,
            column_config={
                "year": st.column_config.NumberColumn("年", format="%d"),
                "periods": st.column_config.NumberColumn("期數"),
                "total_return": st.column_config.NumberColumn("年報酬", format="%.2f%%"),
                "mean_return": st.column_config.NumberColumn("每期平均", format="%.2f%%"),
                "hit_rate": st.column_config.NumberColumn("命中率", format="%.0f%%"),
                "worst": st.column_config.NumberColumn("最差一期", format="%.2f%%"),
                "benchmark": st.column_config.NumberColumn("0050", format="%.2f%%"),
            },
        )
        positive = int((yearly["total_return"] > 0).sum())
        st.caption(f"{positive}/{len(yearly)} 個年份為正。只在一兩年賺錢的因子，賺的是運氣不是 edge。")

with tab_cost:
    model = CostModel(commission_discount=discount, extra_slippage_bps=slippage,
                      participation=participation)
    sample_prices = pd.Series([15.0, 45.0, 88.0, 250.0, 640.0, 1180.0])
    report = cost_report(model, sample_prices, pd.Series([0.35] * 6))
    report.index = [f"{price:,.0f} 元" for price in sample_prices]
    st.dataframe(report.round(1), use_container_width=True)
    st.caption(
        f"單位為 bps。最低可行下單金額 {model.minimum_viable_notional():,.0f} 元——"
        "低於此，20 元的最低手續費會超過按比例計算的手續費。"
        "低價股與高價股成本最高，原因不同：前者 tick 佔價格比例大，後者 tick 跳到 5 元。"
    )

# --- Validation gate ------------------------------------------------------

st.divider()
st.subheader("驗證閘門")
st.caption(
    "閘門的工作是說不。每一項檢驗都因為某個策略曾經通過粗糙的審查、然後賠錢而存在。"
)

if st.button("執行完整驗證", type="primary",
             help="基準回測 + 成本壓力 ×3 + 執行延遲 +1 日 + 隨機化對照，需要一到兩分鐘"):
    st.session_state["validation_args"] = args | {"trials": int(trials), "null_draws": null_draws}

validation_args = st.session_state.get("validation_args")
if validation_args and validation_args.get("factor_key") == factor_key:
    with st.spinner(f"執行 {3 + validation_args['null_draws']} 次回測…"):
        report = run_validation(**validation_args)

    if "error" in report:
        st.error(report["error"])
    else:
        record = report["record"]
        metrics = report["metrics"]
        (st.success if record["approved"] else st.error)(report["verdict"])

        columns = st.columns(4)
        columns[0].metric("Sharpe", f"{metrics['sharpe']:.2f}")
        columns[1].metric(
            "95% bootstrap 區間",
            f"[{metrics['sharpe_ci_low']:.2f}, {metrics['sharpe_ci_high']:.2f}]",
            help="下界必須為正。跨越零代表這個 edge 與『沒有 edge』無法區分。",
        )
        columns[2].metric("Deflated Sharpe", f"{metrics['deflated_sharpe']:.3f}",
                          help=f"已針對 {validation_args['trials']} 次嘗試做多重檢定修正")
        columns[3].metric("成本壓力 ×3", f"{report['stressed_net']:+.4%}/期")

        st.dataframe(report["gates"], hide_index=True, use_container_width=True)

        nulls = report["nulls"]
        if nulls:
            st.caption(
                f"隨機化對照：把因子分數橫斷面重排後跑 {len(nulls)} 次，"
                f"Sharpe 介於 {min(nulls):+.2f} 與 {max(nulls):+.2f}。"
                "打亂後仍有績效，代表 edge 來自流程而非訊號——通常是資料洩漏。"
            )

        st.divider()
        model_id = model_id_for(factor_key, holding, quantile, respect_limits)
        st.markdown(f"**登錄到 model registry**　`{model_id}`")
        note = st.text_input("備註", value=factor.claim, max_chars=200)
        if st.button("寫入 registry"):
            store = registry()
            if model_id in store.all():
                st.warning(f"{model_id} 已經在 registry 中，未重複寫入。")
            else:
                store.register(
                    model_id=model_id,
                    factor_id=FACTORS[factor_key].id,
                    spec=report["spec"] | {"universe_size": report["universe_size"]},
                    validation=record,
                    notes=note,
                )
                st.success(f"已登錄 {model_id}（階段：研究中）。到「模型 Registry」頁面管理晉級。")
        if not record["approved"]:
            st.caption(
                "未通過的模型同樣要登錄。知道什麼失敗過、為什麼失敗，"
                "才不會每季重測同一個想法。"
            )
elif validation_args:
    st.info("因子已更改，請重新執行驗證。")
