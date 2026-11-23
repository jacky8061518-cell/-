from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sector_rotation.config import BENCHMARK, DEFENSIVE_ASSET
from sector_rotation.data import download_adjusted_prices, generate_demo_prices
from sector_rotation.fund_flow import (
    calculate_daily_group_flows,
    calculate_fund_flow_signals,
)
from sector_rotation.holdings import analyze_holding_leadership, fetch_top_holdings
from sector_rotation.metrics import benchmark_returns, drawdown, equity_curve, performance_summary
from sector_rotation.rrg import (
    QUADRANT_LABELS,
    build_rotation_summary,
    calculate_group_rrg,
    classify_quadrant,
)
from sector_rotation.strategy import (
    PERIODS_PER_YEAR,
    BacktestConfig,
    compute_momentum_components,
    run_backtest,
)
from sector_rotation.taiwan import (
    TW_BENCHMARK,
    TW_DEFENSIVE_ASSET,
    TW_THEME_CODES,
    assets_from_official_industries,
    assets_from_taiwan_security_master,
    assets_from_taiwan_themes,
    custom_taiwan_assets,
    fetch_taiwan_security_master,
    official_industry_groups,
)
from sector_rotation.universe import (
    UNIVERSE_GROUPS,
    AssetInfo,
    assets_for,
    custom_assets,
    groups_for,
)


FREQUENCY_SETTINGS = {
    "Daily": {
        "label": "日線",
        "unit": "交易日",
        "lookbacks": {5: 10, 21: 20, 63: 30, 126: 40},
        "volatility_window": 63,
    },
    "Weekly": {
        "label": "週線",
        "unit": "週",
        "lookbacks": {4: 10, 13: 20, 26: 30, 52: 40},
        "volatility_window": 13,
    },
    "Monthly": {
        "label": "月線",
        "unit": "月",
        "lookbacks": {1: 10, 3: 20, 6: 30, 12: 40},
        "volatility_window": 6,
    },
}

DATA_PIPELINE_VERSION = "0.8.0-fund-flow-first"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAIWAN_PRICE_DATABASE = PROJECT_ROOT / "data" / "databases" / "tw" / "adjusted-prices.parquet"
TAIWAN_FLOW_DATABASE = PROJECT_ROOT / "data" / "databases" / "tw" / "institutional-flows.parquet"


# Page configuration and shared styling now live in the app entry point, so
# this page only declares the pieces specific to the research lab.
from trading_desk.ui import inject_styles

inject_styles()


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    pipeline_version: str,
) -> pd.DataFrame:
    del pipeline_version  # Deliberately part of the Streamlit cache key.
    return download_adjusted_prices(list(tickers), start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_split_adjusted_close(
    ticker: str,
    start: date,
    end: date,
    pipeline_version: str,
) -> pd.DataFrame:
    """Load price-only history on a continuous post-split unit basis."""
    del pipeline_version
    return download_adjusted_prices(
        [ticker],
        start,
        end,
        auto_adjust=False,
    )


@st.cache_data(show_spinner=False)
def load_demo_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    return generate_demo_prices(tickers=list(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def load_top_holdings(etfs: tuple[str, ...]) -> pd.DataFrame:
    return fetch_top_holdings(list(etfs))


@st.cache_data(ttl=86400, show_spinner=False)
def load_taiwan_security_master() -> pd.DataFrame:
    return fetch_taiwan_security_master()


@st.cache_data(ttl=3600, show_spinner=False)
def load_taiwan_price_database(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    pipeline_version: str,
) -> pd.DataFrame:
    """Read the daily-refreshed Taiwan database and fetch only cache misses."""
    del pipeline_version
    cached = pd.DataFrame()
    if TAIWAN_PRICE_DATABASE.exists():
        cached = pd.read_parquet(TAIWAN_PRICE_DATABASE)
        cached.index = pd.to_datetime(cached.index)
        cached = cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        cached = cached.reindex(columns=[ticker for ticker in tickers if ticker in cached])

    missing = [
        ticker
        for ticker in tickers
        if ticker not in cached or cached[ticker].notna().sum() == 0
    ]
    fresh = pd.DataFrame()
    if missing:
        fresh = download_adjusted_prices(missing, start, end + timedelta(days=1))
    prices = pd.concat([cached, fresh], axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    return prices.reindex(columns=tickers).dropna(how="all").ffill()


@st.cache_data(ttl=3600, show_spinner=False)
def load_taiwan_institutional_flows(pipeline_version: str) -> pd.DataFrame:
    del pipeline_version
    if not TAIWAN_FLOW_DATABASE.exists():
        return pd.DataFrame()
    flows = pd.read_parquet(TAIWAN_FLOW_DATABASE)
    flows["Date"] = pd.to_datetime(flows["Date"])
    return flows


@st.cache_data(ttl=1800, show_spinner=False)
def load_ticker_news(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Load recent Yahoo Finance headlines for selected flow leaders."""
    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).get_news(count=5)
        except Exception:
            continue
        for item in items or []:
            content = item.get("content", item)
            canonical = content.get("canonicalUrl", {})
            provider = content.get("provider", {})
            rows.append(
                {
                    "Ticker": ticker,
                    "Title": content.get("title", ""),
                    "Summary": content.get("summary", ""),
                    "Publisher": (
                        provider.get("displayName", "")
                        if isinstance(provider, dict)
                        else str(provider)
                    ),
                    "Published": content.get("pubDate", ""),
                    "URL": (
                        canonical.get("url", "")
                        if isinstance(canonical, dict)
                        else str(canonical)
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_tickers(raw: str) -> list[str]:
    return [
        token.upper()
        for token in re.split(r"[\s,;]+", raw.strip())
        if token and re.fullmatch(r"[A-Za-z0-9.^=-]+", token)
    ]


def format_percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def format_number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def performance_chart(
    strategy: pd.Series,
    benchmark: pd.Series,
    benchmark_label: str,
    price_benchmark: pd.Series | None = None,
    price_benchmark_label: str | None = None,
) -> go.Figure:
    curves = [
        (equity_curve(strategy) - 1).rename("Strategy"),
        (equity_curve(benchmark) - 1).rename(benchmark_label),
    ]
    if (
        price_benchmark is not None
        and not price_benchmark.empty
        and price_benchmark_label is not None
    ):
        curves.append(
            (equity_curve(price_benchmark) - 1).rename(price_benchmark_label)
        )
    frame = pd.concat(curves, axis=1).dropna(how="all")
    figure = px.line(
        frame,
        labels={"value": "累積報酬率", "index": "", "variable": ""},
        color_discrete_sequence=["#38bdf8", "#94a3b8", "#f59e0b"],
    )
    figure.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>累積報酬：%{y:.1%}<extra></extra>"
    )
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=430)
    return figure


def allocation_chart(
    weights: pd.DataFrame,
    metadata: dict[str, AssetInfo],
    defensive_ticker: str,
    defensive_label: str,
) -> go.Figure:
    visible = weights.loc[:, (weights.abs().sum() > 0)]
    renamed = visible.rename(
        columns={
            **{ticker: f"{ticker} · {info.name}" for ticker, info in metadata.items()},
            defensive_ticker: defensive_label,
        }
    )
    figure = px.area(
        renamed,
        labels={"value": "Portfolio weight", "index": "", "variable": ""},
    )
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=440)
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return figure


def rrg_chart(
    rs_ratio: pd.DataFrame,
    rs_momentum: pd.DataFrame,
    tail_length: int,
) -> go.Figure:
    """Create an RRG-style four-quadrant trajectory chart."""
    aligned = {
        group: pd.concat(
            [
                rs_ratio[group].rename("RS-Ratio"),
                rs_momentum[group].rename("RS-Momentum"),
            ],
            axis=1,
        )
        .dropna()
        .tail(tail_length)
        for group in rs_ratio.columns
    }
    aligned = {group: frame for group, frame in aligned.items() if not frame.empty}
    if not aligned:
        return go.Figure()

    all_x = np.concatenate([frame["RS-Ratio"].to_numpy() for frame in aligned.values()])
    all_y = np.concatenate(
        [frame["RS-Momentum"].to_numpy() for frame in aligned.values()]
    )
    x_half_range = max(1.0, float(np.nanmax(np.abs(all_x - 100))) * 1.25)
    y_half_range = max(0.25, float(np.nanmax(np.abs(all_y))) * 1.25)
    x_min, x_max = 100 - x_half_range, 100 + x_half_range
    y_min, y_max = -y_half_range, y_half_range

    quadrant_colors = {
        "Leading": "#22c55e",
        "Improving": "#38bdf8",
        "Weakening": "#f59e0b",
        "Lagging": "#ef4444",
    }
    figure = go.Figure()
    for x0, x1, y0, y1, color in [
        (100, x_max, 0, y_max, "rgba(34,197,94,0.10)"),
        (x_min, 100, 0, y_max, "rgba(56,189,248,0.10)"),
        (100, x_max, y_min, 0, "rgba(245,158,11,0.10)"),
        (x_min, 100, y_min, 0, "rgba(239,68,68,0.10)"),
    ]:
        figure.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            layer="below",
        )
    figure.add_vline(x=100, line_dash="dash", line_color="#94a3b8")
    figure.add_hline(y=0, line_dash="dash", line_color="#94a3b8")

    for group, frame in aligned.items():
        latest = frame.iloc[-1]
        quadrant = classify_quadrant(
            float(latest["RS-Ratio"]),
            float(latest["RS-Momentum"]),
        )
        labels = [""] * (len(frame) - 1) + [group]
        figure.add_trace(
            go.Scatter(
                x=frame["RS-Ratio"],
                y=frame["RS-Momentum"],
                mode="lines+markers+text",
                text=labels,
                textposition="top center",
                name=f"{group} · {QUADRANT_LABELS[quadrant]}",
                line={"color": quadrant_colors[quadrant], "width": 2.5},
                marker={
                    "color": quadrant_colors[quadrant],
                    "size": [6] * (len(frame) - 1) + [13],
                },
                customdata=frame.index.strftime("%Y-%m-%d"),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "日期：%{customdata}<br>"
                    "RS-Ratio：%{x:.2f}<br>"
                    "RS-Momentum：%{y:+.2f}%<extra></extra>"
                ),
            )
        )

    for x, y, label in [
        ((100 + x_max) / 2, y_max * 0.88, "領先 Leading"),
        ((x_min + 100) / 2, y_max * 0.88, "轉強 Improving"),
        ((100 + x_max) / 2, y_min * 0.88, "轉弱 Weakening"),
        ((x_min + 100) / 2, y_min * 0.88, "落後 Lagging"),
    ]:
        figure.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            opacity=0.65,
        )
    figure.update_xaxes(title="RS-Ratio（100 = 相對強弱中線）", range=[x_min, x_max])
    figure.update_yaxes(title="RS-Momentum（0 = 動能中線）", range=[y_min, y_max])
    figure.update_layout(
        title=f"產業 RRG 相對輪動軌跡（最近 {tail_length} 期）",
        hovermode="closest",
        legend_orientation="h",
        height=650,
    )
    return figure


st.title("Institutional Fund Flow & Rotation Research Lab")
st.caption(
    "資金先行、價格確認｜三大法人流向｜產業輪動｜個股帶領者｜美股＋台股雙資料庫"
)

with st.sidebar:
    st.header("研究範圍")
    market = st.segmented_control(
        "市場資料庫",
        ["美股", "台股"],
        default="美股",
    )
    if market is None:
        market = "美股"

    data_mode = st.radio(
        "資料來源",
        ["Live Yahoo Finance", "Offline demo"],
        index=1 if os.environ.get("SECTOR_ROTATION_TEST_MODE") == "1" else 0,
        help="Offline demo 使用合成資料，只能測試功能。",
    )

    if market == "美股":
        benchmark_ticker = BENCHMARK
        benchmark_label = "SPY"
        defensive_ticker = DEFENSIVE_ASSET
        defensive_label = "SHY · Defensive"
        universe_options = [*UNIVERSE_GROUPS, "Custom tickers — 自訂"]
        universe_name = st.selectbox("研究層級", universe_options, index=1)
        is_etf_universe = "ETFs" in universe_name

        if universe_name == "Custom tickers — 自訂":
            custom_input = st.text_area(
                "輸入 ticker",
                value="NVDA AMD AVGO MSFT AMZN GOOGL",
                help="用空格、逗號或分號分隔。",
            )
            metadata = custom_assets(parse_tickers(custom_input))
            selected_groups = ["Custom"]
        else:
            group_options = groups_for(universe_name)
            if universe_name in {"Broad sectors — ETFs", "Detailed industries — ETFs"}:
                default_groups = group_options
            else:
                default_groups = group_options[:3]
            selected_groups = st.multiselect(
                "產業／主題分類",
                group_options,
                default=default_groups,
            )
            metadata = assets_for(universe_name, selected_groups)
    else:
        benchmark_ticker = TW_BENCHMARK
        benchmark_label = "0050 · 含息總報酬"
        defensive_ticker = TW_DEFENSIVE_ASSET
        defensive_label = "00679B · 20年美債"
        taiwan_master = load_taiwan_security_master()
        universe_options = [
            "全部上市櫃股票與 ETF",
            "台股官方產業分類",
            "台股主題股票籃子",
            "台股全部 ETFs",
            "自訂台股代號",
        ]
        universe_name = st.selectbox("研究層級", universe_options, index=0)
        is_etf_universe = universe_name == "台股全部 ETFs"

        if universe_name == "全部上市櫃股票與 ETF":
            selected_markets = st.multiselect(
                "掛牌市場",
                ["上市", "上櫃"],
                default=["上市", "上櫃"],
            )
            selected_asset_types = st.multiselect(
                "資產類型",
                ["股票", "ETF"],
                default=["股票", "ETF"],
            )
            metadata = assets_from_taiwan_security_master(
                taiwan_master,
                markets=selected_markets,
                asset_types=selected_asset_types,
            )
            selected_groups = sorted({info.group for info in metadata.values()})
            st.caption(
                f"官方完整清單：目前選取 {len(metadata):,} 檔；"
                "價格資料由本機每日更新資料庫讀取。"
            )
        elif universe_name == "台股官方產業分類":
            group_options = official_industry_groups(taiwan_master)
            preferred = ["半導體業", "電腦及週邊設備", "電子零組件"]
            selected_groups = st.multiselect(
                "官方產業分類",
                group_options,
                default=[group for group in preferred if group in group_options],
                help="不再限制每產業 60 檔，選到的上市與上櫃公司會全部納入。",
            )
            metadata = assets_from_official_industries(
                taiwan_master,
                selected_groups,
            )
        elif universe_name == "台股主題股票籃子":
            group_options = list(TW_THEME_CODES)
            selected_groups = st.multiselect(
                "台股主題",
                group_options,
                default=group_options,
            )
            metadata = assets_from_taiwan_themes(taiwan_master, selected_groups)
        elif universe_name == "台股全部 ETFs":
            etf_master = taiwan_master[taiwan_master["Asset type"] == "ETF"]
            group_options = sorted(etf_master["Industry"].dropna().unique())
            selected_groups = st.multiselect(
                "ETF類型",
                group_options,
                default=group_options,
            )
            metadata = assets_from_taiwan_security_master(
                etf_master,
                asset_types=["ETF"],
                groups=selected_groups,
            )
        else:
            custom_input = st.text_area(
                "輸入台股代號",
                value="2330 2454 2317 2308 3711",
                help="直接輸入四位數代號，系統會自動判斷上市.TW或上櫃.TWO。",
            )
            metadata = custom_taiwan_assets(
                parse_tickers(custom_input),
                taiwan_master,
            )
            selected_groups = ["自訂台股"]

    frequency = st.segmented_control(
        "訊號與再平衡頻率",
        ["Daily", "Weekly", "Monthly"],
        default="Weekly",
        format_func=lambda value: FREQUENCY_SETTINGS[value]["label"],
    )
    if frequency is None:
        frequency = "Weekly"
    frequency_settings = FREQUENCY_SETTINGS[frequency]

    end_date = st.date_input(
        "回測結束日期",
        value=date.today(),
        min_value=date(2013, 2, 5),
        max_value=date.today(),
        key="research_end_date_v1",
        help="Yahoo Finance 的結束日期是排他的；系統會自動多抓一天。",
    )
    latest_allowed_start = end_date - timedelta(days=400)
    if (
        "research_start_date_v2" in st.session_state
        and st.session_state.research_start_date_v2 > latest_allowed_start
    ):
        st.session_state.research_start_date_v2 = date(2012, 1, 1)
    start_date = st.date_input(
        "回測開始日期",
        value=date(2012, 1, 1),
        max_value=latest_allowed_start,
        key="research_start_date_v2",
    )

    st.header("投資組合規則")
    asset_count = max(1, len(metadata))
    top_n = st.slider("持有排名前 N 個標的", 1, min(15, asset_count), min(5, asset_count))
    weighting = st.selectbox(
        "配置方式",
        [
            "Equal weight",
            "Momentum weight",
            "Inverse volatility",
            "Custom rank weight",
        ],
    )
    rank_weights: tuple[float, ...] | None = None
    if weighting == "Custom rank weight":
        st.caption("輸入每個排名的持倉比例；總和不是 100% 時會自動正規化。")
        entered_rank_weights = [
            st.number_input(
                f"Rank {rank} 權重（%）",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / top_n,
                step=1.0,
                key=f"custom_rank_weight_{rank}_{top_n}",
            )
            for rank in range(1, top_n + 1)
        ]
        rank_weights = tuple(entered_rank_weights)
        st.caption(
            f"輸入合計：{sum(entered_rank_weights):.1f}%｜"
            "回測時會依 Rank 1、Rank 2…套用到當期入選股票。"
        )
    risk_adjusted = st.toggle("使用風險調整動能", value=False)
    positive_filter = st.toggle("只持有正動能標的", value=True)
    use_defensive = st.toggle(
        f"無合格標的時持有 {defensive_ticker}",
        value=True,
    )
    cost_bps = st.number_input(
        "每次換手成本（bps）",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    st.header("多週期動能")
    lookback_weights: dict[int, float] = {}
    for periods, default_weight in frequency_settings["lookbacks"].items():
        lookback_weights[periods] = st.number_input(
            f"{periods} {frequency_settings['unit']}權重（%）",
            min_value=0.0,
            max_value=100.0,
            value=float(default_weight),
            step=1.0,
            key=f"lookback_weight_{frequency}_{periods}",
        )
    st.caption("修改任何參數後，系統會自動重新下載所需資料並執行回測。")

# Benchmarks remain in the price database but are not eligible to compete
# against themselves or to duplicate the defensive fallback in rankings.
metadata = {
    ticker: info
    for ticker, info in metadata.items()
    if ticker not in {benchmark_ticker, defensive_ticker}
}

if not metadata:
    st.error("請至少選擇一個分類或輸入一個有效 ticker。")
    st.stop()
if end_date <= start_date:
    st.error("回測結束日期必須晚於開始日期。")
    st.stop()
if sum(lookback_weights.values()) == 0:
    st.error("至少一個動能週期的權重必須大於零。")
    st.stop()

asset_tickers = list(metadata)
download_tickers = tuple(
    dict.fromkeys([*asset_tickers, benchmark_ticker, defensive_ticker])
)

try:
    with st.spinner(
        f"正在下載 {len(download_tickers)} 個標的並執行{frequency_settings['label']}研究…"
    ):
        if data_mode == "Live Yahoo Finance":
            if market == "台股":
                prices = load_taiwan_price_database(
                    download_tickers,
                    start_date,
                    end_date,
                    DATA_PIPELINE_VERSION,
                )
            else:
                prices = load_live_data(
                    download_tickers,
                    start_date,
                    end_date + timedelta(days=1),
                    DATA_PIPELINE_VERSION,
                )
        else:
            prices = load_demo_data(download_tickers).loc[
                pd.Timestamp(start_date) : pd.Timestamp(end_date)
            ]

        available_assets = [ticker for ticker in asset_tickers if ticker in prices]
        if not available_assets:
            raise RuntimeError(
                "Selected assets have no usable data. Choose Live Yahoo Finance or another universe."
            )

        config = BacktestConfig(
            lookback_weights=lookback_weights,
            frequency=frequency,
            top_n=min(top_n, len(available_assets)),
            weighting=weighting,
            rank_weights=rank_weights,
            require_positive_momentum=positive_filter,
            risk_adjusted_score=risk_adjusted,
            defensive_asset=defensive_ticker if use_defensive else None,
            transaction_cost_bps=cost_bps,
            volatility_window=frequency_settings["volatility_window"],
        )
        result = run_backtest(prices, available_assets, config)
except Exception as exc:
    st.error(f"Research run failed: {exc}")
    st.info("請確認日期、ticker 與網路；若只是測試介面，可切換 Offline demo。")
    st.stop()

if result.net_returns.empty or result.scores.empty:
    st.warning("資料歷史不足以計算最長動能週期，請把開始日期往前調。")
    st.stop()

periods_per_year = PERIODS_PER_YEAR[frequency]
benchmark = benchmark_returns(
    result.sampled_prices,
    benchmark_ticker,
    result.net_returns.index[0],
)
price_benchmark = pd.Series(dtype=float)
price_benchmark_label: str | None = None
if market == "台股" and data_mode == "Live Yahoo Finance":
    try:
        raw_benchmark_prices = load_split_adjusted_close(
            benchmark_ticker,
            start_date,
            end_date + timedelta(days=1),
            DATA_PIPELINE_VERSION,
        )
        sampled_raw_benchmark = raw_benchmark_prices.reindex(
            result.sampled_prices.index
        ).ffill()
        price_benchmark = benchmark_returns(
            sampled_raw_benchmark,
            benchmark_ticker,
            result.net_returns.index[0],
        )
        price_benchmark_label = "0050 · 價格報酬（不含息）"
    except Exception:
        # The total-return comparison remains available if the supplementary
        # raw-close request is temporarily unavailable.
        price_benchmark = pd.Series(dtype=float)

strategy_metrics = performance_summary(result.net_returns, periods_per_year)
benchmark_metrics = performance_summary(benchmark, periods_per_year)
price_benchmark_metrics = (
    performance_summary(price_benchmark, periods_per_year)
    if not price_benchmark.empty
    else None
)
benchmark_cumulative_return = (
    float(equity_curve(benchmark).iloc[-1] - 1)
    if not benchmark.empty
    else float("nan")
)
benchmark_start_label = (
    f"{benchmark.index.min():%Y-%m-%d}" if not benchmark.empty else "—"
)
benchmark_max_daily_move = (
    float(prices[benchmark_ticker].pct_change(fill_method=None).abs().max())
    if benchmark_ticker in prices
    else float("nan")
)
latest_signal_date = result.scores.index[-1]

components = compute_momentum_components(
    result.sampled_prices,
    available_assets,
    list(lookback_weights),
)
latest_scores = result.scores.iloc[-1].dropna().sort_values(ascending=False)
latest_weights = result.target_weights.iloc[-1]
custom_rank_weight_line = (
    "- 自訂排名權重：**"
    + " / ".join(
        f"Rank {rank} {weight:.1f}%"
        for rank, weight in enumerate(rank_weights or (), start=1)
    )
    + "**"
    if rank_weights is not None
    else ""
)

ranking_rows = []
for ticker, score in latest_scores.items():
    info = metadata[ticker]
    row = {
        "Ticker": ticker,
        "Name": info.name,
        "Group": info.group,
        "Momentum score": score,
        "Target weight": latest_weights.get(ticker, 0.0),
    }
    for periods in lookback_weights:
        row[f"{periods} {frequency_settings['unit']} return"] = components[periods].loc[
            latest_signal_date, ticker
        ]
    ranking_rows.append(row)
ranking = pd.DataFrame(ranking_rows)
ranking.index = range(1, len(ranking) + 1)

if data_mode == "Offline demo":
    st.warning("Offline demo 使用合成資料，不是歷史投資績效。")
elif market == "台股" and universe_name == "台股主題股票籃子":
    st.warning(
        "台股主題籃子以目前選定公司回填歷史，存在存活偏誤與事後選樣偏誤；"
        "策略績效只能視為研究上限，不能視為可直接複製的實盤報酬。"
    )

coverage_columns = st.columns(5)
coverage_columns[0].metric("市場資料庫", market)
coverage_columns[1].metric("研究標的", len(available_assets))
coverage_columns[2].metric("分類數量", len({metadata[ticker].group for ticker in available_assets}))
coverage_columns[3].metric("頻率", frequency_settings["label"])
coverage_columns[4].metric("資料截止", f"{result.sampled_prices.index[-1]:%Y-%m-%d}")

institutional_flows = pd.DataFrame()
front_flow_securities = pd.DataFrame()
front_flow_groups = pd.DataFrame()
if market == "台股" and data_mode == "Live Yahoo Finance":
    institutional_flows = load_taiwan_institutional_flows(DATA_PIPELINE_VERSION)
    if not institutional_flows.empty:
        front_flow_securities, front_flow_groups = calculate_fund_flow_signals(
            prices,
            institutional_flows,
            taiwan_master,
        )

st.subheader("資金流主題總覽")
if market != "台股":
    st.info("切換至台股即可查看三大法人資金流、產業流向與資金帶領股票。")
elif data_mode != "Live Yahoo Finance":
    st.info("資金流總覽需要真實交易所資料，請切換至 Live Yahoo Finance。")
elif front_flow_groups.empty:
    st.warning("法人資金流資料尚未就緒，請先執行每日更新。")
else:
    front_focus = front_flow_groups[
        front_flow_groups["Stage"].isin(
            [
                "資金累積＋價格確認",
                "資金累積、價格未確認",
                "早期轉入",
            ]
        )
    ].head(4)
    if front_focus.empty:
        front_focus = front_flow_groups.head(4)
    front_cards = st.columns(max(1, len(front_focus)))
    for position, (_, row) in enumerate(front_focus.iterrows()):
        front_cards[position].metric(
            row["Industry"],
            f"{row['Flow score']:.0f} / 100",
            f"5日 {row['5D net value'] / 1e8:+.1f}億",
        )
    st.dataframe(
        front_focus[
            [
                "Industry",
                "Dominant investor",
                "Positive flow breadth",
                "Leading stocks",
                "Flow reason",
                "Research action",
            ]
        ].style.format({"Positive flow breadth": "{:.0%}"}),
        width="stretch",
        hide_index=True,
    )

st.subheader("最新模型訊號")
st.caption(
    f"訊號日期：{latest_signal_date:%Y-%m-%d}。排名會延後一個"
    f"{frequency_settings['unit']}才計入歷史報酬，避免前視偏誤。"
)

allocation = latest_weights[latest_weights > 0].sort_values(ascending=False)
if allocation.empty:
    st.info("目前模型持有現金。")
else:
    card_count = min(6, len(allocation))
    cards = st.columns(card_count)
    for number, (ticker, weight) in enumerate(allocation.items()):
        info = metadata.get(ticker)
        label = info.name if info else "Defensive asset"
        cards[number % card_count].metric(ticker, f"{weight:.1%}", label)

(
    tab_flow,
    tab_overview,
    tab_attention,
    tab_rankings,
    tab_groups,
    tab_portfolio,
    tab_method,
) = st.tabs(
    [
        "資金流雷達",
        "績效",
        "關注與主要玩家",
        "詳細排名",
        "產業輪動 RRG",
        "配置歷史",
        "方法與限制",
    ]
)

with tab_overview:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Strategy CAGR", format_percent(strategy_metrics["CAGR"]))
    metric_columns[1].metric("Strategy Sharpe", format_number(strategy_metrics["Sharpe"]))
    metric_columns[2].metric("Max drawdown", format_percent(strategy_metrics["Max drawdown"]))
    metric_columns[3].metric("Annual volatility", format_percent(strategy_metrics["Volatility"]))
    metric_columns[4].metric(
        "Annual turnover",
        f"{result.turnover.mean() * periods_per_year:.1f}x",
    )

    st.plotly_chart(
        performance_chart(
            result.net_returns,
            benchmark,
            benchmark_label,
            price_benchmark,
            price_benchmark_label,
        ),
        width="stretch",
    )
    if market == "台股":
        st.caption(
            "0050 灰線是股息再投入的含息總報酬；橘線是已校正 2025 年 4：1 "
            "分割、但不含配息的價格報酬。分割本身不會產生投資報酬。"
        )
        st.caption(
            f"資料管線 {DATA_PIPELINE_VERSION}｜基準計算起點 "
            f"{benchmark_start_label}｜0050 累積含息報酬 "
            f"{benchmark_cumulative_return:.1%}｜校正後最大單日變動 "
            f"{benchmark_max_daily_move:.1%}"
        )
    left, right = st.columns([2, 1])
    with left:
        drawdown_series = [
            drawdown(result.net_returns).rename("Strategy"),
            drawdown(benchmark).rename(benchmark_label),
        ]
        if not price_benchmark.empty and price_benchmark_label is not None:
            drawdown_series.append(
                drawdown(price_benchmark).rename(price_benchmark_label)
            )
        drawdowns = pd.concat(drawdown_series, axis=1)
        drawdown_figure = px.area(
            drawdowns,
            labels={"value": "Drawdown", "index": "", "variable": ""},
            color_discrete_sequence=["#fb7185", "#94a3b8", "#f59e0b"],
        )
        drawdown_figure.update_yaxes(tickformat=".0%")
        drawdown_figure.update_layout(hovermode="x unified", height=350)
        st.plotly_chart(drawdown_figure, width="stretch")
    with right:
        comparison_data = {
            "Strategy": strategy_metrics,
            benchmark_label: benchmark_metrics,
        }
        if price_benchmark_metrics is not None and price_benchmark_label is not None:
            comparison_data[price_benchmark_label] = price_benchmark_metrics
        comparison = pd.DataFrame(comparison_data).astype(object)
        for row in ["CAGR", "Volatility", "Max drawdown", "Win rate"]:
            comparison.loc[row] = comparison.loc[row].map(format_percent)
        comparison.loc["Sharpe"] = comparison.loc["Sharpe"].map(format_number)
        st.dataframe(comparison, width="stretch")

with tab_attention:
    st.subheader("目前優先關注方向")
    st.caption(
        "先用分類平均動能找資金主線，再由領先 ETF 的主要持股找出值得深入研究的公司。"
    )

    attention_groups = (
        ranking.groupby("Group", as_index=False)
        .agg(
            Average_score=("Momentum score", "mean"),
            Best_asset_score=("Momentum score", "max"),
            Positive_assets=("Momentum score", lambda values: int((values > 0).sum())),
            Asset_count=("Ticker", "count"),
        )
        .sort_values("Average_score", ascending=False)
    )
    group_cards = st.columns(min(3, len(attention_groups)))
    for position, row in attention_groups.head(3).reset_index(drop=True).iterrows():
        group_cards[position].metric(
            f"#{position + 1} {row['Group']}",
            f"{row['Average_score']:.3f}",
            f"{int(row['Positive_assets'])}/{int(row['Asset_count'])} positive",
        )

    st.markdown("#### 領先 ETF／股票")
    attention_assets = ranking.head(15).copy()
    st.dataframe(
        attention_assets.style.format(
            {
                "Momentum score": "{:.3f}",
                "Target weight": "{:.1%}",
                **{
                    column: "{:.1%}"
                    for column in attention_assets
                    if column.endswith(f"{frequency_settings['unit']} return")
                },
            }
        ),
        width="stretch",
        height=420,
    )

    if data_mode != "Live Yahoo Finance":
        st.info("切換到 Live Yahoo Finance 才能讀取 ETF 最新主要持股。")
    elif not is_etf_universe:
        st.info(
            "目前選擇的是股票籃子；上表本身就是該主題的股票關注清單。"
            "切換到 ETF 研究層級可繼續拆解 ETF 主要持股。"
        )
    else:
        candidate_etfs = list(latest_scores.head(10).index)
        selected_etfs = st.multiselect(
            "拆解哪些領先 ETF",
            candidate_etfs,
            default=candidate_etfs[: min(3, len(candidate_etfs))],
            help="持股資料快取 24 小時；基金持股可能隨時變動。",
        )
        if selected_etfs:
            with st.spinner("正在讀取 ETF 主要持股並計算股票領先度…"):
                holdings = load_top_holdings(tuple(selected_etfs))
                if holdings.empty:
                    leadership = pd.DataFrame()
                else:
                    holding_tickers = tuple(holdings["Holding ticker"].unique())
                    holding_prices = load_live_data(
                        holding_tickers,
                        date.today() - timedelta(days=400),
                        date.today() + timedelta(days=1),
                        DATA_PIPELINE_VERSION,
                    )
                    leadership = analyze_holding_leadership(holdings, holding_prices)

            if leadership.empty:
                st.warning("Yahoo Finance 暫時沒有回傳所選 ETF 的可用持股資料。")
            else:
                leadership["ETF momentum score"] = leadership["ETF"].map(
                    latest_scores.to_dict()
                )
                leadership["ETF group"] = leadership["ETF"].map(
                    {ticker: metadata[ticker].group for ticker in selected_etfs}
                )
                leadership = leadership.sort_values(
                    ["ETF momentum score", "ETF", "ETF rank"],
                    ascending=[False, True, True],
                )

                st.markdown("#### ETF 中的主要帶領玩家")
                leader_display = leadership[
                    [
                        "ETF",
                        "ETF group",
                        "Holding ticker",
                        "Holding name",
                        "Holding weight",
                        "21d return",
                        "63d return",
                        "126d return",
                        "Stock momentum",
                        "Leadership score",
                        "ETF rank",
                    ]
                ]
                st.dataframe(
                    leader_display.style.format(
                        {
                            "Holding weight": "{:.1%}",
                            "21d return": "{:.1%}",
                            "63d return": "{:.1%}",
                            "126d return": "{:.1%}",
                            "Stock momentum": "{:.1%}",
                            "Leadership score": "{:.4f}",
                        }
                    ),
                    width="stretch",
                    height=560,
                )

                chart_data = leadership.copy()
                chart_data["Player"] = (
                    chart_data["Holding ticker"] + " · " + chart_data["Holding name"]
                )
                leader_chart = px.bar(
                    chart_data,
                    x="Leadership score",
                    y="Player",
                    color="ETF",
                    facet_col="ETF",
                    facet_col_wrap=min(3, len(selected_etfs)),
                    orientation="h",
                    title="持股權重 × 股票多週期動能（研究優先度代理值）",
                )
                leader_chart.update_layout(
                    height=max(480, int(len(chart_data) / max(1, len(selected_etfs))) * 35)
                )
                st.plotly_chart(leader_chart, width="stretch")

                st.download_button(
                    "下載 ETF 主要玩家 CSV",
                    data=leader_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"etf-leading-players-{latest_signal_date:%Y-%m-%d}.csv",
                    mime="text/csv",
                )

with tab_rankings:
    return_columns = [
        column for column in ranking if column.endswith(f"{frequency_settings['unit']} return")
    ]
    formatters = {
        "Momentum score": "{:.3f}",
        "Target weight": "{:.1%}",
        **{column: "{:.1%}" for column in return_columns},
    }
    st.dataframe(ranking.style.format(formatters), width="stretch", height=520)
    st.download_button(
        "下載最新排名 CSV",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"rotation-ranking-{latest_signal_date:%Y-%m-%d}.csv",
        mime="text/csv",
    )

    history = result.scores.rename(
        columns={ticker: f"{ticker} · {metadata[ticker].name}" for ticker in available_assets}
    )
    history_periods = {"Daily": 126, "Weekly": 52, "Monthly": 36}[frequency]
    st.plotly_chart(
        px.line(
            history.tail(history_periods),
            labels={"value": "Momentum score", "index": "", "variable": ""},
            title=f"近期動能領先變化（{history_periods} 個{frequency_settings['unit']}）",
        ).update_layout(hovermode="x unified", height=470),
        width="stretch",
    )

with tab_groups:
    unique_groups = {info.group for info in metadata.values()}
    rrg_metadata = (
        {
            ticker: AssetInfo(
                ticker=ticker,
                name=info.name,
                group=f"{ticker} · {info.name}",
            )
            for ticker, info in metadata.items()
        }
        if len(unique_groups) < 2 and len(metadata) > 1
        else metadata
    )
    rrg_defaults = {
        "Daily": {"long": 63, "momentum": 21, "tail": 10},
        "Weekly": {"long": 26, "momentum": 4, "tail": 8},
        "Monthly": {"long": 12, "momentum": 3, "tail": 6},
    }[frequency]
    rrg_controls = st.columns(3)
    rrg_long_window = int(
        rrg_controls[0].number_input(
            f"RS-Ratio 正規化週期（{frequency_settings['unit']}）",
            min_value=4,
            max_value=max(4, len(result.sampled_prices) - 2),
            value=min(
                rrg_defaults["long"],
                max(4, len(result.sampled_prices) - 2),
            ),
            step=1,
            key=f"rrg_long_{frequency}",
        )
    )
    rrg_momentum_window = int(
        rrg_controls[1].number_input(
            f"RS-Momentum 週期（{frequency_settings['unit']}）",
            min_value=1,
            max_value=max(1, min(63, len(result.sampled_prices) // 3)),
            value=min(
                rrg_defaults["momentum"],
                max(1, min(63, len(result.sampled_prices) // 3)),
            ),
            step=1,
            key=f"rrg_momentum_{frequency}",
        )
    )
    rrg_tail_length = int(
        rrg_controls[2].number_input(
            f"軌跡長度（{frequency_settings['unit']}）",
            min_value=2,
            max_value=max(2, min(30, len(result.sampled_prices))),
            value=min(
                rrg_defaults["tail"],
                max(2, min(30, len(result.sampled_prices))),
            ),
            step=1,
            key=f"rrg_tail_{frequency}",
        )
    )
    rs_ratio, rs_momentum, group_indices = calculate_group_rrg(
        result.sampled_prices,
        rrg_metadata,
        benchmark_ticker,
        rrg_long_window,
        rrg_momentum_window,
    )
    rotation_summary = build_rotation_summary(
        result.sampled_prices,
        rrg_metadata,
        benchmark_ticker,
        rs_ratio,
        rs_momentum,
        group_indices,
        short_window=rrg_momentum_window,
        long_window=rrg_long_window,
    )

    st.markdown("### 哪些產業正在轉強？")
    st.caption(
        "優先觀察藍色「轉強 Improving」：相對強度仍在 100 以下，但 "
        "RS-Momentum 已翻正；進入綠色「領先 Leading」代表相對強度與動能都占優。"
    )
    if rotation_summary.empty:
        st.warning("目前歷史資料不足以計算 RRG，請縮短週期或把回測開始日提前。")
    else:
        improving = rotation_summary[rotation_summary["Quadrant"] == "Improving"]
        focus = (
            improving
            if not improving.empty
            else rotation_summary[rotation_summary["Quadrant"] == "Leading"]
        ).head(3)
        focus_columns = st.columns(max(1, len(focus)))
        for position, (_, row) in enumerate(focus.iterrows()):
            focus_columns[position].metric(
                f"{row['狀態']}｜{row['Group']}",
                f"RS-Momentum {row['RS-Momentum']:+.2f}%",
                f"短期超額 {row['短期超額報酬']:+.1%}｜廣度 {row['相對廣度']:.0%}",
            )

        st.plotly_chart(
            rrg_chart(rs_ratio, rs_momentum, rrg_tail_length),
            width="stretch",
        )
        st.caption(
            "此圖採公開、可解釋的 RRG-style 算法：產業等權指數相對基準後，"
            "把滾動相對強度正規化至 100；不是 proprietary JdK 官方數值。"
        )

        st.markdown("#### 輪動判斷與原因")
        rotation_display = rotation_summary[
            [
                "Group",
                "狀態",
                "RS-Ratio",
                "RS-Momentum",
                "短期超額報酬",
                "中期超額報酬",
                "相對廣度",
                "主要帶動股票",
                "轉強／轉弱原因",
            ]
        ]
        st.dataframe(
            rotation_display.style.format(
                {
                    "RS-Ratio": "{:.2f}",
                    "RS-Momentum": "{:+.2f}%",
                    "短期超額報酬": "{:+.1%}",
                    "中期超額報酬": "{:+.1%}",
                    "相對廣度": "{:.0%}",
                }
            ),
            width="stretch",
            height=min(620, 120 + len(rotation_display) * 48),
        )
        st.download_button(
            "下載產業輪動與原因 CSV",
            data=rotation_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"industry-rotation-reasons-{latest_signal_date:%Y-%m-%d}.csv",
            mime="text/csv",
        )

    st.markdown("### 動能與多週期報酬補充")
    comparison_ranking = ranking.copy()
    comparison_ranking["Group"] = comparison_ranking["Ticker"].map(
        {ticker: info.group for ticker, info in rrg_metadata.items()}
    )
    group_scores = (
        comparison_ranking.groupby("Group", as_index=False)
        .agg(
            Average_score=("Momentum score", "mean"),
            Best_score=("Momentum score", "max"),
            Positive_assets=("Momentum score", lambda values: int((values > 0).sum())),
            Asset_count=("Ticker", "count"),
        )
        .sort_values("Average_score", ascending=True)
    )
    group_figure = px.bar(
        group_scores,
        x="Average_score",
        y="Group",
        orientation="h",
        color="Average_score",
        color_continuous_scale="RdYlGn",
        title="分類平均動能",
    )
    group_figure.update_layout(height=max(360, len(group_scores) * 48))
    st.plotly_chart(group_figure, width="stretch")

    heatmap_data = comparison_ranking.set_index("Ticker")[return_columns]
    heatmap_data.columns = [column.replace(" return", "") for column in heatmap_data.columns]
    heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".1%",
        title="每個標的的多週期報酬",
    )
    heatmap.update_layout(height=max(420, len(heatmap_data) * 27))
    st.plotly_chart(heatmap, width="stretch")

with tab_flow:
    st.markdown("### 資金流雷達：錢正在往哪裡走？")
    st.caption(
        "這一頁使用交易所三大法人買賣超，不再把價格上漲直接當成資金流入。"
        "淨買超股數乘以當日收盤價是「估算淨流入金額」，適合比較方向與強度，"
        "不等於逐筆成交現金流。"
    )
    if market != "台股":
        st.info("目前官方法人資金流模組先支援台股；請在左側切換至「台股」。")
    elif data_mode != "Live Yahoo Finance":
        st.info("資金流必須使用真實交易所資料，請把資料來源切換為 Live Yahoo Finance。")
    else:
        institutional_flows = load_taiwan_institutional_flows(DATA_PIPELINE_VERSION)
        if institutional_flows.empty:
            st.warning(
                "尚未建立法人資金流資料庫。請執行 "
                "`.venv/bin/python scripts/daily_update.py` 後重新整理。"
            )
        else:
            st.markdown("#### 自訂綜合資金分數")
            flow_controls = st.columns(4)
            flow_weights = {
                "20D flow intensity": flow_controls[0].number_input(
                    "20日法人流入權重",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=5.0,
                ),
                "5D flow intensity": flow_controls[1].number_input(
                    "5日資金加速權重",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    step=5.0,
                ),
                "Trust 20D intensity": flow_controls[2].number_input(
                    "20日投信流入權重",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=5.0,
                ),
                "20D return": flow_controls[3].number_input(
                    "20日價格確認權重",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=5.0,
                ),
            }
            flow_securities, flow_groups = calculate_fund_flow_signals(
                prices,
                institutional_flows,
                taiwan_master,
                weights=flow_weights,
            )
            if flow_securities.empty or flow_groups.empty:
                st.warning("目前選取標的與法人資料沒有足夠重疊，請擴大研究範圍。")
            else:
                flow_date = pd.Timestamp(flow_securities["Signal date"].max())
                st.caption(
                    f"法人資料截止：{flow_date:%Y-%m-%d}｜"
                    "綜合分數採橫斷面百分位，100 代表目前相對最強。"
                )
                actionable = flow_groups[
                    flow_groups["Stage"].isin(
                        [
                            "早期轉入",
                            "資金累積＋價格確認",
                            "資金累積、價格未確認",
                        ]
                    )
                ].head(4)
                if actionable.empty:
                    actionable = flow_groups.head(4)
                flow_cards = st.columns(max(1, len(actionable)))
                for position, (_, row) in enumerate(actionable.iterrows()):
                    flow_cards[position].metric(
                        row["Industry"],
                        f"{row['Flow score']:.0f} / 100",
                        f"5日 {row['5D net value'] / 1e8:+.1f}億",
                    )

                daily_group_flows = calculate_daily_group_flows(
                    prices,
                    institutional_flows,
                    taiwan_master,
                )

                st.markdown("#### 法人資金流向圖")
                top_flow_groups = flow_groups.nlargest(8, "20D net value")
                investor_links = [
                    ("外資", "Foreign 20D value"),
                    ("投信", "Trust 20D value"),
                    ("自營商", "Dealer 20D value"),
                ]
                sankey_nodes = [name for name, _ in investor_links] + list(
                    top_flow_groups["Industry"]
                )
                sankey_sources: list[int] = []
                sankey_targets: list[int] = []
                sankey_values: list[float] = []
                sankey_labels: list[str] = []
                for investor_index, (investor, column) in enumerate(investor_links):
                    for group_offset, (_, group_row) in enumerate(
                        top_flow_groups.iterrows()
                    ):
                        value = max(0.0, float(group_row[column]) / 1e8)
                        if value == 0:
                            continue
                        sankey_sources.append(investor_index)
                        sankey_targets.append(len(investor_links) + group_offset)
                        sankey_values.append(value)
                        sankey_labels.append(
                            f"{investor} → {group_row['Industry']}：{value:.1f}億"
                        )
                sankey = go.Figure(
                    go.Sankey(
                        node={"label": sankey_nodes, "pad": 18, "thickness": 18},
                        link={
                            "source": sankey_sources,
                            "target": sankey_targets,
                            "value": sankey_values,
                            "label": sankey_labels,
                        },
                    )
                )
                sankey.update_layout(
                    title="外資／投信／自營商 → 20日淨流入最高產業",
                    height=520,
                )
                st.plotly_chart(sankey, width="stretch")

                st.markdown("#### 產業資金流四象限")
                flow_scatter = px.scatter(
                    flow_groups,
                    x="20D flow intensity",
                    y="5D flow intensity",
                    size="Constituents",
                    color="Stage",
                    hover_name="Industry",
                    hover_data={
                        "Flow score": ":.1f",
                        "20D return": ":.1%",
                        "Positive flow breadth": ":.0%",
                        "Leading stocks": True,
                    },
                    title="右上：中期流入且短期加速｜左下：資金持續撤出",
                )
                flow_scatter.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
                flow_scatter.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                flow_scatter.update_xaxes(title="20日法人淨流入／市值代理")
                flow_scatter.update_yaxes(title="5日法人淨流入／市值代理")
                flow_scatter.update_layout(height=610)
                st.plotly_chart(flow_scatter, width="stretch")

                st.markdown("#### 產業五維資金雷達")
                radar_options = list(flow_groups["Industry"])
                radar_groups = st.multiselect(
                    "比較哪些產業",
                    radar_options,
                    default=radar_options[: min(5, len(radar_options))],
                    key="fund_flow_radar_groups",
                )
                radar_metrics = {
                    "20日流入": flow_groups["20D flow intensity"].rank(pct=True) * 100,
                    "5日加速": flow_groups["Flow acceleration"].rank(pct=True) * 100,
                    "投信認同": flow_groups["Trust 20D intensity"].rank(pct=True) * 100,
                    "流入廣度": flow_groups["Positive flow breadth"] * 100,
                    "價格確認": flow_groups["20D return"].rank(pct=True) * 100,
                }
                radar_frame = pd.DataFrame(radar_metrics, index=flow_groups.index)
                radar_figure = go.Figure()
                radar_categories = list(radar_metrics)
                for industry in radar_groups:
                    row_index = flow_groups.index[flow_groups["Industry"] == industry]
                    if row_index.empty:
                        continue
                    values = [
                        float(radar_frame.loc[row_index[0], category])
                        for category in radar_categories
                    ]
                    radar_figure.add_trace(
                        go.Scatterpolar(
                            r=values + values[:1],
                            theta=radar_categories + radar_categories[:1],
                            fill="toself",
                            name=industry,
                            opacity=0.55,
                        )
                    )
                radar_figure.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 100]}},
                    title="同一尺度比較：100 代表目前橫斷面最強",
                    height=610,
                )
                st.plotly_chart(radar_figure, width="stretch")

                if not daily_group_flows.empty:
                    st.markdown("#### 每日資金流軌跡")
                    trend_defaults = list(top_flow_groups["Industry"].head(5))
                    trend_groups = st.multiselect(
                        "追蹤產業",
                        sorted(daily_group_flows["Industry"].unique()),
                        default=trend_defaults,
                        key="fund_flow_trend_groups",
                    )
                    trend_data = daily_group_flows[
                        daily_group_flows["Industry"].isin(trend_groups)
                    ].copy()
                    trend_data["每日法人淨流入（億）"] = (
                        trend_data["Total net value"] / 1e8
                    )
                    st.plotly_chart(
                        px.line(
                            trend_data,
                            x="Date",
                            y="每日法人淨流入（億）",
                            color="Industry",
                            markers=True,
                            title="單日流向：辨識持續流入、突然加速或反轉",
                        ).update_layout(height=500, hovermode="x unified"),
                        width="stretch",
                    )

                st.markdown("#### 法人結構熱圖")
                heat_groups = flow_groups.head(18).set_index("Industry")[
                    [
                        "Foreign 20D value",
                        "Trust 20D value",
                        "Dealer 20D value",
                    ]
                ] / 1e8
                heat_groups.columns = ["外資", "投信", "自營商"]
                investor_heatmap = px.imshow(
                    heat_groups,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    text_auto=".1f",
                    labels={"color": "20日淨流入（億）"},
                    title="誰在買、誰在賣：前18名產業法人分項",
                )
                investor_heatmap.update_layout(height=620)
                st.plotly_chart(investor_heatmap, width="stretch")

                group_display = flow_groups[
                    [
                        "Industry",
                        "Stage",
                        "Flow score",
                        "5D net value",
                        "20D net value",
                        "Foreign 20D value",
                        "Trust 20D value",
                        "Dealer 20D value",
                        "Positive flow breadth",
                        "Flow acceleration",
                        "Top 3 concentration",
                        "20D return",
                        "Leading stocks",
                        "Dominant investor",
                        "Flow reason",
                        "Research action",
                    ]
                ].copy()
                for column in [
                    "5D net value",
                    "20D net value",
                    "Foreign 20D value",
                    "Trust 20D value",
                    "Dealer 20D value",
                    "Flow acceleration",
                ]:
                    group_display[column] /= 1e8
                st.dataframe(
                    group_display.style.format(
                        {
                            "Flow score": "{:.1f}",
                            "5D net value": "{:+.1f} 億",
                            "20D net value": "{:+.1f} 億",
                            "Foreign 20D value": "{:+.1f} 億",
                            "Trust 20D value": "{:+.1f} 億",
                            "Dealer 20D value": "{:+.1f} 億",
                            "Positive flow breadth": "{:.0%}",
                            "Flow acceleration": "{:+.1f} 億",
                            "Top 3 concentration": "{:.0%}",
                            "20D return": "{:+.1%}",
                        }
                    ),
                    width="stretch",
                    height=520,
                )

                st.markdown("#### 資金流動原因與事件驗證")
                reason_industry = st.selectbox(
                    "深入查看哪個產業",
                    list(flow_groups["Industry"]),
                    key="fund_flow_reason_industry",
                )
                reason_row = flow_groups[
                    flow_groups["Industry"] == reason_industry
                ].iloc[0]
                st.info(reason_row["Flow reason"])
                st.caption(f"研究動作：{reason_row['Research action']}")
                reason_tickers = tuple(
                    ticker.strip()
                    for ticker in str(reason_row["Leading stocks"]).split("、")
                    if ticker.strip()
                )
                if st.button(
                    "查找主要帶動股票的最新新聞",
                    key="load_flow_leader_news",
                ):
                    with st.spinner("正在取得主要帶動股票的最新消息…"):
                        news = load_ticker_news(reason_tickers)
                    if news.empty:
                        st.warning("目前沒有取得可用新聞，資金原因仍以法人結構與量價資料為準。")
                    else:
                        for _, news_row in news.head(12).iterrows():
                            title = news_row["Title"] or "未命名消息"
                            url = news_row["URL"]
                            publisher = news_row["Publisher"]
                            if url:
                                st.markdown(f"- [{title}]({url}) — {publisher}")
                            else:
                                st.markdown(f"- {title} — {publisher}")
                        st.caption(
                            "新聞只用來驗證可能催化劑；法人流入本身不能證明某則新聞是因果。"
                        )

                st.markdown("#### 可以怎麼操作")
                st.markdown(
                    """
                    - **早期轉入**：5 日流入轉正、20 日仍未轉正。列入觀察名單，
                      等價格站上中期趨勢或資金廣度超過 50% 再進場。
                    - **資金累積＋價格確認**：5 日、20 日流入與 20 日報酬同時為正。
                      可由產業內資金分數最高的股票分批建立部位。
                    - **資金累積、價格未確認**：法人已連續流入但價格仍弱。
                      保留觀察，不急著進場；等待價格轉正與流入廣度擴大。
                    - **漲勢仍在、資金減速**：價格仍強但 5 日資金轉負。
                      不追價，既有部位可提高停利或降低權重。
                    - **資金撤出**：5 日與 20 日皆為負。避免逆勢加碼，
                      除非有獨立基本面事件與明確風險界線。
                    """
                )
                stock_display = flow_securities[
                    [
                        "Ticker",
                        "Name",
                        "Industry",
                        "Asset type",
                        "Stage",
                        "Flow score",
                        "5D net value",
                        "20D net value",
                        "Foreign 20D value",
                        "Trust 20D value",
                        "Dealer 20D value",
                        "20D return",
                    ]
                ].head(100).copy()
                for column in [
                    "5D net value",
                    "20D net value",
                    "Foreign 20D value",
                    "Trust 20D value",
                    "Dealer 20D value",
                ]:
                    stock_display[column] /= 1e8
                st.markdown("#### 個股資金流排名（前 100）")
                st.dataframe(
                    stock_display.style.format(
                        {
                            "Flow score": "{:.1f}",
                            "5D net value": "{:+.2f} 億",
                            "20D net value": "{:+.2f} 億",
                            "Foreign 20D value": "{:+.2f} 億",
                            "Trust 20D value": "{:+.2f} 億",
                            "Dealer 20D value": "{:+.2f} 億",
                            "20D return": "{:+.1%}",
                        }
                    ),
                    width="stretch",
                    height=600,
                )
                st.caption(
                    "ETF 顯示的是法人在次級市場的買賣超代理，並非 ETF "
                    "申購／贖回造成的基金淨流量；兩者不可混為一談。"
                )
                st.download_button(
                    "下載產業資金流 CSV",
                    data=flow_groups.to_csv(index=False).encode("utf-8"),
                    file_name=f"taiwan-industry-fund-flow-{flow_date:%Y-%m-%d}.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "下載個股資金流 CSV",
                    data=flow_securities.to_csv(index=False).encode("utf-8"),
                    file_name=f"taiwan-stock-fund-flow-{flow_date:%Y-%m-%d}.csv",
                    mime="text/csv",
                )

with tab_portfolio:
    st.plotly_chart(
        allocation_chart(
            result.deployed_weights,
            metadata,
            defensive_ticker,
            defensive_label,
        ),
        width="stretch",
    )
    recent_rows = {"Daily": 30, "Weekly": 26, "Monthly": 12}[frequency]
    recent = result.deployed_weights.tail(recent_rows).rename(
        columns={
            **{ticker: f"{ticker} · {metadata[ticker].name}" for ticker in available_assets},
            defensive_ticker: defensive_label,
        }
    )
    recent = recent.loc[:, (recent.abs().sum() > 0)]
    st.dataframe(recent.style.format("{:.1%}"), width="stretch")
    st.download_button(
        "下載配置歷史 CSV",
        data=result.deployed_weights.to_csv().encode("utf-8"),
        file_name=f"rotation-weights-{frequency.lower()}.csv",
        mime="text/csv",
    )

with tab_method:
    st.markdown(
        f"""
        ### 本次研究設定

        - 市場資料庫：**{market}**
        - 研究層級：**{universe_name}**
        - 分類：**{", ".join(selected_groups)}**
        - 訊號與再平衡：**{frequency_settings["label"]}**
        - 標的數量：**{len(available_assets)}**
        - 持有數量：**Top {min(top_n, len(available_assets))}**
        - 配置方法：**{weighting}**
        {custom_rank_weight_line}
        - 風險調整：**{"開啟" if risk_adjusted else "關閉"}**
        - 正動能過濾：**{"開啟" if positive_filter else "關閉"}**
        - 防禦資產：**{defensive_ticker if use_defensive else "現金"}**
        - 換手成本：**{cost_bps:.0f} bps**

        ### 計算流程

        1. 下載調整後日價格；台股若仍有超過 40% 的企業行動斷層，先將事件前價格重新銜接。
        2. 將日價格轉換成指定的日線、週線或月線資料。
        3. 計算 {", ".join(f"{periods} {frequency_settings['unit']}" for periods in lookback_weights)} 報酬。
        4. 正規化權重並組合成一個 momentum score。
        5. 可選擇把分數除以同期年化波動率。
        6. 通過正動能篩選後，選擇排名前 N 個標的。
        7. 依照指定方式配置，並把訊號延後一期才計算報酬。
        8. 根據每期換手率扣除交易成本。

        ### 重要限制

        主題股票籃子是研究分類，不是官方指數，也不代表完整產業成分。
        日線輪動對交易成本、滑價和訊號雜訊特別敏感。Yahoo Finance 資料可能延遲、
        修正或缺失。本工具未完整處理稅務、市場衝擊、退市偏誤和實際成交限制。
        回測不保證未來績效，畫面內容不是個人化投資建議。
        """
    )
