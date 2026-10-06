"""Shared Streamlit rendering for the trading desk console.

Every page imports from here rather than repeating layout code, so the signal
card looks identical wherever it appears and there is one place to change when
the design changes.

Two rules the console follows throughout:

* every number is traceable — the evidence that produced it is one click away;
* counter-evidence is displayed at the same visual weight as supporting
  evidence, because a console that only argues one side stops being trusted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from .agents import default_pipeline
from .agents.sentiment import SentimentAgent
from .contracts import DeskState, Position, Severity, SignalCard, utc_now
from .decision import DecisionConfig
from .desk import DeskConfig, run_desk
from .loaders import (
    MarketData,
    available_industries,
    build_context,
    database_available,
    filter_universe,
    load_taiwan_market,
)
from .news import YahooFinanceNewsProvider
from .risk import RiskLimits
from .store import VETO_REASONS, DeskStore, Feedback

PAGE_ICON = "🛰️"
APP_TITLE = "自主市場情報交易台"


def configure_shell() -> None:
    """Called once by the entry point, before navigation is built."""
    st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")
    inject_styles()


def page_setup(subtitle: str) -> None:
    """Per-page setup. Page config belongs to the entry point, not here."""
    inject_styles()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.46);
            border: 1px solid rgba(148, 163, 184, 0.20);
            padding: 0.85rem;
            border-radius: 0.75rem;
        }
        .desk-badge {
            display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px;
            font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
        }
        .desk-evidence {
            border-left: 3px solid #38BDF8; padding: 0.25rem 0 0.25rem 0.7rem;
            margin-bottom: 0.35rem;
        }
        .desk-counter {
            border-left: 3px solid #F97316; padding: 0.25rem 0 0.25rem 0.7rem;
            margin-bottom: 0.35rem;
        }
        .desk-note {color: #94A3B8; font-size: 0.82rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Data and pipeline caching -------------------------------------------


@st.cache_data(ttl=1800, show_spinner=False)
def cached_market(min_market_cap: float, industries: tuple[str, ...]) -> MarketData:
    """Load and filter the universe. Cached because it reads the whole panel."""
    raw = load_taiwan_market()
    return filter_universe(
        raw,
        industries=industries or None,
        min_market_cap=min_market_cap,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_run(
    min_market_cap: float,
    industries: tuple[str, ...],
    as_of: pd.Timestamp | None,
    signal_budget: int,
    min_conviction: float,
    min_evidence_kinds: int,
    max_weight_per_name: float,
    target_portfolio_vol: float,
    daily_pnl_pct: float,
    drawdown_pct: float,
    enable_news: bool,
) -> DeskState:
    """Run the full pipeline. Arguments are primitives so caching works."""
    data = cached_market(min_market_cap, industries)
    context = build_context(data, as_of=as_of)
    agents = default_pipeline()
    if enable_news:
        agents = [
            SentimentAgent(provider=YahooFinanceNewsProvider())
            if isinstance(agent, SentimentAgent)
            else agent
            for agent in agents
        ]
    config = DeskConfig(
        decision=DecisionConfig(
            min_conviction=min_conviction,
            signal_budget=signal_budget,
            min_evidence_kinds=min_evidence_kinds,
        ),
        limits=RiskLimits(
            max_weight_per_name=max_weight_per_name,
            target_portfolio_vol=target_portfolio_vol,
            min_conviction=min_conviction,
        ),
        agents=agents,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=drawdown_pct,
    )
    return run_desk(context, config)


def get_store() -> DeskStore:
    return DeskStore()


def sidebar_controls() -> dict[str, Any]:
    """Shared sidebar. Identical on every page so settings never drift."""
    state = st.session_state.setdefault(
        "desk_settings",
        {
            "min_market_cap": 5.0,
            "industries": (),
            "as_of": None,
            "signal_budget": 10,
            "min_conviction": 0.55,
            "min_evidence_kinds": 2,
            "max_weight_per_name": 0.05,
            "target_portfolio_vol": 0.12,
            "daily_pnl_pct": 0.0,
            "drawdown_pct": 0.0,
            "enable_news": False,
        },
    )

    with st.sidebar:
        st.markdown(f"### {PAGE_ICON} {APP_TITLE}")
        st.caption("多 Agent 掃描 → 證據交叉驗證 → 風控定量 → 訊號卡")

        if not database_available():
            st.error("找不到本地資料庫，請先執行 scripts/daily_update.py")
            st.stop()

        with st.expander("掃描範圍", expanded=True):
            state["min_market_cap"] = st.slider(
                "最小市值（十億元，以發行股數×收盤價估算）",
                0.0, 50.0, state["min_market_cap"], 1.0,
                help="流動性代理指標。資料庫沒有成交量，市值是可得的最佳近似。",
            )
            data = cached_market(state["min_market_cap"] * 1e9, ())
            industries = available_industries(data)
            state["industries"] = tuple(
                st.multiselect("產業（留空代表全市場）", industries, default=list(state["industries"]))
            )
            sessions = list(reversed(data.prices.index[-40:]))
            labels = ["最新交易日"] + [stamp.strftime("%Y-%m-%d") for stamp in sessions]
            choice = st.selectbox(
                "決策時點（as-of）", labels,
                help="選擇過去某一天可重現當時的訊號，用於回放與驗證。",
            )
            state["as_of"] = None if choice == "最新交易日" else pd.Timestamp(choice)

        with st.expander("訊號門檻"):
            state["signal_budget"] = st.slider(
                "每日訊號預算", 3, 25, state["signal_budget"],
                help="上限存在的理由：被洗版的告警系統會在第二週被關掉。",
            )
            state["min_conviction"] = st.slider(
                "最低信心", 0.50, 0.85, state["min_conviction"], 0.01
            )
            state["min_evidence_kinds"] = st.slider(
                "最少證據類別數", 1, 4, state["min_evidence_kinds"],
                help="要求跨來源交叉驗證。設為 1 等於接受單一來源的說法。",
            )

        with st.expander("風控參數"):
            state["max_weight_per_name"] = st.slider(
                "單一標的權重上限", 0.01, 0.15, state["max_weight_per_name"], 0.01
            )
            state["target_portfolio_vol"] = st.slider(
                "組合年化波動目標", 0.05, 0.30, state["target_portfolio_vol"], 0.01
            )
            state["daily_pnl_pct"] = st.number_input(
                "當日損益（% NAV，用於熔斷測試）", -10.0, 10.0, state["daily_pnl_pct"], 0.5
            )
            state["drawdown_pct"] = st.number_input(
                "目前回撤（% NAV）", -50.0, 0.0, state["drawdown_pct"], 1.0
            )

        with st.expander("語意層"):
            state["enable_news"] = st.checkbox(
                "啟用新聞情緒分析（需連外網路）",
                value=state["enable_news"],
                help="只對統計層篩出的候選標的抓取新聞，這是成本閘門的核心設計。",
            )
            st.caption(
                "未啟用時，語意層停用而非視為過期來源——沒訂閱的feed不該讓所有訊號被打折。"
            )

    return state


def run_from_settings(settings: dict[str, Any]) -> DeskState:
    return cached_run(
        min_market_cap=settings["min_market_cap"] * 1e9,
        industries=settings["industries"],
        as_of=settings["as_of"],
        signal_budget=settings["signal_budget"],
        min_conviction=settings["min_conviction"],
        min_evidence_kinds=settings["min_evidence_kinds"],
        max_weight_per_name=settings["max_weight_per_name"],
        target_portfolio_vol=settings["target_portfolio_vol"],
        daily_pnl_pct=settings["daily_pnl_pct"],
        drawdown_pct=settings["drawdown_pct"],
        enable_news=settings["enable_news"],
    )


# --- Rendering ------------------------------------------------------------


def severity_badge(severity: Severity) -> str:
    return (
        f'<span class="desk-badge" style="background:{severity.color}22;'
        f'color:{severity.color};border:1px solid {severity.color}55;">'
        f"{severity.label}</span>"
    )


def render_header(state: DeskState) -> None:
    """Regime, counts and health in one glance."""
    regime = state.regime
    columns = st.columns(5)
    columns[0].metric("市場狀態", regime.label if regime else "未判定")
    columns[1].metric("掃描標的", f"{state.universe_size:,}")
    columns[2].metric(
        "今日訊號", len(state.signals),
        delta=f"壓下 {state.suppressed_signals}" if state.suppressed_signals else None,
        delta_color="off",
    )
    columns[3].metric(
        "行動級 (P1)", len(state.signals_by_severity(Severity.P1_ACTION))
    )
    columns[4].metric("執行耗時", f"{state.elapsed_seconds:.1f}s")
    if regime:
        st.caption(regime.detail)


def render_signal_card(
    signal: SignalCard,
    store: DeskStore | None = None,
    expanded: bool = False,
    key_prefix: str = "",
) -> None:
    """One signal, with its full evidence chain and the trader's response."""
    header = (
        f"{signal.direction_label}　{signal.symbol}　{signal.name}"
        f"　｜　信心 {signal.conviction:.0%}"
        f"　｜　建議權重 {signal.risk.suggested_weight_pct:.2f}%"
    )
    with st.expander(header, expanded=expanded):
        st.markdown(severity_badge(signal.severity), unsafe_allow_html=True)
        st.markdown(f"**論點**：{signal.thesis}")

        left, right = st.columns([3, 2])

        with left:
            st.markdown("**支持證據**")
            for item in signal.evidence:
                arrow = "▲" if item.direction > 0 else "▼" if item.direction < 0 else "＝"
                st.markdown(
                    f'<div class="desk-evidence">{arrow} <b>{item.kind}</b>'
                    f'（權重 {item.weight:.2f}，來源 {item.agent}）<br>{item.detail}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("**反面證據**")
            if not signal.counter_evidence:
                st.markdown('<span class="desk-note">未偵測到反面證據</span>', unsafe_allow_html=True)
            for item in signal.counter_evidence:
                st.markdown(
                    f'<div class="desk-counter">⚠ <b>{item.severity}</b>'
                    f'（{item.agent}）<br>{item.detail}</div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("**風險預算**")
            budget = signal.risk
            if budget.is_tradeable:
                st.markdown(
                    f"- 建議權重：**{budget.suggested_weight_pct:.2f}%**\n"
                    f"- 停損：{budget.stop_level}（{budget.stop_basis}）\n"
                    f"- 目標：{budget.target_level}（風報比 {budget.reward_risk}）\n"
                    f"- 最大虧損：{budget.max_loss_pct_nav:.3f}% NAV\n"
                    f"- 綁定限制：{budget.binding_constraint}"
                )
            else:
                st.warning(f"風控未放行：{budget.binding_constraint}")
            st.markdown(f"**失效條件**：{signal.invalidation}")
            st.markdown(f"**期間**：{signal.horizon}｜**環境**：{signal.regime_context}")
            quality = signal.data_quality
            if quality.stale_sources:
                st.warning(f"過期來源：{'、'.join(quality.stale_sources)}")
            if quality.conflicts:
                st.warning(f"證據衝突：{quality.conflicts[0]}")

        with st.popover("決策鏈與回放"):
            st.code(signal.replay_command, language="bash")
            st.json(signal.model_provenance)

        if store is not None:
            _render_feedback(signal, store, key_prefix)


def _render_feedback(signal: SignalCard, store: DeskStore, key_prefix: str) -> None:
    """Adopt / ignore / veto. This is how human judgement becomes training data."""
    recorded = store.latest_feedback().get(signal.signal_id)
    if recorded:
        label = {"adopted": "已採用", "ignored": "已忽略", "vetoed": "已否決"}[recorded["action"]]
        st.success(f"交易員回饋：{label}" + (f"（{recorded['reason_code']}）" if recorded.get("reason_code") else ""))
        return

    st.markdown("**你的判斷**")
    columns = st.columns([1, 1, 2, 2])
    key = f"{key_prefix}{signal.signal_id}"
    adopted = columns[0].button("採用", key=f"adopt_{key}")
    ignored = columns[1].button("忽略", key=f"ignore_{key}")
    reason = columns[2].selectbox("否決理由", VETO_REASONS, key=f"reason_{key}", label_visibility="collapsed")
    vetoed = columns[3].button("否決", key=f"veto_{key}")

    action = "adopted" if adopted else "ignored" if ignored else "vetoed" if vetoed else None
    if action:
        store.record_feedback(
            Feedback(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action=action,
                reason_code=reason if action == "vetoed" else None,
                note="",
                severity=int(signal.severity),
                responded_at=utc_now().isoformat(),
            )
        )
        st.rerun()


def render_freshness_table(state: DeskState) -> pd.DataFrame:
    """Source freshness with an explicit status column, never a silent gap."""
    rows = []
    for source, meta in state.data_freshness.items():
        rows.append(
            {
                "來源": source,
                "最後事件時間": meta.get("last_event_time"),
                "延遲（小時）": round(meta.get("lag_hours", float("nan")), 1),
                "SLA（小時）": meta.get("expected_lag_hours"),
                "狀態": "過期" if meta.get("stale") else "正常",
            }
        )
    if not state.market.get("news_enabled", False):
        rows.append(
            {
                "來源": "news",
                "最後事件時間": None,
                "延遲（小時）": None,
                "SLA（小時）": None,
                "狀態": "未啟用",
            }
        )
    return pd.DataFrame(rows)


def empty_state(message: str, hint: str = "") -> None:
    st.info(message)
    if hint:
        st.caption(hint)
