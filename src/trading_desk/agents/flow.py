"""Flow agent: where the money actually went.

Deterministic. Institutional net buying is the least narrative-contaminated
input available on the Taiwan market: it is a record of executed trades rather
than an opinion about them. This agent wraps the existing fund-flow research
code and converts its output into blackboard evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sector_rotation.fund_flow import calculate_fund_flow_signals

from ..blackboard import Blackboard
from ..contracts import CounterEvidence, Evidence
from ..statistics import rank_to_normal
from .base import Agent, MarketContext

# Flow score is a 0-100 percentile rank, so a 70 cut would flag 30% of the
# board. Only the tails carry information worth a trader's attention.
FLOW_SCORE_TRIGGER = 88.0
FLOW_SCORE_BEARISH = 12.0
DIVERGENCE_TRIGGER = -0.05

STAGE_HINTS = {
    "資金流入初期": ("錢剛進場，價格尚未完全反映", 1),
    "資金流入加速": ("流入放大且價格跟上", 1),
    "資金流入趨緩": ("流入動能開始降溫", 0),
    "資金流出": ("法人轉為調節", -1),
}


class FlowAgent(Agent):
    name = "flow"
    version = "1.0"
    sources = ("flows", "prices")

    def __init__(self, score_trigger: float = FLOW_SCORE_TRIGGER) -> None:
        self.score_trigger = score_trigger

    def run(self, context: MarketContext, board: Blackboard) -> int:
        flows = context.visible_flows()
        if flows.empty:
            board.record_degradation("flow: 無法人買賣超資料，資金流證據停用")
            board.record_freshness("flows", None, expected_lag_hours=30.0)
            return 0

        last_session = pd.to_datetime(flows["Date"]).max()
        board.record_freshness("flows", last_session, expected_lag_hours=30.0)

        prices = context.visible_prices()
        master = context.extras.get("master") if context.extras else None
        if master is None or master.empty:
            board.record_degradation("flow: 缺少 security master，無法對應產業")
            return 0

        securities, groups = calculate_fund_flow_signals(prices, flows, master)
        if securities.empty:
            board.record_degradation("flow: 資金流計算無結果")
            return 0

        securities = securities.set_index("Ticker")
        self._write_features(securities, board)
        board.market["flow_groups"] = groups
        board.market["flow_session"] = last_session

        return self._emit_evidence(securities, context, board)

    def _write_features(self, securities: pd.DataFrame, board: Blackboard) -> None:
        columns = [
            "Flow score",
            "5D flow intensity",
            "20D flow intensity",
            "Trust 20D intensity",
            "20D net value",
            "5D net value",
            "Foreign 20D value",
        ]
        available = [column for column in columns if column in securities.columns]
        renamed = securities[available].rename(
            columns={
                "Flow score": "flow_score",
                "5D flow intensity": "flow_intensity_5d",
                "20D flow intensity": "flow_intensity_20d",
                "Trust 20D intensity": "trust_intensity_20d",
                "20D net value": "flow_net_value_20d",
                "5D net value": "flow_net_value_5d",
                "Foreign 20D value": "foreign_net_value_20d",
            }
        )
        if "flow_intensity_20d" in renamed:
            renamed["flow_intensity_rank"] = rank_to_normal(renamed["flow_intensity_20d"])
        board.write_features(renamed, agent=self.name)

    def _emit_evidence(
        self,
        securities: pd.DataFrame,
        context: MarketContext,
        board: Blackboard,
    ) -> int:
        findings = 0
        for symbol, row in securities.iterrows():
            score = float(row.get("Flow score", np.nan))
            if not np.isfinite(score):
                continue
            bullish = score >= self.score_trigger
            bearish = score <= FLOW_SCORE_BEARISH
            if not (bullish or bearish):
                continue

            stage = str(row.get("Stage", ""))
            hint, stage_direction = STAGE_HINTS.get(stage, ("", 0))
            direction = 1 if bullish else -1
            intensity = float(row.get("20D flow intensity", np.nan))
            net_value = float(row.get("20D net value", np.nan))

            detail = f"資金流分數 {score:.0f}／100"
            if np.isfinite(intensity):
                detail += f"，20 日買超佔股本 {intensity:+.3%}"
            if np.isfinite(net_value):
                detail += f"（約 {net_value / 1e8:+.1f} 億元）"
            if hint:
                detail += f"；{stage}：{hint}"

            board.view(str(symbol), context.name_of(str(symbol)))
            board.add_evidence(
                str(symbol),
                Evidence(
                    kind="flow",
                    detail=detail,
                    weight=0.35,
                    direction=direction,
                    agent=self.name,
                    value=score,
                ),
            )
            board.tag(str(symbol), "flow")

            # Institutions buying while the price falls is a genuine divergence:
            # either they know something, or they are absorbing distribution.
            trailing_return = float(row.get("20D return", np.nan))
            if bullish and np.isfinite(trailing_return) and trailing_return <= DIVERGENCE_TRIGGER:
                board.add_counter_evidence(
                    str(symbol),
                    CounterEvidence(
                        detail=f"法人買超但 20 日報酬 {trailing_return:.1%}，價格尚未確認",
                        severity="medium",
                        agent=self.name,
                    ),
                )
            if stage_direction < 0 and bullish:
                board.add_counter_evidence(
                    str(symbol),
                    CounterEvidence(
                        detail=f"分數偏高但階段判定為「{stage}」，短線流入動能不足",
                        severity="low",
                        agent=self.name,
                    ),
                )
            findings += 1
        return findings
