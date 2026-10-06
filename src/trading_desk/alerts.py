"""Alert routing. The scarce resource being allocated here is attention.

An alerting system that floods the trader gets muted in week two, and a muted
system has negative value because everyone assumes it is still watching. So the
budget is enforced here: anything that does not fit the P1 quota is demoted to
a lower band rather than sent anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Alert, DeskState, Severity, SignalCard, stable_id, utc_now

CHANNELS = {
    Severity.P0_CRISIS: "電話 + 推播 + 桌面彈窗",
    Severity.P1_ACTION: "推播 + 桌面",
    Severity.P2_ATTENTION: "桌面通知",
    Severity.P3_INTEL: "面板內，不主動打擾",
    Severity.P4_LOG: "僅寫入日誌",
}

# Daily quota per band. Exceeding it means the thresholds are too loose, not
# that the trader needs more notifications.
DAILY_QUOTA = {Severity.P1_ACTION: 5, Severity.P2_ATTENTION: 15}


@dataclass
class AlertRouter:
    """Converts signals, breakers and health problems into routed alerts."""

    quota: dict[Severity, int] | None = None

    def route(
        self,
        signals: list[SignalCard],
        breaker=None,
        degradations: list[str] | None = None,
        stale_sources: tuple[str, ...] = (),
    ) -> list[Alert]:
        quota = dict(self.quota or DAILY_QUOTA)
        alerts: list[Alert] = []
        now = utc_now()

        if breaker is not None and breaker.tripped:
            alerts.append(
                Alert(
                    alert_id=stable_id("alt", "breaker", breaker.reason),
                    severity=Severity.P0_CRISIS
                    if breaker.level == "P0"
                    else Severity.P1_ACTION,
                    channel=CHANNELS[Severity.P0_CRISIS],
                    title=f"熔斷觸發：{breaker.reason}",
                    body=f"動作：{breaker.action}。恢復方式：{breaker.recovery}。",
                    created_at=now,
                    source="risk_engine",
                )
            )

        for signal in sorted(signals, key=lambda card: -card.conviction):
            severity = signal.severity
            if severity in quota:
                if quota[severity] <= 0:
                    # Demote rather than drop: the intel stays available in the
                    # console, it just stops interrupting.
                    severity = Severity(min(int(severity) + 1, int(Severity.P3_INTEL)))
                else:
                    quota[severity] -= 1
            # Write the routed band back onto the card so the signal list and
            # the alert list can never tell the trader two different stories.
            signal.severity = severity
            alerts.append(
                Alert(
                    alert_id=stable_id("alt", signal.signal_id),
                    severity=severity,
                    channel=CHANNELS[severity],
                    title=(
                        f"{signal.direction_label} {signal.symbol} {signal.name}"
                        f"（信心 {signal.conviction:.0%}）"
                    ),
                    body=signal.thesis,
                    created_at=now,
                    source="decision_layer",
                    symbol=signal.symbol,
                    signal_id=signal.signal_id,
                )
            )

        for source in stale_sources:
            alerts.append(
                Alert(
                    alert_id=stable_id("alt", "stale", source),
                    severity=Severity.P2_ATTENTION,
                    channel=CHANNELS[Severity.P2_ATTENTION],
                    title=f"資料來源過期：{source}",
                    body="依賴此來源的訊號信心已自動打折，請確認上游擷取任務。",
                    created_at=now,
                    source="data_quality",
                )
            )

        for reason in degradations or []:
            alerts.append(
                Alert(
                    alert_id=stable_id("alt", "degraded", reason),
                    severity=Severity.P2_ATTENTION,
                    channel=CHANNELS[Severity.P2_ATTENTION],
                    title="系統降級",
                    body=reason,
                    created_at=now,
                    source="agent_runner",
                )
            )

        return sorted(alerts, key=lambda alert: (int(alert.severity), alert.title))


def adoption_rate(feedback: list[dict]) -> float:
    """Share of P1 alerts the trader actually acted on.

    The single most important health metric for the desk: below roughly half,
    the P1 definition is too loose and must be tightened.
    """
    actionable = [row for row in feedback if row.get("severity") == int(Severity.P1_ACTION)]
    if not actionable:
        return float("nan")
    adopted = sum(1 for row in actionable if row.get("action") == "adopted")
    return adopted / len(actionable)
