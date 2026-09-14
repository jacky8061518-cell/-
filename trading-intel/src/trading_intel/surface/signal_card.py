"""訊號卡（SPEC 8.1）。

給交易員看的，一個訊號一張。SPEC 明訂**失效條件是最重要的欄位**——
「寫明什麼情況代表這個判斷錯了」。這句話值得多說一句：一張只有進場理由、
沒有失效條件的訊號卡，讀起來像是在說服交易員相信它，而不是幫交易員判斷
什麼時候該不相信它。兩者的差別就是整份 SPEC 想解決的問題。

**反方論點是必填，不是加分項。** 沒有 RedTeamAgent 報告的訊號卡拒絕產生——
這一點在 `build_signal_card` 裡是型別層級的要求，不是文件建議。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_intel.agents.schemas import AttackSeverity, RedTeamReport
from trading_intel.core.enums import DecisionLevel, Direction, Horizon
from trading_intel.core.errors import SchemaValidationError
from trading_intel.core.ids import EntityId, EvidenceId, SignalId
from trading_intel.core.types import Signal


@dataclass(frozen=True)
class CounterArgument:
    """反方論點的一條，來自 RedTeamAgent 的攻擊點。"""

    text: str
    severity: AttackSeverity


@dataclass(frozen=True)
class SignalCard:
    """一張完整的訊號卡。

    欄位直接對應 SPEC 8.1 的清單：方向、標的、進場區間、失效條件、
    預期持有期、信心度、支持證據連結、反方論點。
    """

    signal_id: SignalId
    entity_id: EntityId
    direction: Direction
    #: 進場價格區間 (下界, 上界)。None 代表不設區間（例如市價單）。
    entry_range: tuple[float, float] | None
    #: 失效條件。SPEC：最重要的欄位。
    invalidation_condition: str
    horizon: Horizon
    confidence: float
    evidence_ids: tuple[EvidenceId, ...]
    counter_arguments: tuple[CounterArgument, ...]
    decision_level: DecisionLevel
    generated_at: datetime
    model_version: str

    @property
    def has_blocking_counter_argument(self) -> bool:
        """反方論點裡有沒有嚴重到該擋下這張卡的。"""
        return any(
            arg.severity in {AttackSeverity.CRITICAL, AttackSeverity.HIGH}
            for arg in self.counter_arguments
        )

    def to_display_text(self) -> str:
        """純文字版面，供終端機／簡訊等沒有排版能力的通路使用。"""
        lines = [
            f"【訊號卡】{self.entity_id}　{self.direction.value}",
            f"信心度：{self.confidence:.0%}　持有期：{self.horizon.value}",
        ]
        if self.entry_range is not None:
            lines.append(f"進場區間：{self.entry_range[0]:.2f} ~ {self.entry_range[1]:.2f}")
        lines.append(f"失效條件：{self.invalidation_condition}")
        if self.counter_arguments:
            lines.append("反方論點：")
            lines.extend(f"  [{arg.severity.value}] {arg.text}" for arg in self.counter_arguments)
        lines.append(f"證據：{len(self.evidence_ids)} 筆　決策層級：{self.decision_level.name}")
        return "\n".join(lines)


def build_signal_card(
    signal: Signal,
    redteam_report: RedTeamReport,
    *,
    generated_at: datetime,
    entry_range: tuple[float, float] | None = None,
) -> SignalCard:
    """由 Signal 與 RedTeamAgent 的報告組出一張訊號卡。

    ``redteam_report`` 是必要參數，不是可選——SPEC 8.1 要求每張卡都附
    「反方論點（來自 RedTeamAgent）」。棄權的 RedTeamAgent 輸出（
    ``AgentOutput.abstain=True``）代表沒有做完攻擊清單，同樣不得產卡：
    一張沒人找過碴的訊號卡，跟一張沒寫失效條件的訊號卡是同一種風險。
    """
    if redteam_report.findings == ():
        raise SchemaValidationError(
            "訊號卡拒絕在沒有 RedTeamAgent 報告的情況下產生",
            signal_id=str(signal.signal_id),
        )

    counter_arguments = tuple(
        CounterArgument(text=finding.detail, severity=finding.severity)
        for finding in redteam_report.findings
        if finding.severity is not AttackSeverity.NOT_APPLICABLE
    )

    return SignalCard(
        signal_id=signal.signal_id,
        entity_id=signal.entity_id,
        direction=signal.direction,
        entry_range=entry_range,
        invalidation_condition=signal.invalidation_condition,
        horizon=signal.horizon,
        confidence=signal.confidence,
        evidence_ids=signal.evidence_ids,
        counter_arguments=counter_arguments,
        decision_level=signal.decision_level,
        generated_at=generated_at,
        model_version=signal.model_version,
    )
