"""協調器：排程、預算分配、衝突仲裁（SPEC 4.2 治理層）。

**這是確定性狀態機，不是 LLM。** 這一點是刻意的，而且是本模組最重要的設計：

> 當兩個 agent 結論相反且都高信心時，升級到人工佇列而非取平均。

取平均是錯的。兩個高信心的相反結論，平均出來是一個低信心的中性結論——
那看起來像是「市場方向不明」，但事實是「我們的系統內部有矛盾」。
前者會讓你安心地不動作，後者應該讓你停下來查清楚。

仲裁規則寫成明確條件式，不交給模型判斷——因為仲裁規則本身就是風控的一部分，
而風控不能有不確定性。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from trading_intel.core.clock import utc_now
from trading_intel.core.enums import DecisionLevel, Direction
from trading_intel.core.ids import EntityId
from trading_intel.core.types import AgentOutput

#: 高於此信心即視為「高信心」，衝突時會升級人工。
HIGH_CONFIDENCE: Final = 0.7

#: 一致性低於此值，代表該任務不適合 LLM，應退回確定性做法（SPEC 7.3）。
MIN_CONSISTENCY: Final = 0.67


class Resolution(StrEnum):
    """仲裁結果。"""

    ACCEPTED = "ACCEPTED"
    #: 全體棄權或證據不足，沒有結論。
    NO_OPINION = "NO_OPINION"
    #: 高信心的相反結論，升級人工佇列。
    ESCALATED = "ESCALATED"
    #: 意見分歧但信心都不高，降級為研究參考。
    DOWNGRADED = "DOWNGRADED"


@dataclass(frozen=True)
class Opinion:
    """一個 agent 對某標的的方向性意見。"""

    agent_name: str
    entity_id: EntityId
    direction: Direction
    confidence: float

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= HIGH_CONFIDENCE


@dataclass(frozen=True)
class Arbitration:
    """一次仲裁的結果與理由。"""

    entity_id: EntityId
    resolution: Resolution
    direction: Direction
    decision_level: DecisionLevel
    reason: str
    opinions: tuple[Opinion, ...]
    decided_at: datetime

    @property
    def needs_human(self) -> bool:
        return self.resolution is Resolution.ESCALATED


def arbitrate(opinions: Sequence[Opinion], *, entity_id: EntityId) -> Arbitration:
    """仲裁多個 agent 對同一標的的意見。

    規則依序判定，第一個命中的即為結論：

    1. 沒有意見 → NO_OPINION
    2. 存在高信心的相反意見 → **ESCALATED，交人工**（絕不取平均）
    3. 意見一致 → ACCEPTED
    4. 意見分歧但都不高信心 → DOWNGRADED 為研究參考
    """
    now = utc_now()
    relevant = [item for item in opinions if item.direction is not Direction.FLAT]

    if not relevant:
        return Arbitration(
            entity_id=entity_id,
            resolution=Resolution.NO_OPINION,
            direction=Direction.FLAT,
            decision_level=DecisionLevel.RESEARCH_ONLY,
            reason="沒有任何 agent 提出方向性意見",
            opinions=tuple(opinions),
            decided_at=now,
        )

    directions = {item.direction for item in relevant}

    if len(directions) > 1:
        high_confidence_directions = {
            item.direction for item in relevant if item.is_high_confidence
        }
        if len(high_confidence_directions) > 1:
            names = "、".join(
                f"{item.agent_name}({item.direction.value}, {item.confidence:.2f})"
                for item in relevant
                if item.is_high_confidence
            )
            return Arbitration(
                entity_id=entity_id,
                resolution=Resolution.ESCALATED,
                direction=Direction.FLAT,
                decision_level=DecisionLevel.RESEARCH_ONLY,
                reason=f"高信心的相反結論，升級人工佇列：{names}",
                opinions=tuple(opinions),
                decided_at=now,
            )
        return Arbitration(
            entity_id=entity_id,
            resolution=Resolution.DOWNGRADED,
            direction=Direction.FLAT,
            decision_level=DecisionLevel.RESEARCH_ONLY,
            reason="意見分歧但信心均不足，降級為研究參考",
            opinions=tuple(opinions),
            decided_at=now,
        )

    direction = next(iter(directions))
    all_high = all(item.is_high_confidence for item in relevant)
    return Arbitration(
        entity_id=entity_id,
        resolution=Resolution.ACCEPTED,
        direction=direction,
        decision_level=DecisionLevel.CONFIRM if all_high else DecisionLevel.RESEARCH_ONLY,
        reason=f"{len(relevant)} 個 agent 意見一致",
        opinions=tuple(opinions),
        decided_at=now,
    )


def consistency_gate(
    outputs: Sequence[AgentOutput[Any]],
    *,
    threshold: float = MIN_CONSISTENCY,
) -> tuple[bool, float]:
    """一致性閘門（SPEC 7.3）。

    同一輸入重複問三次，分歧度過高代表該任務不適合 LLM，應退回確定性做法。
    回傳 (是否通過, 實際一致性)。
    """
    from trading_intel.agents.base import measure_consistency

    score = measure_consistency(outputs)
    return score >= threshold, score


@dataclass
class HumanQueue:
    """人工佇列。升級的項目進這裡，等人決定。

    刻意不提供「自動過期」或「逾時自動放行」。一個沒有人看的升級項目
    應該永遠卡著，而不是安靜地自己通過——後者會讓升級機制變成裝飾。
    """

    items: list[Arbitration] = field(default_factory=list)

    def push(self, arbitration: Arbitration) -> None:
        if arbitration.needs_human:
            self.items.append(arbitration)

    @property
    def pending(self) -> tuple[Arbitration, ...]:
        return tuple(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def resolve(self, entity_id: EntityId) -> Arbitration | None:
        """人工處理完一項後移出佇列。"""
        for index, item in enumerate(self.items):
            if item.entity_id == entity_id:
                return self.items.pop(index)
        return None
