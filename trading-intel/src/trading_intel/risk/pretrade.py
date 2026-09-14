"""事前風控：下單意圖產生時的限額檢查（SPEC 7.1）。

**全部是確定性程式碼。** LLM 不參與任何限額判斷——這不是效率考量，
是因為風控不能有不確定性。同一組部位在同一組限額下，答案必須永遠相同。

設計上有一個刻意的選擇：**檢查全部跑完才回傳，不在第一個違規就中斷。**
交易員需要知道「這筆意圖違反了哪三條」，而不是「它違反了某一條，修好再來問」。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading_intel.core.clock import utc_now
from trading_intel.core.ids import EntityId
from trading_intel.core.settings import RiskLimits
from trading_intel.core.types import RiskVerdict


class LimitCode(StrEnum):
    """限額代碼。用列舉而非字串，讓違規類型可以被統計與監控。"""

    MAX_POSITION_WEIGHT = "MAX_POSITION_WEIGHT"
    MAX_SECTOR_WEIGHT = "MAX_SECTOR_WEIGHT"
    MAX_GROSS_EXPOSURE = "MAX_GROSS_EXPOSURE"
    MAX_NET_EXPOSURE = "MAX_NET_EXPOSURE"
    ADV_PARTICIPATION = "ADV_PARTICIPATION"
    TOP5_CONCENTRATION = "TOP5_CONCENTRATION"
    RESTRICTED_LIST = "RESTRICTED_LIST"


@dataclass(frozen=True)
class PortfolioState:
    """檢查所需的組合快照。"""

    weights: Mapping[EntityId, float]
    sectors: Mapping[EntityId, str]
    #: 各標的的 20 日均量金額，用於流動性檢查。
    adv_values: Mapping[EntityId, Decimal]
    equity: Decimal

    def sector_weights(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for entity_id, weight in self.weights.items():
            sector = self.sectors.get(entity_id, "未分類")
            totals[sector] = totals.get(sector, 0.0) + weight
        return totals

    @property
    def gross_exposure(self) -> float:
        return sum(abs(weight) for weight in self.weights.values())

    @property
    def net_exposure(self) -> float:
        return sum(self.weights.values())

    def top_n_concentration(self, n: int = 5) -> float:
        magnitudes = sorted((abs(weight) for weight in self.weights.values()), reverse=True)
        return sum(magnitudes[:n])


@dataclass(frozen=True)
class Breach:
    """一項違規，含實際值與上限，讓訊息可以直接給人看。"""

    code: LimitCode
    entity_id: EntityId | None
    observed: float
    limit: float
    detail: str


def check_pretrade(
    proposed: Mapping[EntityId, float],
    state: PortfolioState,
    limits: RiskLimits,
    *,
    restricted: Sequence[EntityId] = (),
    now: datetime | None = None,
) -> tuple[RiskVerdict, tuple[Breach, ...]]:
    """檢查一組目標權重是否通過所有事前限額。

    回傳 ``RiskVerdict`` 與詳細的違規清單。``RiskVerdict`` 的型別契約
    （Phase 0 定義）保證否決時必須列出至少一項違反的限額。
    """
    stamp = utc_now() if now is None else now
    breaches: list[Breach] = []
    restricted_set = set(restricted)

    # 1. 禁止清單。處置股、全額交割、財報空窗期、內部人名單。
    for entity_id, weight in proposed.items():
        if entity_id in restricted_set and weight != 0:
            breaches.append(
                Breach(
                    code=LimitCode.RESTRICTED_LIST,
                    entity_id=entity_id,
                    observed=weight,
                    limit=0.0,
                    detail=f"{entity_id} 在禁止清單內，不得建立或加碼部位",
                )
            )

    # 2. 單一標的上限。
    for entity_id, weight in proposed.items():
        if abs(weight) > limits.max_position_weight:
            breaches.append(
                Breach(
                    code=LimitCode.MAX_POSITION_WEIGHT,
                    entity_id=entity_id,
                    observed=abs(weight),
                    limit=limits.max_position_weight,
                    detail=(
                        f"{entity_id} 權重 {abs(weight):.2%} "
                        f"超過單一標的上限 {limits.max_position_weight:.2%}"
                    ),
                )
            )

    merged = PortfolioState(
        weights=dict(proposed),
        sectors=state.sectors,
        adv_values=state.adv_values,
        equity=state.equity,
    )

    # 3. 產業上限。
    for sector, weight in merged.sector_weights().items():
        if abs(weight) > limits.max_sector_weight:
            breaches.append(
                Breach(
                    code=LimitCode.MAX_SECTOR_WEIGHT,
                    entity_id=None,
                    observed=abs(weight),
                    limit=limits.max_sector_weight,
                    detail=(
                        f"產業「{sector}」曝險 {abs(weight):.2%} "
                        f"超過上限 {limits.max_sector_weight:.2%}"
                    ),
                )
            )

    # 4. 總曝險與淨曝險。
    if merged.gross_exposure > limits.max_gross_exposure:
        breaches.append(
            Breach(
                code=LimitCode.MAX_GROSS_EXPOSURE,
                entity_id=None,
                observed=merged.gross_exposure,
                limit=limits.max_gross_exposure,
                detail=f"總曝險 {merged.gross_exposure:.2%} 超過上限",
            )
        )
    if abs(merged.net_exposure) > limits.max_net_exposure:
        breaches.append(
            Breach(
                code=LimitCode.MAX_NET_EXPOSURE,
                entity_id=None,
                observed=abs(merged.net_exposure),
                limit=limits.max_net_exposure,
                detail=f"淨曝險 {merged.net_exposure:.2%} 超過上限",
            )
        )

    # 5. 流動性：部位不得超過 20 日均量的設定比例。
    for entity_id, weight in proposed.items():
        adv = state.adv_values.get(entity_id)
        if adv is None or adv <= 0:
            continue
        notional = state.equity * Decimal(str(abs(weight)))
        participation = float(notional / adv)
        if participation > limits.adv_participation_cap:
            breaches.append(
                Breach(
                    code=LimitCode.ADV_PARTICIPATION,
                    entity_id=entity_id,
                    observed=participation,
                    limit=limits.adv_participation_cap,
                    detail=(
                        f"{entity_id} 佔 20 日均量 {participation:.2%}，"
                        f"超過上限 {limits.adv_participation_cap:.2%}。"
                        "回測結果將無法在實盤重現"
                    ),
                )
            )

    # 6. 前五大集中度。
    concentration = merged.top_n_concentration(5)
    if concentration > limits.top5_concentration_cap:
        breaches.append(
            Breach(
                code=LimitCode.TOP5_CONCENTRATION,
                entity_id=None,
                observed=concentration,
                limit=limits.top5_concentration_cap,
                detail=(
                    f"前五大部位合計 {concentration:.2%} "
                    f"超過上限 {limits.top5_concentration_cap:.2%}"
                ),
            )
        )

    approved = not breaches
    verdict = RiskVerdict(
        event_time=stamp,
        ingest_time=stamp,
        intent_hash=_hash_weights(proposed),
        approved=approved,
        breached_limits=tuple(breach.code.value for breach in breaches),
        note="通過全部事前檢查" if approved else f"違反 {len(breaches)} 項限額",
    )
    return verdict, tuple(breaches)


def _hash_weights(weights: Mapping[EntityId, float]) -> str:
    import hashlib
    import json

    canonical = json.dumps(
        {str(key): round(value, 10) for key, value in sorted(weights.items())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
