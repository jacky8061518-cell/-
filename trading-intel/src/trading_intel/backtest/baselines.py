"""三個基準策略，作為回測引擎的煙霧測試（SPEC 第 11 節 Phase 2 第 5 點）。

基準策略的用途不是賺錢，是**檢查引擎有沒有壞**。如果買進持有的績效跟大盤
差很多，那不是策略有問題，是引擎有問題。

它們同時是後續所有策略的比較基準：一個新策略若贏不過買進持有，
它的複雜度就沒有正當性。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from trading_intel.backtest.engine import MarketView
from trading_intel.core.ids import EntityId


@dataclass(frozen=True)
class BuyAndHold:
    """買進持有。最誠實的基準——它不預測任何事。"""

    entity_ids: tuple[EntityId, ...]

    @property
    def name(self) -> str:
        return "買進持有"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
        available = [
            entity_id for entity_id in self.entity_ids if view.latest(entity_id) is not None
        ]
        if not available:
            return {}
        weight = 1.0 / len(available)
        return dict.fromkeys(available, weight)


@dataclass(frozen=True)
class Momentum12m1:
    """12-1 動量：以過去 12 個月報酬排序，但跳過最近 1 個月。

    跳過最近一個月是這個策略的關鍵。短期反轉效應會污染動量訊號——
    上個月漲最多的股票，這個月傾向回吐。文獻上的 12-1 動量之所以穩健，
    很大一部分來自這個「跳過」。
    """

    entity_ids: tuple[EntityId, ...]
    top_n: int = 3
    lookback_days: int = 252
    skip_days: int = 21

    @property
    def name(self) -> str:
        return "12-1 動量"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
        scores: dict[EntityId, float] = {}
        for entity_id in self.entity_ids:
            bars = view.history(entity_id, lookback=self.lookback_days)
            if len(bars) < self.lookback_days:
                continue
            # 跳過最近 skip_days，避開短期反轉。
            end_bar = bars[-self.skip_days - 1]
            start_bar = bars[0]
            if start_bar.close <= 0:
                continue
            scores[entity_id] = float(end_bar.close / start_bar.close - Decimal("1"))

        # 只做多動量為正者。全部為負時空手，不硬選最不差的。
        positive = {entity_id: score for entity_id, score in scores.items() if score > 0}
        if not positive:
            return {}
        ranked = sorted(positive.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[: self.top_n]
        weight = 1.0 / len(selected)
        return {entity_id: weight for entity_id, _ in selected}


@dataclass(frozen=True)
class InstitutionalFlowRanking:
    """三大法人買超排序。

    這是台股特有的訊號。買超金額由外部提供而非從價格推導，
    因此策略本身只負責排序與配置。

    ``flows`` 的鍵是 (交易日, 標的)，值是當日買超金額。
    引擎保證策略只看得到 ``view.as_of_date`` 以前的資料，
    但本策略仍自行過濾一次——訊號來源在引擎之外，這道防線必須自己顧。
    """

    entity_ids: tuple[EntityId, ...]
    flows: Mapping[tuple[date, EntityId], float]
    top_n: int = 3
    lookback_days: int = 20

    @property
    def name(self) -> str:
        return "三大法人買超排序"

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]:
        cutoff = view.as_of_date
        totals: dict[EntityId, float] = {}
        for (flow_date, entity_id), value in self.flows.items():
            # 自行過濾：外部訊號源不受引擎的 MarketView 保護。
            if flow_date > cutoff:
                continue
            if flow_date < cutoff - timedelta(days=self.lookback_days):
                continue
            if entity_id not in self.entity_ids:
                continue
            totals[entity_id] = totals.get(entity_id, 0.0) + value

        positive = {entity_id: total for entity_id, total in totals.items() if total > 0}
        if not positive:
            return {}
        ranked = sorted(positive.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[: self.top_n]
        weight = 1.0 / len(selected)
        return {entity_id: weight for entity_id, _ in selected}


def equal_weights(entity_ids: Sequence[EntityId]) -> dict[EntityId, float]:
    """等權配置。空集合回傳空字典而非除以零。"""
    if not entity_ids:
        return {}
    weight = 1.0 / len(entity_ids)
    return dict.fromkeys(entity_ids, weight)
