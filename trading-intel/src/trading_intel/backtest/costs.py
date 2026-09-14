"""交易成本模型。

SPEC 5.3：**成本模型必須內建於回測，不是事後扣除。**

差別不只是算術。事後扣除會讓策略在回測中做出「不含成本時最佳」的決定，
再對那些決定收費；內建則讓成本影響每一筆決策——高換手策略在內建成本下
會自己變得不那麼想換手。前者高估的不只是報酬，是策略本身。

SPEC 第 12 節特別點名：「交易成本低估，尤其台股證交稅對高換手策略是致命的」。
證交稅 0.3% 只在賣出時課，但對一個月換手一次的策略，一年就是 3.6% 的拖累。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from trading_intel.core.settings import CostModel

#: 一個基點 = 萬分之一。
BASIS_POINT = Decimal("0.0001")


@dataclass(frozen=True)
class TradeCost:
    """一筆交易的成本拆解。拆開是為了讓歸因看得出錢花在哪。"""

    commission: Decimal
    transaction_tax: Decimal
    slippage: Decimal
    market_impact: Decimal

    @property
    def total(self) -> Decimal:
        return self.commission + self.transaction_tax + self.slippage + self.market_impact

    def as_bps_of(self, notional: Decimal) -> float:
        if notional <= 0:
            return 0.0
        return float(self.total / notional / BASIS_POINT)


def taiwan_trade_cost(
    *,
    notional: Decimal,
    is_sell: bool,
    costs: CostModel,
    average_daily_volume_value: Decimal | None = None,
    is_day_trade: bool = False,
) -> TradeCost:
    """計算一筆台股交易的成本。

    - 手續費：買賣各課一次。
    - 證交稅：**只在賣出時課徵**，當沖減半。這個不對稱是台股的關鍵特性。
    - 滑價：以固定基點估計。
    - 市場衝擊：平方根模型（SPEC 5.3 指定），下單量佔日均量越高，衝擊越大。
    """
    if notional < 0:
        msg = "交易金額不得為負數"
        raise ValueError(msg)

    commission = notional * Decimal(str(costs.commission_bps)) * BASIS_POINT

    if is_sell:
        tax_rate = costs.tw_daytrade_tax if is_day_trade else costs.tw_transaction_tax
        transaction_tax = notional * Decimal(str(tax_rate))
    else:
        transaction_tax = Decimal("0")

    slippage = notional * Decimal(str(costs.slippage_bps)) * BASIS_POINT

    market_impact = Decimal("0")
    if average_daily_volume_value and average_daily_volume_value > 0:
        participation = float(notional / average_daily_volume_value)
        # 平方根衝擊模型：impact ∝ sqrt(參與率)。
        impact_fraction = costs.impact_coefficient * math.sqrt(max(participation, 0.0))
        market_impact = notional * Decimal(str(impact_fraction))

    return TradeCost(
        commission=commission,
        transaction_tax=transaction_tax,
        slippage=slippage,
        market_impact=market_impact,
    )


def round_trip_cost_bps(costs: CostModel) -> float:
    """一買一賣的來回成本（基點），不含市場衝擊。

    用於快速判斷一個策略的預期報酬是否連成本都覆蓋不了。
    """
    commission = costs.commission_bps * 2
    tax = costs.tw_transaction_tax / float(BASIS_POINT)
    slippage = costs.slippage_bps * 2
    return commission + tax + slippage
