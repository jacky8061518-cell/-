"""
hub/costs.py
台股交易成本：直接沿用 trading-intel 專案 configs/base.yaml 的數字，
不另外發明一套（tests/test_costs.py 會比對 yaml，兩邊不可能悄悄分歧）。

- 手續費 14.25 bps：買賣各一次（不打折，寧可高估成本）
- 滑價 5 bps：買賣各一次
- 證交稅 30 bps：只在「賣出」時課徵——放空開倉也是賣出，一樣要課
"""

from __future__ import annotations

COMMISSION_BPS = 14.25
SLIPPAGE_BPS = 5.0
TW_TRANSACTION_TAX = 0.003

BUY_COST = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
SELL_COST = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000 + TW_TRANSACTION_TAX
ROUND_TRIP_COST = BUY_COST + SELL_COST

# 融券／借券費用的保守估計（年化）。台股借券費率依標的浮動，取 2% 做為假設值。
SHORT_BORROW_FEE_ANNUAL = 0.02
