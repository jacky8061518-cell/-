"""
core/universe.py
台股股票池：重用 sector-rotation-research 專案已維護的
data/databases/tw/security-master.csv（上市 + 上櫃證券主檔），
篩選出所有個股（排除 ETF），做為 Z-Score 全市場掃描的標的清單。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECURITY_MASTER_PATH = REPO_ROOT / "data" / "databases" / "tw" / "security-master.csv"


@dataclass(frozen=True)
class TwStock:
    """一檔台股個股的基本資料。"""

    code: str
    yahoo_ticker: str
    name: str
    market: str  # "上市" 或 "上櫃"
    industry: str


def load_stock_universe(path: Path = SECURITY_MASTER_PATH) -> list[TwStock]:
    """讀取上市 + 上櫃全部個股清單（排除 ETF），回傳約 1,900 多檔個股。"""
    df = pd.read_csv(path)
    stocks = df[df["Asset type"] == "股票"]
    return [
        TwStock(
            code=str(row["Code"]),
            yahoo_ticker=str(row["Yahoo ticker"]),
            name=str(row["Name"]),
            market=str(row["Market"]),
            industry=str(row["Industry"]),
        )
        for _, row in stocks.iterrows()
    ]
