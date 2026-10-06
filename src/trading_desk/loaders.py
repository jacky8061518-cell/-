"""Data access for the desk. The only module that knows where files live.

Agents receive a :class:`MarketContext` and never reach for a path or a network
call themselves, which is what lets the identical pipeline run against the local
Parquet database offline and against live feeds in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .agents.base import MarketContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TW_DATABASE = PROJECT_ROOT / "data" / "databases" / "tw"
TW_PRICES = TW_DATABASE / "adjusted-prices.parquet"
TW_FLOWS = TW_DATABASE / "institutional-flows.parquet"
TW_MASTER = TW_DATABASE / "security-master.csv"

TW_BENCHMARK = "0050.TW"


@dataclass(frozen=True)
class MarketData:
    """Raw panels plus the metadata needed to interpret them."""

    prices: pd.DataFrame
    flows: pd.DataFrame
    master: pd.DataFrame
    benchmark: str
    source: str

    @property
    def last_session(self) -> pd.Timestamp:
        return pd.Timestamp(self.prices.index.max())

    @property
    def metadata(self) -> pd.DataFrame:
        return self.master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")


def database_available() -> bool:
    return TW_PRICES.exists() and TW_MASTER.exists()


def load_taiwan_market() -> MarketData:
    """Load the committed Taiwan database. Works with no network access."""
    if not TW_PRICES.exists():
        raise FileNotFoundError(
            f"找不到價格資料庫 {TW_PRICES}，請先執行 scripts/daily_update.py"
        )
    prices = pd.read_parquet(TW_PRICES)
    prices.index = pd.to_datetime(prices.index)
    flows = pd.read_parquet(TW_FLOWS) if TW_FLOWS.exists() else pd.DataFrame()
    master = pd.read_csv(TW_MASTER)
    return MarketData(
        prices=prices.sort_index(),
        flows=flows,
        master=master,
        benchmark=TW_BENCHMARK,
        source="台股本地資料庫",
    )


def filter_universe(
    data: MarketData,
    *,
    asset_types: tuple[str, ...] = ("股票",),
    industries: tuple[str, ...] | None = None,
    min_history: int = 260,
    min_market_cap: float = 0.0,
) -> MarketData:
    """Restrict the universe before scanning.

    Narrowing here rather than inside the agents keeps the cost gate honest:
    the scan cost is proportional to the universe the desk was actually asked
    to watch, not to everything that happens to be in the database.
    """
    master = data.master
    if asset_types:
        master = master[master["Asset type"].isin(asset_types)]
    if industries:
        master = master[master["Industry"].isin(industries)]
    if min_market_cap > 0:
        master = master[_market_cap_proxy(data, master) >= min_market_cap]

    keep = set(master["Yahoo ticker"]) | {data.benchmark}
    columns = [column for column in data.prices.columns if column in keep]
    prices = data.prices[columns]
    prices = prices.loc[:, prices.notna().sum() >= min_history]
    if data.benchmark in data.prices.columns and data.benchmark not in prices.columns:
        prices[data.benchmark] = data.prices[data.benchmark]

    flows = data.flows
    if not flows.empty:
        flows = flows[flows["Ticker"].isin(prices.columns)]

    return MarketData(
        prices=prices,
        flows=flows,
        master=master,
        benchmark=data.benchmark,
        source=data.source,
    )


def build_context(
    data: MarketData,
    as_of: datetime | pd.Timestamp | None = None,
    mode: str = "paper",
    news: pd.DataFrame | None = None,
) -> MarketContext:
    """Assemble the point-in-time context handed to every agent."""
    stamp = pd.Timestamp(as_of) if as_of is not None else data.last_session
    return MarketContext(
        as_of=stamp,
        prices=data.prices,
        metadata=data.metadata,
        benchmark=data.benchmark,
        flows=data.flows,
        news=news,
        mode=mode,
        extras={"master": data.master},
    )


def _market_cap_proxy(data: MarketData, master: pd.DataFrame) -> pd.Series:
    """Issued shares times last close.

    A proxy, not a float-adjusted market cap: the database carries issued
    shares but no free float, and no volume at all. It is good enough to keep
    untradeably small names out of a trader's signal queue, which is the only
    job it has here.
    """
    last_close = data.prices.ffill().iloc[-1]
    shares = pd.to_numeric(master["Issued shares"], errors="coerce")
    return (shares * master["Yahoo ticker"].map(last_close)).fillna(0.0)


def available_industries(data: MarketData) -> list[str]:
    if "Industry" not in data.master.columns:
        return []
    industries = data.master.loc[
        data.master["Asset type"] == "股票", "Industry"
    ].dropna().unique()
    return sorted(str(industry) for industry in industries)


def trading_sessions(data: MarketData, count: int = 60) -> list[pd.Timestamp]:
    """Recent session dates, newest first, for the as-of replay selector."""
    return list(reversed(data.prices.index[-count:]))
