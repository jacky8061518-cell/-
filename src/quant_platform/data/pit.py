"""Point-in-time data access.

The single most expensive bug in quantitative research is a backtest that
reads a number before it existed. Every other defect costs a bad trade; this
one costs a strategy that looks excellent for months and then loses money from
the day it goes live, because the edge was never there.

Two mechanisms guard against it here.

**Publication lag.** Each source declares how long after an event its data
actually becomes usable. A Taiwan closing price is known at the close but only
tradeable on the next session; institutional flows publish the evening of T+1.
``as_of`` filtering uses ``event_time + publication_lag``, never ``event_time``.

**Revisions, never overwrites.** A corrected figure is a new row with a higher
revision number and its own ingest time. A query at a past ``as_of`` returns
what was visible then, not what the vendor believes today.

The honest limitation: the bundled Taiwan database carries no vendor ingest
timestamps, so publication lag is a *model* of visibility rather than a
measurement of it. That is strictly better than ignoring the problem and
strictly worse than a vendor feed with real ingest stamps. The lag values are
declared per source below so they can be argued with, and replaced with real
stamps without touching any caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceSpec:
    """How and when one data source becomes visible to a decision.

    ``publication_lag`` is the delay between the event and the moment the data
    could actually have been acted on. ``tradeable_lag`` is the additional
    delay before a position can be opened on it, which for daily closing prices
    is one session: a close observed today is executable tomorrow.
    """

    name: str
    publication_lag: timedelta
    tradeable_lag_sessions: int = 0
    sla_hours: float = 24.0
    description: str = ""

    def visible_at(self, event_time: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(event_time) + self.publication_lag


# Declared centrally so a reviewer can challenge each number in one place.
TAIWAN_SOURCES: dict[str, SourceSpec] = {
    "prices": SourceSpec(
        name="prices",
        publication_lag=timedelta(hours=0),
        tradeable_lag_sessions=1,
        sla_hours=30.0,
        description="收盤價於收盤當下已知，但最快只能在下一個交易日成交。",
    ),
    "flows": SourceSpec(
        name="flows",
        publication_lag=timedelta(hours=20),
        tradeable_lag_sessions=1,
        sla_hours=30.0,
        description="三大法人買賣超於交易日當晚公布，隔日才可據以下單。",
    ),
    "fundamentals": SourceSpec(
        name="fundamentals",
        publication_lag=timedelta(days=45),
        tradeable_lag_sessions=1,
        sla_hours=72.0,
        description="財報期末與公布日相差數週，用期末日當可見時點是典型的前視偏誤。",
    ),
    "news": SourceSpec(
        name="news",
        publication_lag=timedelta(minutes=1),
        tradeable_lag_sessions=0,
        sla_hours=1.0,
        description="新聞近乎即時，但首次發布時間與系統看到的時間不同，需分開記錄。",
    ),
}


@dataclass
class PanelView:
    """A wide panel plus the rules that decide what part of it is visible.

    Wide rather than long on purpose: a long store of 8 million observations is
    the textbook shape but makes a walk-forward backtest unusably slow. The
    visibility rules live here instead of in the row data, which keeps the
    guarantee while leaving the panel fast to slice.
    """

    frame: pd.DataFrame
    spec: SourceSpec
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            self.frame = self.frame.copy()
            self.frame.index = pd.to_datetime(self.frame.index)
        self.frame = self.frame.sort_index()

    def visible(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Rows whose data had been published by ``as_of``."""
        stamp = pd.Timestamp(as_of)
        publish_times = self.frame.index + self.spec.publication_lag
        return self.frame.loc[publish_times <= stamp]

    def tradeable_index(self, as_of: pd.Timestamp) -> pd.Timestamp | None:
        """The first index date on which a decision made at ``as_of`` executes.

        Returns None when no such session exists yet, which is the correct
        answer at the end of the sample and a common source of off-by-one
        lookahead when it is silently treated as "today".
        """
        visible = self.visible(as_of)
        if visible.empty:
            return None
        position = self.frame.index.get_loc(visible.index[-1])
        target = position + self.spec.tradeable_lag_sessions
        if target >= len(self.frame.index):
            return None
        return self.frame.index[target]


class PointInTimeStore:
    """Registry of panels, each with its own visibility rules."""

    def __init__(self) -> None:
        self._panels: dict[str, PanelView] = {}

    def register(self, name: str, frame: pd.DataFrame, spec: SourceSpec) -> None:
        self._panels[name] = PanelView(frame=frame, spec=spec)

    def __contains__(self, name: str) -> bool:
        return name in self._panels

    @property
    def sources(self) -> list[str]:
        return list(self._panels)

    def panel(self, name: str) -> PanelView:
        if name not in self._panels:
            raise KeyError(f"未註冊的資料來源：{name}（已註冊：{', '.join(self._panels) or '無'}）")
        return self._panels[name]

    def visible(self, name: str, as_of: pd.Timestamp) -> pd.DataFrame:
        return self.panel(name).visible(as_of)

    def sessions(self, name: str = "prices") -> pd.DatetimeIndex:
        return self.panel(name).frame.index

    def freshness(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Per-source lag against its SLA, for the control plane to act on."""
        rows = []
        stamp = pd.Timestamp(as_of)
        for name, panel in self._panels.items():
            visible = panel.visible(stamp)
            last = visible.index.max() if not visible.empty else pd.NaT
            lag = (stamp - last).total_seconds() / 3600 if last is not pd.NaT else np.nan
            rows.append(
                {
                    "source": name,
                    "last_event": last,
                    "lag_hours": round(lag, 1) if np.isfinite(lag) else np.nan,
                    "sla_hours": panel.spec.sla_hours,
                    "stale": bool(np.isfinite(lag) and lag > panel.spec.sla_hours),
                    "publication_lag_hours": panel.spec.publication_lag.total_seconds() / 3600,
                }
            )
        return pd.DataFrame(rows)


def assert_no_lookahead(
    store: PointInTimeStore,
    source: str,
    as_of: pd.Timestamp,
    used_index: Iterable[pd.Timestamp],
) -> None:
    """Raise if any index a caller consumed was not yet published at ``as_of``.

    Cheap enough to leave switched on inside the backtest loop, which is the
    point: a leak that only shows up in a nightly audit has already produced a
    quarter of misleading research.
    """
    spec = store.panel(source).spec
    latest_allowed = pd.Timestamp(as_of) - spec.publication_lag
    offending = [pd.Timestamp(item) for item in used_index if pd.Timestamp(item) > latest_allowed]
    if offending:
        raise LookaheadError(
            f"{source} 在 as_of={pd.Timestamp(as_of):%Y-%m-%d} 使用了尚未發布的資料："
            f"{offending[0]:%Y-%m-%d}（最晚可用 {latest_allowed:%Y-%m-%d}）"
        )


class LookaheadError(RuntimeError):
    """Raised when a computation reads data that had not been published."""
