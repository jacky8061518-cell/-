"""Append-only run log and trader feedback.

Every run and every trader response is written as one JSON object per line.
The format is deliberately dumb: it survives schema changes, it is trivially
greppable during an incident, and it can be replayed without a database server.

Nothing is ever updated in place. A correction is a new record with a later
timestamp, which is the same rule the point-in-time data model follows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .contracts import DeskState, SignalCard, Severity, stable_id, utc_now

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "desk"
RUNS_FILE = "runs.jsonl"
SIGNALS_FILE = "signals.jsonl"
FEEDBACK_FILE = "feedback.jsonl"

VETO_REASONS = (
    "資料有誤",
    "我有更新的資訊",
    "風險太高",
    "與現有部位衝突",
    "時機不對",
    "其他",
)


@dataclass(frozen=True)
class Feedback:
    """One trader response to one signal. This is how human judgement is captured."""

    signal_id: str
    symbol: str
    action: str
    reason_code: str | None
    note: str
    severity: int
    responded_at: str

    def __post_init__(self) -> None:
        if self.action not in {"adopted", "ignored", "vetoed"}:
            raise ValueError(f"unknown feedback action: {self.action}")


class DeskStore:
    """File-backed store. One directory, three append-only logs."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- Writing ----------------------------------------------------------

    def record_run(self, state: DeskState) -> None:
        """Persist the run header and every signal it produced."""
        self._append(
            RUNS_FILE,
            {
                "run_id": state.run_id,
                "as_of": state.as_of.isoformat(),
                "universe_size": state.universe_size,
                "regime": state.regime.label if state.regime else None,
                "regime_detail": state.regime.detail if state.regime else None,
                "signals": len(state.signals),
                "suppressed": state.suppressed_signals,
                "elapsed_seconds": state.elapsed_seconds,
                "healthy": state.healthy,
                "degradations": list(getattr(state.board, "degradations", [])),
                "written_at": utc_now().isoformat(),
            },
        )
        for signal in state.signals:
            payload = signal.to_dict()
            payload["run_id"] = state.run_id
            payload["as_of"] = state.as_of.isoformat()
            self._append(SIGNALS_FILE, payload)

    def record_feedback(self, feedback: Feedback) -> None:
        self._append(FEEDBACK_FILE, asdict(feedback))

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.root / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    # --- Reading ----------------------------------------------------------

    def _read(self, filename: str) -> Iterator[dict[str, Any]]:
        path = self.root / filename
        if not path.exists():
            return iter(())
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def runs(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._read(RUNS_FILE)))

    def signals(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._read(SIGNALS_FILE)))

    def feedback(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._read(FEEDBACK_FILE)))

    def latest_feedback(self) -> dict[str, dict[str, Any]]:
        """Most recent response per signal, since a trader may change their mind."""
        rows = list(self._read(FEEDBACK_FILE))
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[row["signal_id"]] = row
        return latest

    def find_decision(self, decision_id: str) -> list[dict[str, Any]]:
        """Every recorded signal belonging to one decision, for replay."""
        return [row for row in self._read(SIGNALS_FILE) if row.get("decision_id") == decision_id]


def evaluate_signals(
    store: DeskStore,
    prices: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Score past signals against what actually happened.

    This is the post-mortem loop. It is intentionally blunt: forward return over
    a fixed horizon, direction-adjusted, with no attempt to model the stop. A
    stop-aware evaluation belongs with the triple-barrier labelling described in
    the design notes, not here.
    """
    signals = store.signals()
    if signals.empty:
        return pd.DataFrame()

    index = pd.to_datetime(prices.index)
    rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        symbol = signal["symbol"]
        if symbol not in prices.columns:
            continue
        as_of = pd.Timestamp(signal["as_of"]).tz_localize(None)
        future = index[index > as_of]
        if len(future) < horizon_days:
            continue
        series = prices[symbol].ffill()
        entry = float(series.asof(as_of))
        exit_price = float(series.asof(future[horizon_days - 1]))
        if not (entry > 0 and exit_price > 0):
            continue
        raw = exit_price / entry - 1.0
        realised = raw if signal["direction"] == "long" else -raw
        rows.append(
            {
                "signal_id": signal["signal_id"],
                "as_of": as_of,
                "symbol": symbol,
                "name": signal.get("name", symbol),
                "direction": signal["direction"],
                "conviction": signal["conviction"],
                "severity": signal.get("severity"),
                "forward_return": realised,
                "hit": realised > 0,
            }
        )
    return pd.DataFrame(rows)


def performance_by_conviction(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Does higher conviction actually mean a higher hit rate?

    If this table is flat, the conviction score is decoration and the decision
    layer needs rework before anyone sizes a position with it.
    """
    if evaluation.empty:
        return pd.DataFrame()
    frame = evaluation.copy()
    frame["band"] = pd.cut(
        frame["conviction"],
        bins=[0, 0.6, 0.7, 0.8, 1.0],
        labels=["0.55-0.60", "0.60-0.70", "0.70-0.80", "0.80+"],
    )
    grouped = frame.groupby("band", observed=True).agg(
        signals=("signal_id", "count"),
        hit_rate=("hit", "mean"),
        mean_return=("forward_return", "mean"),
        worst=("forward_return", "min"),
    )
    return grouped.reset_index()
