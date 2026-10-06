"""Model registry and lifecycle. The control plane's veto over deployment.

A model reaches production by walking a fixed path, one step at a time, and
each step has an entry condition that this module enforces rather than
suggests. There is no argument you can make to the registry: if the validation
record says the gate failed, promotion raises.

    research → validated → shadow → paper → canary → production
                                                        ↓
                                                    retired

The registry deliberately does not import the research plane. It stores the
*record* of a validation, not the machinery that produced one, so the rules for
deploying a model cannot drift when the research code is refactored, and a
researcher cannot widen a gate by editing the module that checks it.

Every transition is appended to a log. Nothing is ever edited in place, because
the question asked after a loss is always "what did we know, and when".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "registry"


class Stage(str, Enum):
    RESEARCH = "research"
    VALIDATED = "validated"
    SHADOW = "shadow"
    PAPER = "paper"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"

    @property
    def label(self) -> str:
        return {
            Stage.RESEARCH: "研究中",
            Stage.VALIDATED: "已驗證",
            Stage.SHADOW: "影子執行",
            Stage.PAPER: "模擬交易",
            Stage.CANARY: "小額實盤",
            Stage.PRODUCTION: "正式部署",
            Stage.RETIRED: "已退役",
        }[self]

    @property
    def trades_real_money(self) -> bool:
        return self in {Stage.CANARY, Stage.PRODUCTION}


# The only legal moves. Anything else is a bug or an attempt to skip a gate.
ALLOWED: dict[Stage, set[Stage]] = {
    Stage.RESEARCH: {Stage.VALIDATED, Stage.RETIRED},
    Stage.VALIDATED: {Stage.SHADOW, Stage.RETIRED},
    Stage.SHADOW: {Stage.PAPER, Stage.RESEARCH, Stage.RETIRED},
    Stage.PAPER: {Stage.CANARY, Stage.SHADOW, Stage.RETIRED},
    Stage.CANARY: {Stage.PRODUCTION, Stage.PAPER, Stage.RETIRED},
    Stage.PRODUCTION: {Stage.CANARY, Stage.RETIRED},
    Stage.RETIRED: set(),
}

# Minimum observation before the next step. Time in shadow and paper is not
# bureaucracy: it is the only way to see the gap between modelled and realised
# execution, which no backtest can show.
MIN_DAYS_IN_STAGE: dict[Stage, int] = {
    Stage.SHADOW: 30,
    Stage.PAPER: 60,
    Stage.CANARY: 30,
}

# Stages a human must sign off on, regardless of how good the numbers look.
REQUIRES_HUMAN_APPROVAL = {Stage.CANARY, Stage.PRODUCTION}


class PromotionRefused(RuntimeError):
    """Raised when a transition would skip or violate a gate."""


@dataclass
class ModelRecord:
    """One versioned model and where it stands."""

    model_id: str
    factor_id: str
    stage: Stage
    created_at: str
    updated_at: str
    spec: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    @property
    def validated(self) -> bool:
        return bool(self.validation.get("approved"))

    def days_in_stage(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        entered = datetime.fromisoformat(self.updated_at)
        if entered.tzinfo is None:
            entered = entered.replace(tzinfo=timezone.utc)
        return (now - entered).total_seconds() / 86400.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload


class ModelRegistry:
    """Append-only registry with enforced transitions."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "models.jsonl"

    # --- Reading ----------------------------------------------------------

    def _all_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def all(self) -> dict[str, ModelRecord]:
        """Current state of every model, rebuilt from the event log."""
        records: dict[str, ModelRecord] = {}
        for event in self._all_events():
            record = ModelRecord(
                model_id=event["model_id"],
                factor_id=event["factor_id"],
                stage=Stage(event["stage"]),
                created_at=event["created_at"],
                updated_at=event["updated_at"],
                spec=event.get("spec", {}),
                validation=event.get("validation", {}),
                history=event.get("history", []),
                notes=event.get("notes", ""),
            )
            records[record.model_id] = record
        return records

    def get(self, model_id: str) -> ModelRecord:
        records = self.all()
        if model_id not in records:
            raise KeyError(f"registry 中沒有模型 {model_id}")
        return records[model_id]

    def deployable(self) -> list[ModelRecord]:
        """Models the trading plane is allowed to execute. Nothing else."""
        return [r for r in self.all().values() if r.stage.trades_real_money]

    # --- Writing ----------------------------------------------------------

    def register(
        self,
        model_id: str,
        factor_id: str,
        spec: dict[str, Any],
        validation: dict[str, Any],
        notes: str = "",
    ) -> ModelRecord:
        """Record a newly researched model. Always starts at ``research``."""
        now = datetime.now(timezone.utc).isoformat()
        record = ModelRecord(
            model_id=model_id,
            factor_id=factor_id,
            stage=Stage.RESEARCH,
            created_at=now,
            updated_at=now,
            spec=spec,
            validation=validation,
            notes=notes,
            history=[{"at": now, "to": Stage.RESEARCH.value, "reason": "registered"}],
        )
        self._append(record)
        return record

    def promote(
        self,
        model_id: str,
        target: Stage,
        approver: str | None = None,
        reason: str = "",
        now: datetime | None = None,
    ) -> ModelRecord:
        """Move a model one step, or refuse and say exactly why."""
        record = self.get(model_id)
        current = record.stage

        if target not in ALLOWED[current]:
            legal = "、".join(s.value for s in ALLOWED[current]) or "無（終態）"
            raise PromotionRefused(
                f"{model_id} 不能從 {current.label} 直接進入 {target.label}；"
                f"合法的下一步是：{legal}"
            )

        if target is Stage.VALIDATED and not record.validated:
            failures = record.validation.get("hard_failures", [])
            detail = "、".join(failures) if failures else "驗證報告未標記為通過"
            raise PromotionRefused(f"{model_id} 未通過驗證閘門，不得晉級：{detail}")

        minimum = MIN_DAYS_IN_STAGE.get(current)
        if minimum is not None and target is not Stage.RETIRED:
            elapsed = record.days_in_stage(now)
            if elapsed < minimum:
                raise PromotionRefused(
                    f"{model_id} 在 {current.label} 僅 {elapsed:.1f} 天，"
                    f"需滿 {minimum} 天才可晉級"
                )

        if target in REQUIRES_HUMAN_APPROVAL and not approver:
            raise PromotionRefused(
                f"進入 {target.label} 需要具名的人工核准；系統不會自行決定動用資金"
            )

        stamp = (now or datetime.now(timezone.utc)).isoformat()
        record.history = list(record.history) + [
            {
                "at": stamp,
                "from": current.value,
                "to": target.value,
                "approver": approver,
                "reason": reason,
            }
        ]
        record.stage = target
        record.updated_at = stamp
        self._append(record)
        return record

    def retire(self, model_id: str, reason: str) -> ModelRecord:
        return self.promote(model_id, Stage.RETIRED, approver="system", reason=reason)

    def _append(self, record: ModelRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")


def validation_record(report) -> dict[str, Any]:
    """Flatten a research-plane validation report into a storable record.

    Lives here as a plain adapter so the registry never imports the research
    plane: it takes any object exposing the same surface and keeps only data.
    """
    return {
        "approved": bool(report.approved),
        "verdict": report.verdict(),
        "hard_failures": [g.name for g in report.hard_failures],
        "soft_failures": [g.name for g in report.soft_failures],
        "metrics": {
            key: (None if value != value else round(float(value), 6))
            for key, value in report.metrics.items()
        },
        "gates": [
            {"name": g.name, "passed": bool(g.passed), "threshold": g.threshold, "hard": g.hard}
            for g in report.gates
        ],
    }
