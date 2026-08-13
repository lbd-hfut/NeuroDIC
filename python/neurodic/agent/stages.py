"""Canonical-DAG closure and dry-run execution mapping for trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .adapters import ADAPTERS
from .execution_registry import capability_for


@dataclass(frozen=True)
class StagePlanRecord:
    stage_id: str
    status: str
    reasons: Sequence[str]
    scientifically_reusable: bool
    adapter_can_skip: bool

    def to_dict(self) -> dict:
        return {"stage_id": self.stage_id, "status": self.status, "reasons": list(self.reasons),
                "scientifically_reusable": self.scientifically_reusable, "adapter_can_skip": self.adapter_can_skip}


@dataclass(frozen=True)
class ExecutionActionRecord:
    action_id: str
    adapter_id: str
    covers_stages: Sequence[str]
    mode: str = "would_execute"
    execution_supported: bool = False
    notes: str = "Current APIs expose only coarse workflow calls; this planner never invokes them."

    def to_dict(self) -> dict:
        capability = capability_for(self.action_id).to_dict()
        return {"action_id": self.action_id, "adapter_id": self.adapter_id, "covers_stages": list(self.covers_stages),
                "mode": self.mode, "execution_supported": capability["execution_supported"], "notes": self.notes, **capability}


def stage_specs(solver: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...], str]]:
    return {stage: (tuple(deps), tuple(expected), granularity)
            for stage, deps, _required, expected, granularity in ADAPTERS[solver].stages()}


def downstream_closure(solver: str, direct: Iterable[str]) -> tuple[str, ...]:
    specs = stage_specs(solver); reverse = {name: set() for name in specs}
    for stage, (deps, _, _) in specs.items():
        for dependency in deps: reverse[dependency].add(stage)
    pending = list(direct); seen = set(pending)
    while pending:
        stage = pending.pop()
        for child in reverse[stage]:
            if child not in seen: seen.add(child); pending.append(child)
    return tuple(name for name in specs if name in seen)


def execution_actions(solver: str, required: Iterable[str]) -> tuple[ExecutionActionRecord, ...]:
    specs = stage_specs(solver); requested = set(required); grouped: dict[str, list[str]] = {}
    selected_granularities = {granularity for stage, (_, _, granularity) in specs.items()
                              if stage in requested and granularity != "preparation"}
    for stage, (_, _, granularity) in specs.items():
        if granularity in selected_granularities:
            grouped.setdefault(granularity, []).append(stage)
    return tuple(ExecutionActionRecord(action_id=f"{solver}.{granularity}", adapter_id=solver, covers_stages=tuple(stages))
                 for granularity, stages in sorted(grouped.items()))
