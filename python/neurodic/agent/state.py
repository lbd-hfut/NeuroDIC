"""Minimal immutable state records and atomic JSON publication primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .artifacts import require_path_within
from .schemas import AGENT_SCHEMA_VERSION, CapabilityRecord, canonical_json, is_utc_timestamp, new_id, utc_now


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


_TRANSITIONS = {
    StageStatus.PENDING: frozenset({StageStatus.RUNNING}),
    StageStatus.RUNNING: frozenset({StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.INTERRUPTED}),
    StageStatus.COMPLETED: frozenset(),
    StageStatus.FAILED: frozenset(),
    StageStatus.INTERRUPTED: frozenset(),
}


def valid_stage_transition(before: StageStatus, after: StageStatus) -> bool:
    """Pure lifecycle validation; stale running state is never silently failed."""
    return after in _TRANSITIONS[before]


@dataclass(frozen=True)
class StageRecord:
    stage_id: str
    status: StageStatus
    attempt: int
    producer_version: str
    stage_attempt_id: str = field(default_factory=lambda: new_id("stage"))
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: Sequence[str] = ()
    capabilities: CapabilityRecord = field(default_factory=CapabilityRecord)
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.stage_id or not self.producer_version or self.attempt < 1:
            raise ValueError("StageRecord requires id, producer version, and attempt >= 1")
        if self.status is StageStatus.RUNNING and self.started_at is None:
            raise ValueError("A running StageRecord requires started_at")
        if self.status in {StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.INTERRUPTED} and self.finished_at is None:
            raise ValueError("A terminal StageRecord requires finished_at")
        for name, value in (("started_at", self.started_at), ("finished_at", self.finished_at)):
            if value is not None and not is_utc_timestamp(value):
                raise ValueError(f"StageRecord.{name} must be a UTC ISO-8601 timestamp")

    def transition(self, status: StageStatus, *, timestamp: str | None = None) -> "StageRecord":
        if not valid_stage_transition(self.status, status):
            raise ValueError(f"Illegal stage transition: {self.status.value} -> {status.value}")
        timestamp = timestamp or utc_now()
        return StageRecord(stage_id=self.stage_id, status=status, attempt=self.attempt,
                           producer_version=self.producer_version, stage_attempt_id=self.stage_attempt_id,
                           started_at=timestamp if status is StageStatus.RUNNING else self.started_at,
                           finished_at=timestamp if status is not StageStatus.RUNNING else None,
                           artifacts=self.artifacts, capabilities=self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "stage_attempt_id": self.stage_attempt_id,
                "stage_id": self.stage_id, "status": self.status.value, "attempt": self.attempt,
                "producer_version": self.producer_version, "started_at": self.started_at,
                "finished_at": self.finished_at, "artifacts": list(self.artifacts),
                "capabilities": self.capabilities.to_dict()}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    solver: str
    case_identity: str
    case_root: str
    scope: Mapping[str, Any]
    evaluation_policy: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now)
    capabilities: CapabilityRecord = field(default_factory=CapabilityRecord)
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((self.run_id, self.solver, self.case_identity, self.case_root)):
            raise ValueError("RunRecord requires id, solver, case identity, and case root")
        if not is_utc_timestamp(self.created_at):
            raise ValueError("RunRecord.created_at must be a UTC ISO-8601 timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "run_id": self.run_id, "solver": self.solver,
                "case_identity": self.case_identity, "case_root": self.case_root, "scope": dict(self.scope),
                "evaluation_policy": dict(self.evaluation_policy), "created_at": self.created_at,
                "capabilities": self.capabilities.to_dict()}


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    run_id: str
    effective_config_identity: str
    seed_policy: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now)
    parent_trial_id: str | None = None
    stages: Sequence[str] = ()
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((self.trial_id, self.run_id, self.effective_config_identity)):
            raise ValueError("TrialRecord requires id, run id, and effective config identity")
        if not is_utc_timestamp(self.created_at):
            raise ValueError("TrialRecord.created_at must be a UTC ISO-8601 timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "trial_id": self.trial_id, "run_id": self.run_id,
                "parent_trial_id": self.parent_trial_id, "effective_config_identity": self.effective_config_identity,
                "seed_policy": dict(self.seed_policy), "stages": list(self.stages), "created_at": self.created_at}


def atomic_write_json(path: str | Path, value: Any, *, root: str | Path) -> None:
    """Validate strict JSON, fsync it, then atomically replace an existing target.

    ``root`` is required so this low-level state primitive cannot accidentally
    publish outside the permitted case/run tree. The parent must already exist;
    this primitive intentionally never creates directories and is never called
    by read-only helpers.
    """
    target = require_path_within(path, root)
    if not target.parent.is_dir():
        raise ValueError(f"Atomic JSON parent directory does not exist: {target.parent}")
    payload = canonical_json(value).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
