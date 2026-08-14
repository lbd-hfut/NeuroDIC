"""Stable control-plane errors; scientific diagnosis is deliberately elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ERROR_CODES = frozenset({
    "SCHEMA.INVALID", "FILESYSTEM.NOT_FOUND", "FILESYSTEM.OUTSIDE_ROOT",
    "FILESYSTEM.INVALID_PATH", "ARTIFACT.INVALID", "STATE.INVALID",
    "STATE.INVALID_TRANSITION", "CAPABILITY.UNSUPPORTED", "INTERNAL.ERROR",
    "TRIAL.PLAN_STALE", "TRIAL.ROOT_EXISTS", "EXECUTION.UNSUPPORTED",
    "EXECUTION.INPUT_MISMATCH", "EXECUTION.ARTIFACT_INVALID", "EXECUTION.PUBLISH_FAILED",
    "EXECUTION.INTERRUPTED",
    "DEPENDENCY.INVALID", "DEPENDENCY.PRODUCER_MISMATCH", "DEPENDENCY.SCOPE_MISMATCH",
    "DEPENDENCY.CONTENT_MISMATCH",
    "NDEF.CALIBRATION_NOT_MANAGED", "NDEF.ROI_NOT_MANAGED", "NDEF.AUTO_BATCH_UNRESOLVED",
    "NDEF.REAL_SMOKE_UNBOUNDED", "NDEF.RESUME_UNSUPPORTED", "NDEF.ROI_INPUTS_NOT_READY",
    "NDEF.CALIBRATION_REPROJECTION_GATE", "NDEF.CONFIG_TYPE_INVALID",
})


@dataclass(frozen=True)
class ErrorRecord:
    """Machine-readable public error with optional structured context."""

    code: str
    message: str
    recoverable: bool
    stage: str | None = None
    path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"Unknown control-plane error code: {self.code}")
        if not self.message:
            raise ValueError("ErrorRecord.message must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": self.message,
                                 "recoverable": self.recoverable, "details": dict(self.details)}
        if self.stage is not None:
            value["stage"] = self.stage
        if self.path is not None:
            value["path"] = self.path
        return value


class ControlPlaneError(Exception):
    """Exception wrapper used internally; callers serialize ``record`` only."""

    def __init__(self, record: ErrorRecord) -> None:
        self.record = record
        super().__init__(record.message)


def error_envelope(operation: str, error: ErrorRecord, *, request_id: str | None = None):
    """Create an error envelope without exposing an exception traceback."""
    from .schemas import Envelope

    return Envelope(status="error", operation=operation, request_id=request_id, data={}, errors=(error,))
