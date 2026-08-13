"""Versioned, strict-JSON record contracts for the agent control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


AGENT_SCHEMA_VERSION = "neurodic.agent/v1"
QUALITY_SCHEMA_VERSION = "neurodic.quality/v1"
DIAGNOSIS_SCHEMA_VERSION = "neurodic.diagnosis/v1"


class Availability(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    NOT_AVAILABLE = "not_available"
    NOT_APPLICABLE = "not_applicable"
    CORRUPT = "corrupt"


def utc_now() -> str:
    """Return a UTC ISO-8601 timestamp with a canonical ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_utc_timestamp(value: str) -> bool:
    """Whether a persistent timestamp uses the required UTC ISO-8601 form."""
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def new_id(prefix: str) -> str:
    """Return an object record ID, never a scientific/content identity."""
    if not prefix or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in prefix):
        raise ValueError("ID prefix must use lowercase ASCII letters, digits, '_' or '-'")
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Strict JSON records cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported public JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Stable strict JSON. Sorted keys define canonical-hash ordering."""
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class CapabilityRecord:
    """Capability dimensions stay independent: reuse, cache, and resume differ."""

    schema_version: str = AGENT_SCHEMA_VERSION
    reuse_supported: bool = False
    cache_supported: bool = False
    resume_supported: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class Envelope:
    """Uniform outer contract for future agent-facing structured operations.

    ``status`` describes operation delivery only. Scientific warnings belong in
    a future QualityReport, so an operation may be ``ok`` while its data reports
    scientific uncertainty.
    """

    status: str
    operation: str
    data: Mapping[str, Any]
    request_id: str | None = None
    warnings: Sequence[Mapping[str, Any]] = ()
    errors: Sequence[Any] = ()
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"ok", "warning", "error"}:
            raise ValueError("Envelope.status must be ok, warning, or error")
        if not self.operation:
            raise ValueError("Envelope.operation must not be empty")
        if self.request_id is None:
            object.__setattr__(self, "request_id", new_id("req"))
        if self.status == "error" and not self.errors:
            raise ValueError("An error envelope requires at least one error")
        if self.status != "error" and self.errors:
            raise ValueError("Only an error envelope may include errors")

    def to_dict(self) -> dict[str, Any]:
        errors = [item.to_dict() if hasattr(item, "to_dict") else item for item in self.errors]
        return _json_value({"schema_version": self.schema_version, "status": self.status,
                            "operation": self.operation, "request_id": self.request_id,
                            "data": self.data, "warnings": list(self.warnings), "errors": errors})


@dataclass(frozen=True)
class QualityReport:
    """Foundation only: no metric, threshold, or finding logic is implemented."""

    solver: str
    scope: Mapping[str, Any]
    status: str = "unknown"
    metrics: Sequence[Mapping[str, Any]] = ()
    threshold_results: Sequence[Mapping[str, Any]] = ()
    findings: Sequence[Mapping[str, Any]] = ()
    failure_stage: str | None = None
    eligibility: Mapping[str, Any] = field(default_factory=lambda: {"best_candidate": False, "reasons": []})
    profile: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"unknown", "pass", "warning", "fail"}:
            raise ValueError("QualityReport.status must be unknown, pass, warning, or fail")
        if not self.solver:
            raise ValueError("QualityReport.solver must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class MetricRecord:
    """One observed or derived scientific metric; never a hidden quality score."""
    id: str
    availability: Availability
    unit: str
    source: Mapping[str, Any]
    value: int | float | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)
    aggregation: str | None = None
    sample_count: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.unit:
            raise ValueError("MetricRecord requires id and unit")
        if self.availability in {Availability.OBSERVED, Availability.DERIVED} and self.value is None:
            raise ValueError("Available MetricRecord requires a value")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class ThresholdResult:
    metric_id: str
    operator: str
    threshold: Any
    availability: Availability
    evaluated: bool
    passed: bool | None
    required: bool
    observed_value: int | float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class FindingRecord:
    code: str
    severity: str
    stage: str
    evidence_refs: Sequence[str]
    message: str
    source: str

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error", "critical"}:
            raise ValueError("Invalid finding severity")
        if self.source not in {"threshold", "integrity"}:
            raise ValueError("Finding source must be threshold or integrity")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class DiagnosisRecord:
    code: str
    failure_stage: str
    failure_family: str
    support: str
    supporting_evidence: Sequence[Mapping[str, Any]] = ()
    contradicting_evidence: Sequence[Mapping[str, Any]] = ()
    missing_evidence: Sequence[Mapping[str, Any]] = ()
    candidate_causes: Sequence[Mapping[str, Any]] = ()
    next_observation: str | None = None
    role: str = "secondary"

    def __post_init__(self) -> None:
        if self.support not in {"strong", "moderate", "weak", "insufficient"}:
            raise ValueError("Invalid diagnosis support")
        if self.role not in {"primary", "secondary", "consequent"}:
            raise ValueError("Invalid diagnosis role")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class DiagnosisReport:
    solver: str
    scope: Mapping[str, Any]
    overall_status: str
    diagnoses: Sequence[DiagnosisRecord] = ()
    primary_diagnosis: str | None = None
    checked_stages: Sequence[str] = ()
    missing_evidence: Sequence[Mapping[str, Any]] = ()
    notes: Sequence[str] = ()
    diagnosis_version: str = "neurodic-diagnosis-rules/v1"
    schema_version: str = DIAGNOSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.overall_status not in {"diagnosed", "partial", "insufficient_evidence", "no_failure_detected"}:
            raise ValueError("Invalid diagnosis overall_status")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)
