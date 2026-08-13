"""Native-free, read-only trial/config dry-run planner (Loop 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (ConfigChangeRecord, PolicyViolationRecord, canonical_sparse_override, effective_config_identity,
                     merge_sparse_override, owner_stages, protected_violations)
from .errors import ControlPlaneError, ErrorRecord
from .inspect import inspect_case, resolve_config
from .artifacts import content_identity
from .schemas import Envelope, canonical_json
from .stages import ExecutionActionRecord, StagePlanRecord, downstream_closure, execution_actions, stage_specs


TRIAL_PLAN_SCHEMA_VERSION = "neurodic.trial_plan/v1"


@dataclass(frozen=True)
class ArtifactReuseRecord:
    artifact_type: str
    producer_stage: str
    location: str | None
    identity: Mapping[str, Any] | None
    reuse_status: str
    reason: str
    shared_input: bool = False
    adapter_can_reuse: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_type": self.artifact_type, "producer_stage": self.producer_stage,
                "location": self.location, "identity": dict(self.identity) if self.identity else None,
                "reuse_status": self.reuse_status, "reason": self.reason, "shared_input": self.shared_input,
                "adapter_can_reuse": self.adapter_can_reuse}


@dataclass(frozen=True)
class TrialPlan:
    solver: str
    scope: Mapping[str, Any]
    baseline: Mapping[str, Any]
    trial: Mapping[str, Any]
    override: Mapping[str, Any]
    effective_config_identity: str
    changes: Sequence[ConfigChangeRecord]
    policy_violations: Sequence[PolicyViolationRecord]
    config_invalidated_stages: Sequence[str]
    artifact_missing_or_unusable_stages: Sequence[str]
    stage_plan: Sequence[StagePlanRecord]
    artifact_reuse: Sequence[ArtifactReuseRecord]
    minimum_rerun_stages: Sequence[str]
    execution_actions: Sequence[ExecutionActionRecord]
    plan_status: str
    warnings: Sequence[str] = ()
    dry_run: bool = True
    execution_performed: bool = False
    baseline_writes: Sequence[str] = ()
    schema_version: str = TRIAL_PLAN_SCHEMA_VERSION

    @property
    def plan_identity(self) -> str:
        import hashlib
        core = {"baseline": self.baseline, "trial": self.trial, "solver": self.solver, "scope": self.scope,
                "override": self.override, "effective_config_identity": self.effective_config_identity,
                "changes": [item.to_dict() for item in self.changes],
                "config_invalidated_stages": list(self.config_invalidated_stages),
                "artifact_missing_or_unusable_stages": list(self.artifact_missing_or_unusable_stages),
                "minimum_rerun_stages": list(self.minimum_rerun_stages),
                "execution_actions": [item.to_dict() for item in self.execution_actions]}
        return "sha256:" + hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "plan_identity": self.plan_identity, "plan_status": self.plan_status, "dry_run": self.dry_run,
                "execution_performed": self.execution_performed, "solver": self.solver, "scope": dict(self.scope),
                "baseline": dict(self.baseline), "trial": dict(self.trial), "override": dict(self.override),
                "effective_config_identity": self.effective_config_identity,
                "changes": [item.to_dict() for item in self.changes],
                "policy_violations": [item.to_dict() for item in self.policy_violations],
                "config_invalidated_stages": list(self.config_invalidated_stages),
                "artifact_missing_or_unusable_stages": list(self.artifact_missing_or_unusable_stages),
                "stage_plan": [item.to_dict() for item in self.stage_plan],
                "reusable_stages": [item.stage_id for item in self.stage_plan if item.scientifically_reusable],
                "artifact_reuse": [item.to_dict() for item in self.artifact_reuse],
                "minimum_rerun_stages": list(self.minimum_rerun_stages),
                "execution_actions": [item.to_dict() for item in self.execution_actions],
                "warnings": list(self.warnings), "baseline_writes": list(self.baseline_writes),
                "would_write": []}


def _file_values(value: Any):
    if isinstance(value, str): yield value
    elif isinstance(value, Mapping):
        for item in value.values(): yield from _file_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from _file_values(item)


def _input_identities(case_report: Mapping[str, Any]) -> dict[str, Any]:
    """Content identities for inspector-observed shared inputs only."""
    root = Path(case_report["resolved_case_root"])
    candidates = set(_file_values(case_report.get("frames", {})))
    candidates.update(_file_values(case_report.get("inputs", {})))
    for artifact in case_report.get("artifacts", []):
        if str(artifact.get("producer_stage", "")).endswith(".inputs"):
            candidates.add(str(artifact.get("location", "")))
    identities: dict[str, Any] = {}
    for item in sorted(candidates):
        path = Path(item)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            try: identities[str(path.relative_to(root))] = content_identity(path).to_dict()
            except (OSError, ValueError): pass
    return identities


def _baseline_identity(resolved: Mapping[str, Any], case_report: Mapping[str, Any]) -> dict[str, Any]:
    values = resolved["effective_config"]
    return {"config_source": resolved["solver_config_path"], "config_source_identity": content_identity(resolved["solver_config_path"]).to_dict(),
            "case_paths_source": resolved["case_paths_path"], "case_paths_source_identity": content_identity(resolved["case_paths_path"]).to_dict(),
            "case_key": resolved["case_key"], "effective_config_identity": effective_config_identity(values),
            "case_root": case_report["resolved_case_root"], "solver": resolved["solver"],
            "selected_scope": {"selected_frame": values.get("case", {}).get("frame")},
            "shared_input_identities": _input_identities(case_report)}


def _artifact_assessment(solver: str, report: Mapping[str, Any], invalidated: set[str]) -> tuple[tuple[ArtifactReuseRecord, ...], tuple[str, ...]]:
    specs = stage_specs(solver); artifacts = report["artifacts"]
    by_stage: dict[str, list[Mapping[str, Any]]] = {stage: [] for stage in specs}
    for artifact in artifacts:
        if artifact.get("producer_stage") in by_stage: by_stage[artifact["producer_stage"]].append(artifact)
    missing_or_unusable: list[str] = []; records: list[ArtifactReuseRecord] = []
    for stage, (_deps, expected, _granularity) in specs.items():
        observed = by_stage[stage]
        if stage in invalidated:
            for artifact in observed or ({"artifact_type": item} for item in expected):
                records.append(ArtifactReuseRecord(artifact["artifact_type"], stage, artifact.get("location"), artifact.get("identity"),
                                                    "not_reusable", "Producer stage is invalidated by configuration or upstream dependency."))
            continue
        if not expected:
            continue
        # Inspection's legacy adapter records have only metadata identities and no
        # producer signature/effective-config/input identity, so they never prove reuse.
        if not observed:
            missing_or_unusable.append(stage)
            records.append(ArtifactReuseRecord(expected[0], stage, None, None, "missing", "Required producer output is not observed."))
        else:
            missing_or_unusable.append(stage)
            for artifact in observed:
                records.append(ArtifactReuseRecord(artifact["artifact_type"], stage, artifact.get("location"), artifact.get("identity"),
                                                    "candidate_unverified", "Legacy artifact lacks producer signature, effective-config identity, and compatible input proof."))
    return tuple(records), tuple(name for name in specs if name in set(missing_or_unusable))


def plan_trial(solver_config: str | Path, *, override: Mapping[str, Any] | None = None,
               case_key: str | None = None, case_paths: str | Path = "config/case_paths.yaml",
               case_root: str | Path | None = None, solver: str | None = None,
               trial_id: str | None = None, diagnosis: Mapping[str, Any] | None = None,
               restore_missing: bool = False, scope: Mapping[str, Any] | None = None) -> Envelope:
    """Describe a trial without creating files, importing native code, or executing a solver."""
    resolved = resolve_config(solver_config, case_key=case_key, case_paths=case_paths, solver=solver)
    canonical_solver, baseline_config = resolved["solver"], resolved["effective_config"]
    try:
        effective, changes = merge_sparse_override(baseline_config, override or {}, solver=canonical_solver)
    except ValueError as error:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Invalid sparse trial override", True,
                                            details={"reason": str(error)})) from error
    case = inspect_case(solver_config, case_key=case_key, case_paths=case_paths, case_root=case_root, solver=solver).data
    planning_scope = {**case["scope"], **dict(scope or {})}
    baseline = _baseline_identity(resolved, case)
    sparse_override = canonical_sparse_override(changes)
    trial = {"trial_id": trial_id, "parent": baseline["effective_config_identity"],
             "effective_config_identity": effective_config_identity(effective)}
    violations = list(protected_violations(canonical_solver, [item.path for item in changes]))
    protected_paths = {item.path for item in violations}
    direct: set[str] = set()
    for change in changes:
        if change.path in protected_paths:
            continue
        owners = owner_stages(canonical_solver, change.path)
        if owners is None:
            violations.append(PolicyViolationRecord(change.path, "TRIAL.UNKNOWN_OWNERSHIP", "No explicit scientific stage ownership is registered."))
        else: direct.update(owners)
    if violations:
        plan = TrialPlan(canonical_solver, planning_scope, baseline, trial, sparse_override, trial["effective_config_identity"], changes,
                         tuple(violations), (), (), (), (), (), (), "blocked",
                         warnings=("No execution plan is generated for a policy-blocked override.",))
        return Envelope(status="ok", operation="trial.plan", data={"trial_plan": plan.to_dict()})
    config_invalidated = downstream_closure(canonical_solver, direct)
    artifact_reuse, unavailable = _artifact_assessment(canonical_solver, case, set(config_invalidated))
    # A no-op trial means no scientific parameter change.  Missing-artifact
    # restoration is available only as an explicit planning request so that a
    # no-op never masquerades as an invalidating trial.
    expand_artifacts = bool(changes) or restore_missing
    unavailable = unavailable if expand_artifacts else ()
    artifact_closure = downstream_closure(canonical_solver, unavailable)
    minimum = tuple(name for name in stage_specs(canonical_solver) if name in set(config_invalidated) | set(artifact_closure))
    stage_records: list[StagePlanRecord] = []
    unavailable_set = set(unavailable)
    for stage in stage_specs(canonical_solver):
        if stage in config_invalidated:
            stage_records.append(StagePlanRecord(stage, "invalidated", ("config_change",), False, False))
        elif stage in artifact_closure:
            reason = "missing_artifact" if stage in unavailable_set else "upstream_dependency"
            # Its config/dependency semantics remain valid.  The rerun is an
            # execution-safety consequence of absent or unproven artifacts,
            # not a claim that this stage was scientifically invalidated.
            stage_records.append(StagePlanRecord(stage, "required_rerun", (reason,), True, False))
        else:
            stage_records.append(StagePlanRecord(stage, "scientifically_reusable", (), True, False))
    missing_inputs = case["readiness"].get("missing", [])
    warnings = ["All planned actions are dry-run descriptions; current adapters do not expose guarded stage execution."]
    if diagnosis:
        warnings.append("Diagnosis context is recorded only by the caller; it did not select or modify this override.")
    status = "ready" if changes and not missing_inputs else ("no_effect" if not changes and not restore_missing else "partial")
    plan = TrialPlan(canonical_solver, planning_scope, baseline, trial, sparse_override, trial["effective_config_identity"], changes, (),
                     config_invalidated, unavailable, tuple(stage_records), artifact_reuse, minimum,
                     execution_actions(canonical_solver, minimum), status, tuple(warnings))
    return Envelope(status="ok", operation="trial.plan", data={"trial_plan": plan.to_dict(), "diagnosis_context": dict(diagnosis or {})})
