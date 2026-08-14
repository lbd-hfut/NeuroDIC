"""Native-free, read-only trial/config dry-run planner (Loop 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    upstream_dependencies: Sequence[Mapping[str, Any]] = ()
    schema_version: str = TRIAL_PLAN_SCHEMA_VERSION
    planning_intent: Mapping[str, Any] = field(default_factory=lambda: {"restore_missing": False})

    def __post_init__(self) -> None:
        if not isinstance(self.planning_intent, Mapping) or not isinstance(self.planning_intent.get("restore_missing"), bool):
            raise ValueError("TrialPlan planning_intent.restore_missing must be boolean")

    @property
    def plan_identity(self) -> str:
        import hashlib
        core = {"baseline": self.baseline, "trial": self.trial, "solver": self.solver, "scope": self.scope,
                "override": self.override, "effective_config_identity": self.effective_config_identity,
                "planning_intent": dict(self.planning_intent),
                "changes": [item.to_dict() for item in self.changes],
                "config_invalidated_stages": list(self.config_invalidated_stages),
                "artifact_missing_or_unusable_stages": list(self.artifact_missing_or_unusable_stages),
                "minimum_rerun_stages": list(self.minimum_rerun_stages),
                "execution_actions": [item.to_dict() for item in self.execution_actions],
                "upstream_dependencies": [dict(item) for item in self.upstream_dependencies]}
        return "sha256:" + hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "plan_identity": self.plan_identity, "plan_status": self.plan_status, "dry_run": self.dry_run,
                "execution_performed": self.execution_performed, "solver": self.solver, "scope": dict(self.scope),
                "baseline": dict(self.baseline), "trial": dict(self.trial), "override": dict(self.override),
                "planning_intent": dict(self.planning_intent),
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
                "would_write": [], "upstream_dependencies": [dict(item) for item in self.upstream_dependencies]}


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


def _validate_scope(solver: str, scope: Mapping[str, Any], case: Mapping[str, Any], *, action_required: bool) -> PolicyViolationRecord | None:
    """Validate solver-owned scope before a plan can become executable."""
    if solver == "pin_multi":
        # Pair ROI legacy plans are retained for compatibility.  The C1
        # adapter itself requires both explicit scope fields before execution.
        return None
    if solver not in {"pin", "pin_stereo"}:
        return None
    selected = scope.get("selected_frame")
    if selected is None:
        if action_required:
            return PolicyViolationRecord("scope.selected_frame", "TRIAL.SCOPE_REQUIRED",
                                         f"{solver} guarded execution requires an explicit selected_frame in the TrialPlan scope")
        return None
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 0:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", f"Invalid {solver} selected_frame scope", True,
                                            details={"selected_frame": selected, "reason": "must be a non-negative integer"}))
    count = case.get("frames", {}).get("count")
    if not isinstance(count, int) or selected >= count:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", f"{solver} selected_frame is outside the resolved case", True,
                                            details={"selected_frame": selected, "frame_count": count}))
    return None


def plan_trial(solver_config: str | Path, *, override: Mapping[str, Any] | None = None,
               case_key: str | None = None, case_paths: str | Path = "config/case_paths.yaml",
               case_root: str | Path | None = None, solver: str | None = None,
               trial_id: str | None = None, diagnosis: Mapping[str, Any] | None = None,
               restore_missing: bool = False, scope: Mapping[str, Any] | None = None,
               upstream_dependencies: Sequence[Mapping[str, Any]] | None = None) -> Envelope:
    """Describe a trial without creating files, importing native code, or executing a solver."""
    resolved = resolve_config(solver_config, case_key=case_key, case_paths=case_paths, solver=solver)
    canonical_solver, baseline_config = resolved["solver"], resolved["effective_config"]
    try:
        effective, changes = merge_sparse_override(baseline_config, override or {}, solver=canonical_solver)
    except ValueError as error:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Invalid sparse trial override", True,
                                            details={"reason": str(error)})) from error
    case = inspect_case(solver_config, case_key=case_key, case_paths=case_paths, case_root=case_root, solver=solver).data
    if scope is not None and not isinstance(scope, Mapping):
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Trial scope must be a JSON mapping", True))
    if upstream_dependencies is not None and (not isinstance(upstream_dependencies, Sequence) or isinstance(upstream_dependencies, (str, bytes)) or not all(isinstance(item, Mapping) for item in upstream_dependencies)):
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "upstream_dependencies must be a JSON list of mappings", True))
    planning_scope = {**case["scope"], **dict(scope or {})}
    if canonical_solver == "pin_stereo" and (scope is None or "selected_frame" not in scope):
        # The legacy wrapper's -1 default is not an approved guarded scope.
        planning_scope["selected_frame"] = None
    if canonical_solver == "pin_multi":
        pair_id, frame = planning_scope.get("pair_id"), planning_scope.get("selected_frame")
        if pair_id is not None and (not isinstance(pair_id, str) or "__" not in pair_id):
            raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Invalid PIN Multi pair_id scope", True))
        if scope is not None and "selected_frame" in scope and (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or frame >= case["frames"].get("count", 0)):
            raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Invalid PIN Multi selected_frame scope", True))
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
                         warnings=("No execution plan is generated for a policy-blocked override.",),
                         planning_intent={"restore_missing": bool(restore_missing)}, upstream_dependencies=tuple(upstream_dependencies or ()))
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
    actions = execution_actions(canonical_solver, minimum)
    c3_requested = canonical_solver == "pin_multi" and "planned_pair_ids" in planning_scope
    if c3_requested:
        pairs = planning_scope.get("planned_pair_ids")
        required_scope = (planning_scope.get("pair_set_status") == "ready" and isinstance(planning_scope.get("fusion_input_identity"), str)
                          and isinstance(planning_scope.get("planned_pair_set_identity"), str) and isinstance(pairs, list)
                          and pairs and len(set(pairs)) == len(pairs) and all(isinstance(item, str) and "__" in item for item in pairs))
        expected_ids = {f"pair/{pair}" for pair in pairs} if required_scope else set()
        actual_ids = {item.get("dependency_id") for item in (upstream_dependencies or ()) if isinstance(item, Mapping)}
        if not required_scope or actual_ids != expected_ids:
            violations.append(PolicyViolationRecord("scope.planned_pair_ids", "PAIR_SET.NOT_READY",
                                                     "C3 fusion requires a C2-ready complete ordered managed pair set."))
        else:
            actions = (ExecutionActionRecord("pin_multi.fusion_postprocess_call", "pin_multi",
                                             ("pin_multi.fusion", "pin_multi.postprocess")),)
            minimum = ("pin_multi.fusion", "pin_multi.postprocess")
    ndef_roi_only = canonical_solver == "ndef" and planning_scope.get("ndef_roi_only") is True
    ndef_precalculation_only = canonical_solver == "ndef" and planning_scope.get("ndef_precalculation_only") is True
    ndef_deformation_only = canonical_solver == "ndef" and planning_scope.get("ndef_deformation_only") is True
    if ndef_precalculation_only:
        # Sparse pre-calculation is an independently authorized public call.
        # It may consume only completed managed ROI and surface producers; the
        # case inspector's configured/legacy downstream paths never authorize
        # a fallback execution.
        from .adapters.execution_ndef_precalculation import ACTION_ID as PRECALC_ACTION_ID, INPUTS_KEY as PRECALC_INPUTS_KEY, managed_precalculation_inputs, precalculation_readiness
        if scope is not None and PRECALC_INPUTS_KEY in scope:
            violations.append(PolicyViolationRecord(f"scope.{PRECALC_INPUTS_KEY}", "SCHEMA.INVALID",
                                                     "NDeF managed precalculation input snapshots are planner-owned"))
        readiness = precalculation_readiness(effective, tuple(upstream_dependencies or ()))
        violations.extend(PolicyViolationRecord("ndef.precalculation", code, message) for code, message in readiness)
        if not readiness and not any(item.path == f"scope.{PRECALC_INPUTS_KEY}" for item in violations):
            try:
                planning_scope[PRECALC_INPUTS_KEY] = managed_precalculation_inputs(
                    {"upstream_dependencies": tuple(upstream_dependencies or ())}, effective)
            except (OSError, ValueError, ControlPlaneError) as error:
                violations.append(PolicyViolationRecord("ndef.precalculation", "NDEF.PRECALCULATION_INPUTS_NOT_READY", str(error)))
        actions = (ExecutionActionRecord(PRECALC_ACTION_ID, "ndef", ("ndef.precalculation",)),)
        minimum = ("ndef.precalculation",)
        if any(isinstance(item, Mapping) and item.get("dependency_id") in {"ndef_roi", "ndef_surface"}
               for item in (upstream_dependencies or ())):
            missing_inputs = [item for item in missing_inputs if item.get("stage") == "ndef.inputs"]
    elif ndef_roi_only:
        # ROI is an atomic public preprocessing operation.  Keep its plan
        # independent from the later surface action so a real ROI generation
        # can be approved and published without authorizing surface science.
        from .adapters.execution_ndef_roi import ACTION_ID as ROI_ACTION_ID, INPUTS_KEY as ROI_INPUTS_KEY, managed_roi_inputs, roi_readiness
        if scope is not None and ROI_INPUTS_KEY in scope:
            violations.append(PolicyViolationRecord(f"scope.{ROI_INPUTS_KEY}", "SCHEMA.INVALID",
                                                     "NDeF managed ROI input snapshots are planner-owned"))
        violations.extend(PolicyViolationRecord("ndef.roi", code, message)
                          for code, message in roi_readiness(effective))
        if not any(item.path == "scope." + ROI_INPUTS_KEY for item in violations):
            try:
                planning_scope[ROI_INPUTS_KEY] = managed_roi_inputs(effective)
            except (OSError, ValueError):
                pass
        actions = (ExecutionActionRecord(ROI_ACTION_ID, "ndef",
                                         ("ndef.roi",)),)
        minimum = ("ndef.roi",)
        missing_inputs = [item for item in missing_inputs if item.get("stage") == "ndef.inputs"]
    elif ndef_deformation_only:
        # Deformation is one public combined call.  D, E, ROI, calibration,
        # images, and the explicit model/training contract are all frozen in a
        # single planner-owned snapshot; no legacy configured paths may make
        # this branch appear partially ready.
        from .adapters.execution_ndef_deformation import ACTION_ID as DEFORMATION_ACTION_ID, INPUTS_KEY as DEFORMATION_INPUTS_KEY, managed_deformation_inputs, deformation_readiness
        if scope is not None and DEFORMATION_INPUTS_KEY in scope:
            violations.append(PolicyViolationRecord(f"scope.{DEFORMATION_INPUTS_KEY}", "SCHEMA.INVALID",
                                                     "NDeF managed deformation input snapshots are planner-owned"))
        readiness = deformation_readiness(effective, tuple(upstream_dependencies or ()))
        violations.extend(PolicyViolationRecord("ndef.deformation", code, message) for code, message in readiness)
        if not readiness and not any(item.path == f"scope.{DEFORMATION_INPUTS_KEY}" for item in violations):
            try:
                planning_scope[DEFORMATION_INPUTS_KEY] = managed_deformation_inputs(
                    {"upstream_dependencies": tuple(upstream_dependencies or ())}, effective)
                # Bind the resolved non-negative frame in the public plan
                # scope as well as inside the managed input snapshot.  The
                # producer signature must never carry the legacy -1 sentinel.
                planning_scope["selected_frame"] = planning_scope[DEFORMATION_INPUTS_KEY]["images"]["resolved_index"]
            except (OSError, ValueError, ControlPlaneError) as error:
                violations.append(PolicyViolationRecord("ndef.deformation", "NDEF.DEFORMATION_INPUTS_NOT_READY", str(error)))
        actions = (ExecutionActionRecord(DEFORMATION_ACTION_ID, "ndef",
                                         ("ndef.deformation.train", "ndef.deformation.infer", "ndef.postprocess", "ndef.evaluate")),)
        minimum = ("ndef.deformation.train", "ndef.deformation.infer", "ndef.postprocess", "ndef.evaluate")
        dependency_ids = {item.get("dependency_id") for item in (upstream_dependencies or ()) if isinstance(item, Mapping)}
        if {"ndef_surface", "ndef_precalculation", "ndef_roi"}.issubset(dependency_ids):
            missing_inputs = [item for item in missing_inputs if item.get("stage") == "ndef.inputs"]
    elif canonical_solver == "ndef" and any(
            isinstance(item, Mapping) and item.get("dependency_id") == "ndef_roi"
            for item in (upstream_dependencies or ())):
        # A managed ROI producer is the approved source for the surface action;
        # the case inspector's configured ROI/surface/precalculation paths are
        # not allowed to downgrade this future surface plan to partial.
        missing_inputs = [item for item in missing_inputs if item.get("stage") == "ndef.inputs"]
    # Apply managed-surface readiness only when this plan actually requests a
    # surface recomputation: a surface-owned config change, or an explicit
    # restore-missing request.  A downstream deformation tuning plan can carry
    # legacy unavailable artifacts in its descriptive closure without gaining
    # authority to run (or being blocked by) the surface action.
    ndef_surface_requested = "ndef.surface" in config_invalidated or (restore_missing and not changes)
    if (canonical_solver == "ndef" and ndef_surface_requested
            and any(item.action_id == "ndef.combined_surface_call" for item in actions)):
        # NDeF surface execution has one combined public API but several inputs
        # that the legacy wrapper otherwise rediscovers.  Freeze the native-free
        # content contract in the plan so a later case-input mutation makes the
        # plan stale before the trusted adapter imports the scientific API.
        from .adapters.execution_ndef import INPUTS_KEY, managed_surface_inputs, surface_readiness
        if scope is not None and INPUTS_KEY in scope:
            violations.append(PolicyViolationRecord(f"scope.{INPUTS_KEY}", "SCHEMA.INVALID",
                                                     "NDeF managed input snapshots are planner-owned"))
        readiness = surface_readiness(effective, tuple(upstream_dependencies or ()))
        violations.extend(PolicyViolationRecord("ndef.surface", code, message) for code, message in readiness)
        if not readiness and not any(item.code == "SCHEMA.INVALID" for item in violations):
            try:
                planning_scope[INPUTS_KEY] = managed_surface_inputs({"upstream_dependencies": tuple(upstream_dependencies or ())}, effective)
            except ValueError as error:
                code = str(error) if str(error).startswith("NDEF.") else "NDEF.CALIBRATION_NOT_MANAGED"
                violations.append(PolicyViolationRecord("ndef.surface", code, str(error)))
    scope_violation = _validate_scope(canonical_solver, planning_scope, case, action_required=bool(actions))
    if scope_violation:
        violations.append(scope_violation)
    if violations:
        plan = TrialPlan(canonical_solver, planning_scope, baseline, trial, sparse_override, trial["effective_config_identity"], changes,
                         tuple(violations), config_invalidated, unavailable, tuple(stage_records), artifact_reuse, minimum,
                         actions, "blocked", warnings=("No execution plan is generated until required scope is explicit and resolvable.",),
                         planning_intent={"restore_missing": bool(restore_missing)}, upstream_dependencies=tuple(upstream_dependencies or ()))
        return Envelope(status="ok", operation="trial.plan", data={"trial_plan": plan.to_dict(), "diagnosis_context": dict(diagnosis or {})})
    warnings = ["All planned actions are dry-run descriptions; guarded execution requires an approved ready plan."]
    if diagnosis:
        warnings.append("Diagnosis context is recorded only by the caller; it did not select or modify this override.")
    has_execution_intent = bool(changes) or (restore_missing and bool(actions))
    if not has_execution_intent:
        status = "no_effect"
    elif missing_inputs:
        status = "partial"
    else:
        status = "ready"
    plan = TrialPlan(canonical_solver, planning_scope, baseline, trial, sparse_override, trial["effective_config_identity"], changes, (),
                     config_invalidated, unavailable, tuple(stage_records), artifact_reuse, minimum,
                     actions, status, tuple(warnings), planning_intent={"restore_missing": bool(restore_missing)}, upstream_dependencies=tuple(upstream_dependencies or ()))
    return Envelope(status="ok", operation="trial.plan", data={"trial_plan": plan.to_dict(), "diagnosis_context": dict(diagnosis or {})})
