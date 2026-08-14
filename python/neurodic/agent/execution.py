"""Guarded, allowlisted execution control plane (Loop 7).

This module owns trial-local mutation.  It does not import a scientific API;
real solver adapters remain deliberately unavailable until their output
redirection contracts are independently validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .artifacts import content_identity, require_path_within
from .config import action_config_projection, effective_config_identity
from .errors import ControlPlaneError, ErrorRecord
from .inspect import resolve_config
from .schemas import Envelope, canonical_json, utc_now
from .stages import stage_specs
from .execution_registry import capability_for
from .trials import plan_trial


EXECUTION_SCHEMA_VERSION = "neurodic.execution/v1"
_TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


@dataclass(frozen=True)
class ProducerSignature:
    stage_id: str
    implementation: Mapping[str, Any]
    stage_config_identity: str
    input_identities: Mapping[str, Any]
    scope: Mapping[str, Any]
    output_contract: str = "neurodic.managed-artifact/v1"

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self.to_dict(include_digest=False)).encode()).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {"stage_id": self.stage_id, "implementation": dict(self.implementation),
                 "stage_config_identity": self.stage_config_identity, "input_identities": dict(self.input_identities),
                 "scope": dict(self.scope), "output_contract": self.output_contract}
        if include_digest: value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class TrustedAction:
    """An allowlisted adapter with an explicit, stable producer identity."""
    action_id: str
    run: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Sequence[str | "ProducedArtifact"]]
    implementation_identity: str
    output_contract: str = "neurodic.managed-artifact/v1"
    input_identities: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None
    config_projection: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    output_paths: Sequence[str] | None = None
    # Some combined APIs emit conditional evaluation files and optional
    # visualization products.  A resolver keeps safe-reuse exact while
    # allowing the producer signature to determine the conditional set.
    output_paths_resolver: Callable[[ProducerSignature], Sequence[str]] | None = None


@dataclass(frozen=True)
class ProducedArtifact:
    """A validated adapter output with stable managed-artifact metadata."""
    path: str
    artifact_type: str
    schema: str


def _error(code: str, message: str, **details: Any) -> ControlPlaneError:
    return ControlPlaneError(ErrorRecord(code, message, True, details=details))


def _atomic_json(path: Path, value: Mapping[str, Any], root: Path) -> None:
    target = require_path_within(path, root)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _identity(path: Path) -> dict[str, Any]:
    return content_identity(path).to_dict()


def _plan_request(plan: Mapping[str, Any]) -> tuple[Path, Path, str, str | None, Mapping[str, Any]]:
    baseline = plan.get("baseline", {})
    try:
        return (Path(baseline["config_source"]), Path(baseline["case_paths_source"]), baseline["case_key"],
                plan.get("trial", {}).get("trial_id"), plan["override"])
    except (KeyError, TypeError) as error:
        raise _error("SCHEMA.INVALID", "Plan lacks required immutable baseline fields") from error


def _revalidate(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    if plan.get("schema_version") != "neurodic.trial_plan/v1":
        raise _error("SCHEMA.INVALID", "Unsupported trial plan schema")
    intent = plan.get("planning_intent", {"restore_missing": False})
    if not isinstance(intent, Mapping) or not isinstance(intent.get("restore_missing"), bool):
        raise _error("SCHEMA.INVALID", "Plan has invalid planning_intent")
    config, paths, case_key, trial_id, override = _plan_request(plan)
    revalidation_scope = dict(plan.get("scope", {}))
    # NDeF stores a planner-owned snapshot of its managed scientific inputs in
    # scope.  It is regenerated from the case and declared dependencies below;
    # passing it as user scope would incorrectly reject every valid revalidation.
    revalidation_scope.pop("ndef_surface_inputs", None)
    revalidation_scope.pop("ndef_roi_inputs", None)
    revalidation_scope.pop("ndef_precalculation_inputs", None)
    revalidation_scope.pop("ndef_deformation_inputs", None)
    if plan.get("solver") == "pin_multi" and revalidation_scope.get("selected_frame") == -1:
        # Legacy pair-ROI plans inherited the config sentinel; C1 explicitly
        # supplies a non-negative frame and therefore never takes this path.
        revalidation_scope.pop("selected_frame")
    recomputed = plan_trial(config, case_key=case_key, case_paths=paths, override=override, trial_id=trial_id,
                            scope=revalidation_scope, restore_missing=intent["restore_missing"],
                            upstream_dependencies=plan.get("upstream_dependencies", ())).to_dict()["data"]["trial_plan"]
    keys = ("plan_identity", "effective_config_identity", "baseline", "changes", "config_invalidated_stages",
            "minimum_rerun_stages", "execution_actions", "policy_violations", "plan_status", "planning_intent", "upstream_dependencies")
    if any(plan.get(key) != recomputed.get(key) for key in keys):
        raise _error("TRIAL.PLAN_STALE", "Plan does not match current baseline, policy, or derived planning state")
    if recomputed["plan_status"] != "ready":
        raise _error("CAPABILITY.UNSUPPORTED", "Only ready plans can be executed", plan_status=recomputed["plan_status"])
    return recomputed


def _implementation_identity() -> Mapping[str, Any]:
    """Record a deterministic source-revision policy without trusting a caller."""
    try:
        revision = subprocess.run(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[3],
                                  text=True, capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(("git", "status", "--porcelain"), cwd=Path(__file__).resolve().parents[3],
                                    text=True, capture_output=True, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unavailable", True
    return {"revision_policy": "neurodic.git-head-plus-dirty/v1", "git_revision": revision, "dirty": dirty}


def _stage_signature(plan: Mapping[str, Any], effective: Mapping[str, Any], action: TrustedAction,
                     covered_stages: Sequence[str]) -> ProducerSignature:
    projection = (action.config_projection(effective) if action.config_projection
                  else action_config_projection(plan["solver"], covered_stages, effective))
    config_id = "sha256:" + hashlib.sha256(canonical_json(projection).encode()).hexdigest()
    inputs = action.input_identities(plan, effective) if action.input_identities else {
        "shared_inputs": plan["baseline"].get("shared_input_identities", {}),
        "baseline_config": plan["baseline"]["effective_config_identity"],
    }
    dependencies = plan.get("upstream_dependencies", ())
    if dependencies:
        inputs = {**inputs, "upstream_dependencies": [{
            "dependency_id": item.get("dependency_id"), "producer_action_id": item.get("producer_action_id"),
            "producer_signature": item.get("producer_signature"), "scope": item.get("scope"),
            "required_artifacts": [{"relative_path": artifact.get("relative_path"), "identity": artifact.get("identity")}
                                   for artifact in item.get("required_artifacts", ())]
        } for item in dependencies]}
    return ProducerSignature(action.action_id, {"adapter": action.implementation_identity,
                                     "neurodic": _implementation_identity()}, config_id, inputs,
                             plan["scope"], action.output_contract)


def _verified_reuse(managed_root: Path, signature: ProducerSignature, action: TrustedAction | None = None) -> list[dict[str, Any]] | None:
    """Return a prior managed attempt only if its full provenance and bytes verify."""
    trials = managed_root.resolve() / "trials"
    if not trials.is_dir(): return None
    expected = signature.to_dict()
    for candidate in sorted(path for path in trials.iterdir() if path.is_dir()):
        manifest_path = candidate / "manifest.json"
        try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError): continue
        if not isinstance(manifest, Mapping) or manifest.get("trial_id") != candidate.name:
            continue
        attempts = manifest.get("stage_attempts", [])
        published = manifest.get("produced_artifacts", [])
        if not isinstance(attempts, Sequence) or not isinstance(published, Sequence):
            continue
        completed = [item for item in attempts if isinstance(item, Mapping)
                     and item.get("status") == "completed"
                     and item.get("action_id") == signature.stage_id
                     and item.get("producer_signature") == expected]
        if len(completed) != 1: continue
        attempt_id = completed[0].get("stage_attempt_id")
        artifacts = [item for item in published if isinstance(item, Mapping)
                     and item.get("stage_attempt_id") == attempt_id]
        # Never reuse a partially retagged attempt: every recorded output must
        # retain the exact producer action and scientific signature.
        if not artifacts or any(item.get("producer_action_id") != signature.stage_id
                                or item.get("producer_signature") != expected for item in artifacts):
            continue
        expected_paths = None
        if action is not None:
            if action.output_paths_resolver is not None:
                expected_paths = tuple(action.output_paths_resolver(signature))
            elif action.output_paths is not None:
                expected_paths = tuple(action.output_paths)
        if expected_paths is not None:
            prefix = f"artifacts/{signature.stage_id}/{attempt_id}/"
            actual_paths = {str(item.get("location", ""))[len(prefix):]
                            for item in artifacts if str(item.get("location", "")).startswith(prefix)}
            published_root = candidate / "artifacts" / signature.stage_id / str(attempt_id)
            filesystem_paths = {path.relative_to(published_root).as_posix() for path in published_root.rglob("*") if path.is_file()} if published_root.is_dir() else set()
            wildcard_prefixes = tuple(item[:-3] for item in expected_paths if item.endswith("/**"))
            exact_expected = {item for item in expected_paths if not item.endswith("/**")}
            allowed_actual = {item for item in actual_paths if any(item.startswith(prefix) for prefix in wildcard_prefixes)}
            allowed_filesystem = {item for item in filesystem_paths if any(item.startswith(prefix) for prefix in wildcard_prefixes)}
            actual_exact = actual_paths - allowed_actual
            filesystem_exact = filesystem_paths - allowed_filesystem
            if (actual_exact != exact_expected or filesystem_exact != exact_expected
                    or len(artifacts) != len(actual_paths)):
                continue
        verified: list[dict[str, Any]] = []
        for artifact in artifacts:
            try:
                location = require_path_within(candidate / artifact["location"], candidate, require_exists=True)
                if not location.is_file() or _identity(location) != artifact["identity"]: raise ValueError("identity mismatch")
            except (KeyError, OSError, ValueError):
                verified = []; break
            verified.append({**artifact, "reuse_source_trial": candidate.name})
        if verified: return verified
    return None


def _workspace(managed_root: Path, trial_id: str) -> Path:
    if not _TRIAL_ID.fullmatch(trial_id): raise _error("SCHEMA.INVALID", "Invalid trial_id")
    root = managed_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    trial = root / "trials" / trial_id
    if trial.exists(): raise _error("TRIAL.ROOT_EXISTS", "Trial workspace already exists", trial_id=trial_id)
    trial.mkdir(parents=True); (trial / "state").mkdir(); (trial / "staging").mkdir(); (trial / "artifacts").mkdir()
    return trial


def _resolve_dependencies(plan: Mapping[str, Any], managed_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve only explicit, producer-signed managed artifacts from the plan."""
    resolved: dict[str, dict[str, Any]] = {}
    for dependency in plan.get("upstream_dependencies", ()):
        try:
            dependency_id = dependency["dependency_id"]; trial_id = dependency["source_trial_id"]
            attempt_id = dependency["source_attempt_id"]; action_id = dependency["producer_action_id"]
            signature = dependency["producer_signature"]; expected_scope = dependency["scope"]
            required = dependency["required_artifacts"]
        except (KeyError, TypeError) as error:
            raise _error("DEPENDENCY.INVALID", "Managed dependency record is incomplete") from error
        if (not isinstance(dependency_id, str) or not dependency_id or not isinstance(trial_id, str) or not _TRIAL_ID.fullmatch(trial_id)
                or not isinstance(attempt_id, str) or not attempt_id or not isinstance(action_id, str) or not action_id
                or not isinstance(signature, Mapping) or not isinstance(expected_scope, Mapping)
                or not isinstance(required, Sequence) or isinstance(required, (str, bytes)) or dependency_id in resolved):
            raise _error("DEPENDENCY.INVALID", "Managed dependency record has invalid types")
        upstream_scope = signature.get("scope")
        legacy_stage_alias = (action_id == "pin_multi.separate_pair_roi_call" and signature.get("stage_id") == "pin_multi.pair_roi")
        # Historical managed NDeF surface producers may retain the public
        # case.frame=-1 sentinel.  The deformation combined action resolves
        # that frame to an explicit non-negative image index before signing;
        # permit only this exact D→F compatibility while preserving every
        # other dependency-scope equality check.
        legacy_ndef_frame_alias = (action_id in {"ndef.combined_surface_call", "ndef.precalculation_call", "ndef.roi.generate_call"}
                                   and plan.get("execution_actions", [{}])[0].get("action_id") == "ndef.deformation_combined_call"
                                   and isinstance(upstream_scope, Mapping)
                                   and upstream_scope.get("selected_frame") == -1
                                   and plan.get("scope", {}).get("selected_frame") is not None
                                   and plan.get("scope", {}).get("selected_frame") >= 0)
        # A downstream dependency scope is the relevant projection of the
        # complete, manifest-bound producer signature.  Historical pair-ROI
        # signatures carry the legacy selected_frame=-1 sentinel, whereas C1
        # binds its actual solve frame separately; requiring equality here
        # would reject an otherwise exact, verified historical producer.
        if ((signature.get("stage_id") != action_id and not legacy_stage_alias) or not isinstance(upstream_scope, Mapping)
                or any(upstream_scope.get(key) != value for key, value in expected_scope.items()
                       if not (legacy_ndef_frame_alias and key == "selected_frame"))):
            raise _error("DEPENDENCY.PRODUCER_MISMATCH", "Dependency producer signature does not match its declared source")
        for key, value in expected_scope.items():
            if legacy_ndef_frame_alias and key == "selected_frame":
                continue
            if key in plan.get("scope", {}) and plan["scope"][key] != value:
                raise _error("DEPENDENCY.SCOPE_MISMATCH", "Dependency scope conflicts with downstream scope", dependency_id=dependency_id, key=key)
        source_trial = require_path_within(managed_root.resolve() / "trials" / trial_id, managed_root.resolve(), require_exists=True)
        try: manifest = json.loads((source_trial / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error: raise _error("DEPENDENCY.INVALID", "Dependency source manifest is unreadable") from error
        if not isinstance(manifest, Mapping) or manifest.get("trial_id") != trial_id:
            raise _error("DEPENDENCY.PRODUCER_MISMATCH", "Dependency source manifest does not belong to the declared trial")
        attempts = manifest.get("stage_attempts", [])
        if not isinstance(attempts, Sequence) or not any(isinstance(item, Mapping)
                                                          and item.get("stage_attempt_id") == attempt_id
                                                          and item.get("status") == "completed"
                                                          and item.get("action_id") == action_id
                                                          and item.get("producer_signature") == signature
                                                          for item in attempts):
            raise _error("DEPENDENCY.PRODUCER_MISMATCH", "Dependency source attempt is not a completed declared producer attempt")
        indexed = {item.get("location"): item for item in manifest.get("produced_artifacts", [])
                   if isinstance(item, Mapping) and item.get("stage_attempt_id") == attempt_id
                   and (item.get("producer_action_id", item.get("producer_signature", {}).get("stage_id")) == action_id
                        or (legacy_stage_alias and item.get("producer_signature", {}).get("stage_id") == signature.get("stage_id")))
                   and item.get("producer_signature") == signature}
        files: dict[str, str] = {}
        for artifact in required:
            if not isinstance(artifact, Mapping): raise _error("DEPENDENCY.INVALID", "Dependency required artifact is invalid")
            relative, identity = artifact.get("relative_path"), artifact.get("identity")
            if not isinstance(relative, str) or not isinstance(identity, Mapping): raise _error("DEPENDENCY.INVALID", "Dependency artifact lacks path or identity")
            # Producer declarations use their stable, staging-relative output
            # paths (for example ``roi/per_camera/cam_0_mask.npy``), while
            # publication places those files beneath the deterministic action
            # and attempt namespace.  Resolve only those two exact spellings;
            # this is deliberately not a search/fallback mechanism.
            published_relative = f"artifacts/{action_id}/{attempt_id}/{relative}"
            record = indexed.get(relative) or indexed.get(published_relative)
            if record is None or record.get("identity") != identity:
                raise _error("DEPENDENCY.PRODUCER_MISMATCH", "Dependency artifact is not published by the declared producer", dependency_id=dependency_id)
            location = require_path_within(source_trial / (published_relative if record.get("location") == published_relative else relative),
                                           source_trial, require_exists=True)
            if not location.is_file() or _identity(location) != identity:
                raise _error("DEPENDENCY.CONTENT_MISMATCH", "Dependency artifact content no longer matches the approved plan", dependency_id=dependency_id, path=relative)
            files[Path(relative).name] = str(location)
        resolved[dependency_id] = {"files": files, "producer_signature": signature, "scope": dict(expected_scope),
                                   "source_trial_id": trial_id, "source_attempt_id": attempt_id}
    return resolved


def _normalized_outputs(outputs: Sequence[str | ProducedArtifact]) -> list[ProducedArtifact]:
    return [item if isinstance(item, ProducedArtifact)
            else ProducedArtifact(str(item), Path(str(item)).stem, "unknown/v1")
            for item in outputs]


def _publish(staging: Path, trial: Path, artifact_namespace: str, attempt_id: str, outputs: Sequence[str | ProducedArtifact], signature: ProducerSignature) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []
    destination = trial / "artifacts" / artifact_namespace / attempt_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise _error("EXECUTION.PUBLISH_FAILED", "Attempt output location already exists")
    normalized = _normalized_outputs(outputs)
    for item in normalized:
        source = require_path_within(staging / item.path, staging, require_exists=True)
        if not source.is_file() or source.stat().st_size == 0: raise _error("EXECUTION.ARTIFACT_INVALID", "Output must be a non-empty regular file", path=str(source))
    os.replace(staging, destination)
    for item in normalized:
        artifact = destination / item.path
        published.append({"artifact_type": item.artifact_type, "location": str(artifact.relative_to(trial)), "producer_action_id": signature.stage_id,
                          "identity": _identity(artifact), "producer_signature": signature.to_dict(),
                          "stage_attempt_id": attempt_id, "schema": item.schema, "size_bytes": artifact.stat().st_size})
    return published


def execute_trial(plan: Mapping[str, Any], *, managed_root: str | Path,
                  trusted_actions: Mapping[str, TrustedAction] | None = None,
                  action_id: str | None = None) -> Envelope:
    """Revalidate and execute one approved plan through an explicit allowlist.

    Production callers receive `EXECUTION.UNSUPPORTED` until a real adapter is
    registered after its output isolation contract has been verified. Tests may
    inject a `TrustedAction`, exercising the exact workspace/publish path.
    """
    approved = _revalidate(plan)
    trial_id = approved["trial"]["trial_id"]
    if not trial_id: raise _error("SCHEMA.INVALID", "Execution requires an explicit trial_id")
    default_registry = trusted_actions is None
    if default_registry:
        # Lazy import preserves the native-free boundary for planning and only
        # exposes adapters whose output contract has been audited.
        from .adapters.execution_pin import guarded_pin_action
        from .adapters.execution_stereo import guarded_stereo_action
        from .adapters.execution_pin_multi import guarded_pair_roi_action, guarded_pair_solve_quality_action, guarded_fusion_postprocess_action
        from .adapters.execution_ndef import guarded_ndef_surface_action
        from .adapters.execution_ndef_roi import guarded_ndef_roi_action
        from .adapters.execution_ndef_precalculation import guarded_ndef_precalculation_action
        from .adapters.execution_ndef_deformation import guarded_ndef_deformation_action
        pin = guarded_pin_action()
        stereo = guarded_stereo_action()
        pair_roi = guarded_pair_roi_action()
        pair_solve_quality = guarded_pair_solve_quality_action()
        fusion_postprocess = guarded_fusion_postprocess_action()
        ndef_surface = guarded_ndef_surface_action()
        ndef_roi = guarded_ndef_roi_action()
        ndef_precalculation = guarded_ndef_precalculation_action()
        ndef_deformation = guarded_ndef_deformation_action()
        allowed = {pin.action_id: pin, stereo.action_id: stereo, pair_roi.action_id: pair_roi,
                   pair_solve_quality.action_id: pair_solve_quality, fusion_postprocess.action_id: fusion_postprocess,
                   ndef_surface.action_id: ndef_surface, ndef_roi.action_id: ndef_roi}
        allowed[ndef_precalculation.action_id] = ndef_precalculation
        allowed[ndef_deformation.action_id] = ndef_deformation
    else:
        allowed = trusted_actions
    actions = approved["execution_actions"]
    selected = [item for item in actions if item["action_id"] == action_id] if action_id else actions
    if len(selected) != 1 or (default_registry and not capability_for(selected[0]["action_id"]).execution_supported) or selected[0]["action_id"] not in allowed:
        raise _error("EXECUTION.UNSUPPORTED", "No verified guarded adapter is registered for this approved action",
                     actions=[item["action_id"] for item in actions], requested_action=action_id)
    config_path, paths_path, case_key, _unused, _override = _plan_request(approved)
    effective = resolve_config(config_path, case_key=case_key, case_paths=paths_path)["effective_config"]
    # Reapply the frozen sparse override via planner recomputation rather than
    # trusting a serialized effective config supplied by the caller.
    from .config import merge_sparse_override
    effective, _changes = merge_sparse_override(effective, approved["override"], solver=approved["solver"])
    if effective_config_identity(effective) != approved["effective_config_identity"]:
        raise _error("TRIAL.PLAN_STALE", "Effective configuration identity changed before execution")
    if selected[0]["action_id"] == "pin_multi.fusion_postprocess_call":
        # A serialized readiness string is not authority for a mutating C3
        # action. Recompute the managed C2 report immediately before resolving
        # dependencies, so changed/missing pair products fail closed.
        from .pair_set_readiness import inspect_pin_multi_pair_set_readiness
        readiness = inspect_pin_multi_pair_set_readiness(config_path, case_key=case_key, case_paths=paths_path,
                                                          managed_root=managed_root, selected_frame=approved["scope"].get("selected_frame")).data
        if (readiness.get("status") != "ready" or readiness.get("planned_pair_set_identity") != approved["scope"].get("planned_pair_set_identity")
                or readiness.get("fusion_input_identity") != approved["scope"].get("fusion_input_identity")
                or readiness.get("scope", {}).get("planned_pair_ids") != approved["scope"].get("planned_pair_ids")):
            raise _error("DEPENDENCY.INVALID", "C3 pair-set readiness or fusion input identity is stale")
    dependencies = _resolve_dependencies(approved, Path(managed_root))
    trial = _workspace(Path(managed_root), trial_id)
    manifest = {"schema_version": EXECUTION_SCHEMA_VERSION, "trial_id": trial_id, "plan_identity": approved["plan_identity"],
                "execution_status": "prepared", "baseline": approved["baseline"], "override": approved["override"],
                "effective_config_identity": approved["effective_config_identity"], "stage_attempts": [], "produced_artifacts": [],
                "reused_artifacts": [], "baseline_writes": []}
    _atomic_json(trial / "manifest.json", manifest, trial)
    _atomic_json(trial / "override.json", approved["override"], trial)
    _atomic_json(trial / "effective_config.json", effective, trial)
    action = allowed[selected[0]["action_id"]]
    attempt_id = "attempt_" + hashlib.sha256((approved["plan_identity"] + utc_now()).encode()).hexdigest()[:16]
    stage = selected[0]["covers_stages"][-1]
    signature = _stage_signature(approved, effective, action, selected[0]["covers_stages"])
    reuse = _verified_reuse(Path(managed_root), signature, action)
    if reuse:
        attempt = {"stage_attempt_id": attempt_id, "action_id": action.action_id, "stage_id": stage, "status": "reused",
                   "started_at": utc_now(), "finished_at": utc_now(), "producer_signature": signature.to_dict(),
                   "reused_from": {"trial_id": reuse[0]["reuse_source_trial"], "stage_attempt_id": reuse[0]["stage_attempt_id"]}}
        manifest["stage_attempts"].append(attempt); manifest["reused_artifacts"].extend(reuse)
        manifest["execution_status"] = "completed" if len(selected) == len(actions) else "partial"
        _atomic_json(trial / "manifest.json", manifest, trial)
        return Envelope(status="ok", operation="trial.execute", data={"execution": manifest})
    staging = trial / "staging" / attempt_id; staging.mkdir()
    attempt = {"stage_attempt_id": attempt_id, "action_id": action.action_id, "stage_id": stage, "status": "running",
               "started_at": utc_now(), "staging_root": str(staging.relative_to(trial)), "producer_signature": signature.to_dict()}
    manifest["execution_status"] = "running"; manifest["stage_attempts"].append(attempt); _atomic_json(trial / "manifest.json", manifest, trial)
    try:
        runtime_scope = {**approved["scope"], "_managed_dependencies": dependencies,
                         "_planned_dependencies": approved.get("upstream_dependencies", ())}
        # The sidecar records the exact signature that is about to publish;
        # this is passed only through the adapter's private runtime scope.
        runtime_scope["_producer_signature"] = signature.to_dict()
        outputs = list(action.run(effective, staging, runtime_scope))
        if not outputs: raise _error("EXECUTION.ARTIFACT_INVALID", "Trusted action reported no outputs")
        published = _publish(staging, trial, action.action_id, attempt_id, outputs, signature)
    except KeyboardInterrupt:
        attempt.update({"status": "interrupted", "finished_at": utc_now()}); manifest["execution_status"] = "interrupted"
        _atomic_json(trial / "manifest.json", manifest, trial)
        return Envelope(status="warning", operation="trial.execute", data={"execution": manifest}, warnings=({"code": "EXECUTION.INTERRUPTED"},))
    except Exception as error:
        attempt.update({"status": "failed", "finished_at": utc_now(), "error": str(error)}); manifest["execution_status"] = "failed"
        _atomic_json(trial / "manifest.json", manifest, trial)
        code = error.record.code if isinstance(error, ControlPlaneError) else "EXECUTION.FAILED"
        return Envelope(status="warning", operation="trial.execute", data={"execution": manifest}, warnings=({"code": code},))
    attempt.update({"status": "completed", "finished_at": utc_now(), "published_artifacts": published})
    manifest["produced_artifacts"].extend(published)
    manifest["execution_status"] = "completed" if len(selected) == len(actions) else "partial"
    _atomic_json(trial / "manifest.json", manifest, trial)
    return Envelope(status="ok", operation="trial.execute", data={"execution": manifest})
