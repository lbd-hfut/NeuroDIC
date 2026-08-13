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
from .config import effective_config_identity, stage_config_projection
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
    run: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Sequence[str]]
    implementation_identity: str
    output_contract: str = "neurodic.managed-artifact/v1"
    input_identities: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None


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
    config, paths, case_key, trial_id, override = _plan_request(plan)
    recomputed = plan_trial(config, case_key=case_key, case_paths=paths, override=override, trial_id=trial_id,
                            scope=plan.get("scope", {})).to_dict()["data"]["trial_plan"]
    keys = ("plan_identity", "effective_config_identity", "baseline", "changes", "config_invalidated_stages",
            "minimum_rerun_stages", "execution_actions", "policy_violations", "plan_status")
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


def _stage_signature(plan: Mapping[str, Any], effective: Mapping[str, Any], stage: str,
                     action: TrustedAction) -> ProducerSignature:
    projection = stage_config_projection(plan["solver"], stage, effective)
    config_id = "sha256:" + hashlib.sha256(canonical_json(projection).encode()).hexdigest()
    inputs = action.input_identities(plan, effective) if action.input_identities else {
        "shared_inputs": plan["baseline"].get("shared_input_identities", {}),
        "baseline_config": plan["baseline"]["effective_config_identity"],
    }
    return ProducerSignature(stage, {"adapter": action.implementation_identity,
                                     "neurodic": _implementation_identity()}, config_id, inputs,
                             plan["scope"], action.output_contract)


def _verified_reuse(managed_root: Path, signature: ProducerSignature) -> list[dict[str, Any]] | None:
    """Return a prior managed attempt only if its full provenance and bytes verify."""
    trials = managed_root.resolve() / "trials"
    if not trials.is_dir(): return None
    expected = signature.to_dict()
    for candidate in sorted(path for path in trials.iterdir() if path.is_dir()):
        manifest_path = candidate / "manifest.json"
        try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError): continue
        artifacts = [item for item in manifest.get("produced_artifacts", [])
                     if item.get("producer_signature") == expected]
        if not artifacts: continue
        attempt_ids = {item.get("stage_attempt_id") for item in artifacts}
        if len(attempt_ids) != 1: continue
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


def _publish(staging: Path, trial: Path, stage: str, attempt_id: str, outputs: Sequence[str], signature: ProducerSignature) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []
    destination = trial / "artifacts" / stage / attempt_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise _error("EXECUTION.PUBLISH_FAILED", "Attempt output location already exists")
    for item in outputs:
        source = require_path_within(staging / item, staging, require_exists=True)
        if not source.is_file() or source.stat().st_size == 0: raise _error("EXECUTION.ARTIFACT_INVALID", "Output must be a non-empty regular file", path=str(source))
    os.replace(staging, destination)
    for item in outputs:
        artifact = destination / item
        published.append({"artifact_type": Path(item).stem, "location": str(artifact.relative_to(trial)),
                          "identity": _identity(artifact), "producer_signature": signature.to_dict(),
                          "stage_attempt_id": attempt_id, "schema": "unknown/v1", "size_bytes": artifact.stat().st_size})
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
        from .adapters.execution_pin_multi import guarded_pair_roi_action
        pair_roi = guarded_pair_roi_action()
        allowed = {pair_roi.action_id: pair_roi}
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
    signature = _stage_signature(approved, effective, stage, action)
    reuse = _verified_reuse(Path(managed_root), signature)
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
        outputs = list(action.run(effective, staging, approved["scope"]))
        if not outputs: raise _error("EXECUTION.ARTIFACT_INVALID", "Trusted action reported no outputs")
        published = _publish(staging, trial, stage, attempt_id, outputs, signature)
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
