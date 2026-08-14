"""Native-free, zero-write readiness inspection for managed PIN Multi C1 results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..pin_multi_roi import _options_from_config, pair_id_for, select_pin_multi_pairs
from .adapters.execution_pin_multi import _solve_outputs, validate_pair_solve_quality_outputs
from .artifacts import content_identity
from .inspect import resolve_config
from .schemas import Envelope, canonical_json


SCHEMA = "neurodic.pin_multi.pair-set-readiness/v1"
ACTION = "pin_multi.pair_solve_quality_call"
IMPLEMENTATION = "neurodic.pin_multi.pair_solve_quality/v1"
CONTRACT = "neurodic.pin_multi.pair-solve-quality-artifacts/v1"
FUSION_TYPES = frozenset({"pin_multi_reconstruction.reference", "pin_multi_reconstruction.current"})


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity(value: Any) -> Any:
    """Remove management-only fields from a producer identity projection."""
    if isinstance(value, Mapping):
        return {str(key): _identity(item) for key, item in value.items()
                if key not in {"trial_id", "attempt_id", "stage_attempt_id", "location", "path", "managed_root",
                               "staging_root", "publish_path", "started_at", "finished_at", "request_id"}}
    if isinstance(value, list): return [_identity(item) for item in value]
    return value


def _selection_identity(values: Mapping[str, Any]) -> str:
    return _digest({"schema": "neurodic.pin_multi.selection/v1", "camera_pairs": values.get("camera_pairs", {})})


def _roles(pair_id: str) -> tuple[str, str]:
    if not isinstance(pair_id, str) or pair_id.count("__") != 1:
        raise ValueError("pair_id must contain exactly one '__' separator")
    left, right = pair_id.split("__", 1)
    if not left or not right: raise ValueError("pair_id camera roles must be non-empty")
    return left, right


def compute_planned_pair_set_identity(*, planned_pair_ids: Sequence[str], selected_frame: int,
                                      selection_config_identity: str, calibration_identity: Mapping[str, Any]) -> str:
    """Identity of intended ordered membership, independent of managed locations."""
    return _digest({"schema": "neurodic.pin_multi.planned-pair-set/v1", "planned_pair_ids": list(planned_pair_ids),
                    "selected_frame": selected_frame, "selection_config_identity": selection_config_identity,
                    "calibration_identity": _identity(calibration_identity)})


def _stage_scope(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    signature = stage.get("producer_signature")
    return signature.get("scope", {}) if isinstance(signature, Mapping) and isinstance(signature.get("scope"), Mapping) else {}


def _legacy_exists(case_root: Path, pair_id: str) -> bool:
    return (case_root / "result" / "pin_multi_slover" / "pairs" / pair_id).is_dir()


def _candidate(stage: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]], trial: Path, *, pair_id: str,
               frame: int, calibration_identity: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one C1 stage without trusting its manifest alone."""
    reasons: list[str] = []
    signature = stage.get("producer_signature")
    if stage.get("action_id") != ACTION: reasons.append("PAIR_RESULT.INVALID_PROVENANCE")
    if not isinstance(signature, Mapping): return None, ["PAIR_RESULT.INVALID_PROVENANCE"]
    scope = signature.get("scope", {})
    if scope.get("pair_id") != pair_id: reasons.append("PAIR_RESULT.SCOPE_MISMATCH")
    if scope.get("selected_frame") != frame: reasons.append("PAIR_RESULT.FRAME_MISMATCH")
    if signature.get("output_contract") != CONTRACT or signature.get("implementation", {}).get("adapter") != IMPLEMENTATION:
        reasons.append("PAIR_RESULT.INVALID_PROVENANCE")
    inputs = signature.get("input_identities", {})
    if not isinstance(inputs, Mapping) or inputs.get("calibration") != calibration_identity:
        reasons.append("PAIR_RESULT.CALIBRATION_MISMATCH")
    attempt = stage.get("stage_attempt_id")
    records = [item for item in artifacts if item.get("stage_attempt_id") == attempt and item.get("producer_action_id") == ACTION
               and item.get("producer_signature") == signature]
    expected = _solve_outputs(pair_id)
    files: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for output in expected:
        matches = [item for item in records if item.get("artifact_type") == output.artifact_type and str(item.get("location", "")).endswith(output.path)]
        if len(matches) != 1:
            reasons.append("PAIR_RESULT.ARTIFACT_CONTRACT")
            continue
        record = matches[0]; path = trial / str(record["location"])
        try:
            actual = content_identity(path).to_dict()
        except Exception:
            reasons.append("PAIR_RESULT.CONTENT_MISMATCH"); continue
        if actual != record.get("identity"):
            reasons.append("PAIR_RESULT.CONTENT_MISMATCH"); continue
        files[output.artifact_type] = (record, path)
    if not files:
        return None, sorted(set(reasons or ["PAIR_RESULT.ARTIFACT_CONTRACT"]))
    root = next(iter(files.values()))[1]
    # C1 publishes all required paths below a single attempt root.  Find it from a known required path.
    for parent in root.parents:
        if parent.name == "scientific":
            attempt_root = parent.parent; break
    else:
        attempt_root = trial
    if not reasons:
        try:
            validate_pair_solve_quality_outputs(attempt_root, pair_id)
            metadata = json.loads((attempt_root / f"scientific/pairs/{pair_id}/pair_metadata.json").read_text(encoding="utf-8"))
            left, right = _roles(pair_id)
            if metadata.get("pair_id") != pair_id or metadata.get("reference_camera") != left or metadata.get("secondary_camera") != right:
                reasons.append("PAIR_RESULT.ROLE_MISMATCH")
            if metadata.get("selected_frame") != frame: reasons.append("PAIR_RESULT.FRAME_MISMATCH")
        except (OSError, ValueError, KeyError):
            reasons.append("PAIR_RESULT.STRUCTURE_INVALID")
    if reasons: return None, sorted(set(reasons))
    quality = json.loads((attempt_root / f"scientific/pairs/{pair_id}/quality/quality.json").read_text(encoding="utf-8"))
    fusion_artifacts = [{"artifact_type": key, "identity": value[0]["identity"]} for key, value in files.items() if key in FUSION_TYPES]
    return {"producer_signature": _identity(signature), "producer_signature_digest": signature.get("digest"),
            "implementation_identity": signature["implementation"]["adapter"], "artifact_contract": CONTRACT,
            "required_artifacts": [{"artifact_type": key, "identity": value[0]["identity"]} for key, value in sorted(files.items())],
            "fusion_input_artifacts": sorted(fusion_artifacts, key=canonical_json),
            "quality_evidence": {key: quality.get(key) for key in ("total_points", "valid_points", "valid_ratio", "mean_reprojection_error_px", "p95_reprojection_error_px", "max_reprojection_error_px", "pin_diagnostics")},
            "scientific_key": _digest({"signature": _identity(signature), "fusion_input_artifacts": sorted(fusion_artifacts, key=canonical_json), "metadata": metadata})}, []


def inspect_pin_multi_pair_set_readiness(config: str | Path, *, managed_root: str | Path, selected_frame: int,
                                          case_key: str | None = None, case_paths: str | Path = "config/case_paths.yaml") -> Envelope:
    """Inspect managed C1 pair results only; never runs or writes scientific work."""
    if not isinstance(selected_frame, int) or isinstance(selected_frame, bool) or selected_frame < 0:
        raise ValueError("selected_frame must be an explicit non-negative integer")
    resolved = resolve_config(config, case_key=case_key, case_paths=case_paths, solver="pin_multi")
    values = resolved["effective_config"]; case_root = Path(values["case"]["root"]).resolve()
    calibration = case_root / values["case"]["calibration"]
    calibration_identity = content_identity(calibration).to_dict()
    selection, _roi = _options_from_config(values)
    selected = select_pin_multi_pairs(json.loads(calibration.read_text(encoding="utf-8")), selection)
    planned = [pair_id_for(left, right) for left, right, _details in selected]
    selection_identity = _selection_identity(values)
    planned_identity = compute_planned_pair_set_identity(planned_pair_ids=planned, selected_frame=selected_frame,
                                                         selection_config_identity=selection_identity, calibration_identity=calibration_identity)
    root = Path(managed_root).resolve()
    if not root.is_dir(): raise FileNotFoundError(str(root))
    trials = root / "trials"
    by_pair: dict[str, list[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Path]]] = {pair: [] for pair in planned}
    for trial in sorted((item for item in trials.iterdir() if item.is_dir()), key=lambda item: item.name) if trials.is_dir() else []:
        try: manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError): continue
        if not isinstance(manifest, Mapping): continue
        artifacts = [item for item in manifest.get("produced_artifacts", []) if isinstance(item, Mapping)]
        for stage in manifest.get("stage_attempts", []):
            if not isinstance(stage, Mapping): continue
            scope = _stage_scope(stage); pair = scope.get("pair_id")
            if pair in by_pair and scope.get("selected_frame") == selected_frame:
                by_pair[pair].append((stage, artifacts, trial))
    pair_reports: list[dict[str, Any]] = []; blocks: list[dict[str, Any]] = []; resolved_inputs: list[dict[str, Any]] = []
    duplicates = sorted({pair for pair in planned if planned.count(pair) > 1})
    for pair in planned:
        left, right = _roles(pair); candidates = by_pair[pair]; valid: list[tuple[str, str, dict[str, Any]]] = []; observed: list[str] = []; invalid: list[str] = []
        for stage, artifacts, trial in candidates:
            status = stage.get("status")
            if status in {"completed", "reused"}:
                value, reasons = _candidate(stage, artifacts, trial, pair_id=pair, frame=selected_frame, calibration_identity=calibration_identity)
                if value is None: invalid.extend(reasons)
                else: valid.append((value["scientific_key"], trial.name, {**value, "state": "complete" if status == "completed" else "reused"}))
            elif status in {"failed", "interrupted"}: observed.append(str(status))
        if valid:
            _key, _location, chosen = sorted(valid, key=lambda item: (item[0], item[1]))[0]; report = chosen
        elif invalid: report = {"state": "invalid", "reasons": sorted(set(invalid))}
        elif "failed" in observed: report = {"state": "failed", "reasons": ["PAIR_RESULT.FAILED"]}
        elif "interrupted" in observed: report = {"state": "interrupted", "reasons": ["PAIR_RESULT.INTERRUPTED"]}
        elif _legacy_exists(case_root, pair): report = {"state": "legacy_only", "reasons": ["PAIR_RESULT.LEGACY_ONLY"]}
        else: report = {"state": "missing", "reasons": ["PAIR_RESULT.MISSING"]}
        report = {"pair_id": pair, "reference_camera": left, "secondary_camera": right, "selected_frame": selected_frame, **report}
        pair_reports.append(report)
        if report["state"] not in {"complete", "reused"}: blocks.append({"pair_id": pair, "code": report["reasons"][0]})
        else: resolved_inputs.append({"pair_id": pair, "reference_camera": left, "secondary_camera": right,
                                      "producer_signature": report["producer_signature"], "fusion_input_artifacts": report["fusion_input_artifacts"]})
    fusion_enabled = bool(values.get("fusion", {}).get("enabled", False))
    if not fusion_enabled: status = "not_applicable"
    elif not planned: status = "not_ready"; blocks.append({"code": "PAIR_SET.EMPTY"})
    elif duplicates: status = "not_ready"; blocks.extend({"pair_id": pair, "code": "PAIR_SET.DUPLICATE_PAIR"} for pair in duplicates)
    elif blocks: status = "not_ready"
    else: status = "ready"
    fusion_input_identity = _digest({"schema": "neurodic.pin_multi.fusion-input/v1", "planned_pair_set_identity": planned_identity,
                                     "inputs": resolved_inputs}) if status == "ready" else None
    summary = {state: sum(item["state"] == state for item in pair_reports) for state in ("complete", "reused", "missing", "failed", "interrupted", "invalid", "legacy_only")}
    data = {"schema": SCHEMA, "planned_pair_set_identity": planned_identity, "fusion_input_identity": fusion_input_identity,
            "scope": {"selected_frame": selected_frame, "planned_pair_ids": planned},
            "selection": {"mode": selection.mode, "config_identity": selection_identity}, "calibration_identity": calibration_identity,
            "fusion_enabled": fusion_enabled, "status": status, "pairs": pair_reports,
            "blocking_reasons": sorted(blocks, key=canonical_json), "summary": {"planned_count": len(planned), **summary}}
    return Envelope(status="ok", operation="inspect.pin_multi_pair_set_readiness", data=data)
