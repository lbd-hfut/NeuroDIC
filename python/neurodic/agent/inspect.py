"""Native-free, read-only inspection for existing NeuroDIC cases and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..case_io import multiview_image_pairs, planar_image_series, stereo_image_pairs
from ..config import load_case_config, load_config
from .adapters import ADAPTERS
from .artifacts import ArtifactRecord, canonical_path, path_within
from .errors import ControlPlaneError, ErrorRecord
from .schemas import AGENT_SCHEMA_VERSION, CapabilityRecord, Envelope


_ALIASES = {"pin": "pin", "pin_2d": "pin", "pin_stereo": "pin_stereo", "stereo": "pin_stereo",
            "pin_multi": "pin_multi", "pin_multi_slover": "pin_multi", "ndef": "ndef"}
_CONFIG_TO_SOLVER = {"pin": "pin", "pin_multi_slover": "pin_multi", "ndef": "ndef"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _exists(path: Path, root: Path, label: str) -> dict[str, Any]:
    return {"id": label, "location": _relative(path, root), "resolved_path": str(path),
            "status": "available" if path.is_file() else "missing"}


def _safe_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _solver_from_config(config: Mapping[str, Any], requested: str | None) -> str:
    if requested:
        try:
            return _ALIASES[requested]
        except KeyError as error:
            raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Unknown canonical solver", True,
                                                details={"solver": requested, "supported": sorted(set(_ALIASES.values()))})) from error
    raw = str(config.get("solver", ""))
    mode = str(config.get("mode", ""))
    if raw == "pin" and mode == "stereo":
        return "pin_stereo"
    if raw in _CONFIG_TO_SOLVER:
        return _CONFIG_TO_SOLVER[raw]
    raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Solver cannot be resolved safely from config", True,
                                        details={"config_solver": raw, "config_mode": mode}))


def resolve_config(solver_config: str | Path, *, case_key: str | None = None,
                   case_paths: str | Path = "config/case_paths.yaml", solver: str | None = None) -> dict[str, Any]:
    """Read and compose existing configs through the project's canonical loader."""
    config_path = canonical_path(solver_config, require_exists=True)
    base = load_config(config_path)
    canonical_solver = _solver_from_config(base, solver)
    adapter = ADAPTERS[canonical_solver]
    selected_key = case_key or adapter.CASE_KEY
    paths_path = canonical_path(case_paths, require_exists=True)
    try:
        effective = load_case_config(config_path, selected_key, paths_path)
    except (OSError, ValueError) as error:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Unable to resolve effective config", True,
                                            path=str(config_path), details={"case_key": selected_key, "reason": str(error)})) from error
    return {"solver": canonical_solver, "legacy_solver": base.get("solver"), "mode": effective.get("mode"),
            "solver_config_path": str(config_path), "case_paths_path": str(paths_path), "case_key": selected_key,
            "effective_config": effective}


def _case_root(values: Mapping[str, Any], config_path: Path, supplied: str | Path | None) -> Path:
    raw = supplied if supplied is not None else values.get("case", {}).get("root")
    if raw is None:
        raise ControlPlaneError(ErrorRecord("SCHEMA.INVALID", "Effective config has no case.root", True))
    path = Path(raw)
    return canonical_path(path if path.is_absolute() else config_path.parent.parent / path)


def _inventory(root: Path, specs: list[tuple[str, Path, str, str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind, path, schema, stage, source in specs:
        if not path.is_file():
            continue
        record = ArtifactRecord.from_file(path, artifact_type=kind, artifact_schema=schema,
                                          producer_stage=stage, root=root,
                                          compatibility={"provenance_status": "legacy_incomplete", "path_source": source})
        value = record.to_dict()
        value["resolved_path"] = str(path.resolve())
        value["provenance_status"] = "legacy_incomplete"
        value["path_source"] = source
        records.append(value)
    return records


def _stage_reports(adapter, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for stage_id, dependencies, required, expected, granularity in adapter.stages():
        items = [item for item in artifacts if item["artifact_type"] in expected]
        observed = [item["artifact_id"] for item in items]
        configured = {item["artifact_type"] for item in items if item["path_source"] == "configured"}
        legacy = {item["artifact_type"] for item in items if item["path_source"] == "legacy_detected"}
        if expected and len(configured) == len(expected): status = "observed_complete"
        elif expected and len(legacy) == len(expected): status = "legacy_observed_complete"
        elif observed: status = "partial"
        else: status = "missing"
        if stage_id.endswith("inputs"):
            status = "observed" if not required else "unknown"
        reports.append({"stage_id": stage_id, "dependencies": dependencies, "required_inputs": required,
                        "expected_artifacts": expected, "observed_artifacts": observed, "status": status,
                        "provenance_status": "legacy_incomplete", "execution_granularity": granularity,
                        "capabilities": CapabilityRecord(reuse_supported=bool(expected), cache_supported=False,
                                                          resume_supported=False,
                                                          notes="Legacy artifacts have unverified compatibility").to_dict()})
    return reports


def _pin(root: Path, values: Mapping[str, Any]):
    case = values["case"]
    inputs, missing, inventory = [], [], []
    try:
        reference, frames, roi = planar_image_series(root, case.get("images_dir", "."))
        inputs = [{"reference": _relative(reference, root), "roi": _relative(roi, root)}]
        frame_info = {"reference": _relative(reference, root), "current_frames": [_relative(item, root) for item in frames],
                      "count": len(frames), "roi_semantics": "last_sorted_image"}
    except (FileNotFoundError, ValueError) as error:
        frame_info = {"count": 0, "error": str(error)}; missing.append({"stage": "pin.inputs", "artifact": "planar_image_series", "reason": "not_found_or_invalid"})
    output = root / values.get("output", {}).get("result", "result/pin")
    inventory = _inventory(root, [("pin_result", output / "pin_result.npz", "npz/v1", "pin.infer", "configured"), ("evaluation", output / "diagnostics_evaluation.json", "json/v1", "pin.evaluate", "configured"), ("pin_result", root / "result/pin/pin_result.npz", "npz/v1", "pin.infer", "legacy_detected")])
    return frame_info, inputs, missing, inventory


def _stereo(root: Path, values: Mapping[str, Any]):
    case = values["case"]; missing = []
    try:
        reference, frames = stereo_image_pairs(root / case["left_images"], root / case["right_images"])
        frame_info = {"reference": [_relative(item, root) for item in reference], "current_frames": [[_relative(item, root) for item in pair] for pair in frames], "count": len(frames), "synchronized": True}
    except (FileNotFoundError, ValueError, KeyError) as error:
        frame_info = {"count": 0, "synchronized": False, "error": str(error)}; missing.append({"stage": "stereo.inputs", "artifact": "stereo_image_series", "reason": "not_found_or_invalid"})
    expected = [("roi", root / case.get("roi", "ROI.bmp"), "image/v1", "stereo.inputs", "configured"), ("camera_pair", root / case.get("camera_pair", ""), "json/v1", "stereo.inputs", "configured")]
    output = root / values.get("output", {}).get("result", "result")
    expected += [("reference_disparity", output / "disp/reference_disparity.npz", "npz/v1", "stereo.planar_fields", "configured"), ("left_temporal", output / "disp/left_temporal.npz", "npz/v1", "stereo.planar_fields", "configured"), ("deformed_disparity", output / "disp/deformed_disparity.npz", "npz/v1", "stereo.planar_fields", "configured"), ("reference_reconstruction", output / "reconstruct/initial.npz", "npz/v1", "stereo.reconstruct", "configured"), ("current_reconstruction", output / "reconstruct/last.npz", "npz/v1", "stereo.reconstruct", "configured"), ("deformation", output / "deformation/initial_to_last.npz", "npz/v1", "stereo.reconstruct", "configured"), ("evaluation", output / "disp/reference_disparity_evaluation.json", "json/v1", "stereo.evaluate", "configured")]
    for kind, path, _, stage, _ in expected[:2]:
        if not path.is_file(): missing.append({"stage": stage, "artifact": kind, "reason": "not_found"})
    return frame_info, [], missing, _inventory(root, expected)


def _multiview(root: Path, values: Mapping[str, Any], *, ndef: bool):
    case = values["case"]; missing = []; image_root = root / case.get("images", "images")
    try:
        names, reference, frames = multiview_image_pairs(image_root)
        frame_info = {"reference": [_relative(item, root) for item in reference], "current_frames": [[_relative(item, root) for item in group] for group in frames], "count": len(frames), "synchronized": True}
    except (FileNotFoundError, ValueError) as error:
        names = []; frame_info = {"count": 0, "synchronized": False, "error": str(error)}; missing.append({"stage": "ndef.inputs" if ndef else "pin_multi.inputs", "artifact": "multiview_image_series", "reason": "not_found_or_invalid"})
    calibration = root / case.get("calibration", "")
    if not calibration.is_file(): missing.append({"stage": "ndef.inputs" if ndef else "pin_multi.inputs", "artifact": "calibration", "reason": "not_found"})
    return names, frame_info, missing, calibration


def _pin_multi(root: Path, values: Mapping[str, Any]):
    names, frames, missing, calibration = _multiview(root, values, ndef=False)
    output = root / values.get("output", {}).get("result", "result/pin_multi")
    legacy = root / "result/pin_multi_slover"
    specs = [("calibration", calibration, "json/v1", "pin_multi.inputs", "configured"), ("pin_multi_manifest", output / "manifest.json", "json/v1", "pin_multi.pair_solve", "configured"), ("pin_multi_manifest", legacy / "manifest.json", "json/v1", "pin_multi.pair_solve", "legacy_detected")]
    for base, source in ((output, "configured"), (legacy, "legacy_detected")):
        for meta in sorted((base / "pair_roi").glob("*/meta.json")) if (base / "pair_roi").is_dir() else []: specs.append(("pair_roi", meta, "json/v1", "pin_multi.pair_roi", source))
        for quality in sorted((base / "pairs").glob("*/quality/quality.json")) if (base / "pairs").is_dir() else []: specs.append(("pair_quality", quality, "json/v1", "pin_multi.pair_quality", source))
        specs += [("fused_surface", base / "fused/reference_surface.npz", "npz/v1", "pin_multi.fusion", source), ("fusion_summary", base / "fused/summary.json", "json/v1", "pin_multi.fusion", source)]
    manifest = _safe_json(output / "manifest.json") or _safe_json(legacy / "manifest.json")
    return frames, names, missing, _inventory(root, specs), {"manifest_observed": manifest is not None, "fusion_enabled": bool(values.get("fusion", {}).get("enabled", False))}


def _ndef(root: Path, values: Mapping[str, Any]):
    names, frames, missing, calibration = _multiview(root, values, ndef=True)
    case, output = values["case"], values.get("output", {})
    result_base = root / output.get("result", "result"); subdir = output.get("ndef_subdir", "ndef")
    result = result_base / subdir if subdir else result_base
    masks = root / case.get("masks", "")
    surface = root / case.get("reference_surface", "")
    sparse = root / values.get("precalculation", {}).get("displacement", "")
    if not surface.is_file(): missing.append({"stage": "ndef.surface", "artifact": "reference_surface", "reason": "not_found"})
    if not sparse.is_file(): missing.append({"stage": "ndef.precalculation", "artifact": "sparse_tracks", "reason": "not_found"})
    if names and not all((masks / f"{name}_mask.npy").is_file() for name in names): missing.append({"stage": "ndef.roi", "artifact": "roi_masks", "reason": "not_found"})
    legacy = root / "result/ndef_multi_slover"
    specs = [("calibration", calibration, "json/v1", "ndef.inputs", "configured"), ("reference_surface", surface, "npz/v1", "ndef.surface", "configured"), ("sparse_tracks", sparse, "npz/v1", "ndef.precalculation", "configured"), ("reference_surface", legacy / "surface/deformation_surface_dataset.npz", "npz/v1", "ndef.surface", "legacy_detected"), ("sparse_tracks", legacy / "precalculation/sparse_tracks.npz", "npz/v1", "ndef.precalculation", "legacy_detected")]
    for base, source in ((result, "configured"), (legacy, "legacy_detected")):
        specs += [("sparse_scale", base / "precalculation/sparse_scale.json", "json/v1", "ndef.precalculation", source), ("training_history", base / "diagnostics/training_history.json", "json/v1", "ndef.deformation.train", source), ("ndef_summary", base / "diagnostics/summary.json", "json/v1", "ndef.deformation.infer", source), ("projection_diagnostics", base / "diagnostics/projection.npz", "npz/v1", "ndef.deformation.infer", source), ("checkpoint_final", base / "deformation/deformation_field.pt", "torch-checkpoint/v1", "ndef.deformation.train", source), ("checkpoint_best", base / "deformation/deformation_field_best.pt", "torch-checkpoint/v1", "ndef.deformation.train", source), ("evaluation", base / "diagnostics/evaluation.json", "json/v1", "ndef.evaluate", source)]
    if masks.is_dir():
        for name in names:
            specs.append(("roi_mask", masks / f"{name}_mask.npy", "npy/v1", "ndef.roi", "configured"))
        specs.append(("roi_metadata", masks.parent / "mask_meta.json", "json/v1", "ndef.roi", "configured"))
    return frames, names, missing, _inventory(root, specs), {"result_path": _relative(result, root), "checkpoint_present": any(path.is_file() for kind, path, _, _, _ in specs if kind.startswith("checkpoint")), "resolved_seeds": {"runtime": values.get("runtime", {}).get("random_seed"), "surface_dense": values.get("surface_dense_training", {}).get("seed"), "sparse": values.get("precalculation", {}).get("sparse", {}).get("random_seed"), "deformation": values.get("deformation_training", {}).get("seed")}}


def inspect_case(solver_config: str | Path, *, case_key: str | None = None, case_paths: str | Path = "config/case_paths.yaml", case_root: str | Path | None = None, solver: str | None = None) -> Envelope:
    """Return a native-free structural report; it never calls a solver or writes files."""
    resolved = resolve_config(solver_config, case_key=case_key, case_paths=case_paths, solver=solver)
    root = _case_root(resolved["effective_config"], Path(resolved["solver_config_path"]), case_root)
    values, canonical = resolved["effective_config"], resolved["solver"]
    if canonical == "pin": frames, inputs, missing, artifacts = _pin(root, values); cameras = [] ; extra = {"inputs": inputs}
    elif canonical == "pin_stereo": frames, inputs, missing, artifacts = _stereo(root, values); cameras = ["left", "right"]; extra = {"inputs": inputs}
    elif canonical == "pin_multi": frames, cameras, missing, artifacts, extra = _pin_multi(root, values)
    else: frames, cameras, missing, artifacts, extra = _ndef(root, values)
    adapter = ADAPTERS[canonical]
    stages = _stage_reports(adapter, artifacts)
    return Envelope(status="ok", operation="inspect.case", data={"solver": canonical, "solver_alias": resolved["legacy_solver"], "mode": resolved["mode"], "case_root": _relative(root, Path.cwd()), "resolved_case_root": str(root), "config": {key: resolved[key] for key in ("solver_config_path", "case_paths_path", "case_key", "effective_config")}, "scope": {"selected_frame": values.get("case", {}).get("frame")}, "frames": frames, "cameras": cameras, "readiness": {"ready": not missing, "missing": missing, "invalid": [], "unknown": []}, "artifacts": artifacts, "stages": stages, "capabilities": CapabilityRecord(reuse_supported=True, cache_supported=False, resume_supported=False, notes="Discovery only; legacy compatibility is unverified").to_dict(), "reuse_candidates": [{"artifact_id": item["artifact_id"], "status": "candidate", "compatibility": "unverified"} for item in artifacts if item["artifact_type"] in {"reference_surface", "reference_disparity", "pair_roi", "fused_surface"}], "legacy_paths": [], **extra})


def inspect_config(*args, **kwargs) -> Envelope:
    kwargs.pop("case_root", None)
    resolved = resolve_config(*args, **kwargs)
    return Envelope(status="ok", operation="inspect.config", data=resolved)


def inspect_pipeline(*args, **kwargs) -> Envelope:
    report = inspect_case(*args, **kwargs)
    return Envelope(status="ok", operation="inspect.pipeline", request_id=report.request_id,
                    data={"solver": report.data["solver"], "stages": report.data["stages"], "readiness": report.data["readiness"], "capabilities": report.data["capabilities"]})


def inspect_artifact(path: str | Path, *, case_root: str | Path, artifact_type: str = "unknown", artifact_schema: str = "unknown/v1", producer_stage: str = "unknown") -> Envelope:
    root = canonical_path(case_root, require_exists=True); resolved = canonical_path(path, require_exists=True)
    if not path_within(resolved, root):
        raise ControlPlaneError(ErrorRecord("FILESYSTEM.OUTSIDE_ROOT", "Artifact is outside the established case root", False, path=str(resolved), details={"case_root": str(root)}))
    record = ArtifactRecord.from_file(resolved, artifact_type=artifact_type, artifact_schema=artifact_schema, producer_stage=producer_stage, root=root, compatibility={"provenance_status": "legacy_incomplete"})
    value = record.to_dict(); value["resolved_path"] = str(resolved)
    value["provenance_status"] = "legacy_incomplete"
    return Envelope(status="ok", operation="inspect.artifact", data={"artifact": value, "provenance_status": "legacy_incomplete"})


def inspect_result(*args, **kwargs) -> Envelope:
    report = inspect_case(*args, **kwargs)
    return Envelope(status="ok", operation="inspect.result", request_id=report.request_id,
                    data={"solver": report.data["solver"], "artifacts": report.data["artifacts"], "stages": report.data["stages"], "readiness": report.data["readiness"]})
