"""Guarded managed NDeF deformation combined action.

This module is intentionally a control-plane adapter.  It freezes the D/E/ROI
evidence and all deformation determinants before importing the public
``ndef_dic`` boundary.  It never calls the C++ solver directly and never loads
pickle-backed checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from ...case_io import named_multiview_image_pairs
from ...ndef_paths import camera_name_from_label
from ..artifacts import content_identity, require_path_within
from ..errors import ControlPlaneError, ErrorRecord
from ..execution import ProducedArtifact, TrustedAction
from ..schemas import canonical_json


ACTION_ID = "ndef.deformation_combined_call"
IMPLEMENTATION_ID = "neurodic.ndef.deformation_combined/v1"
OUTPUT_CONTRACT = "neurodic.ndef.deformation-combined-artifacts/v1"
INPUTS_KEY = "ndef_deformation_inputs"
SURFACE_ACTION_ID = "ndef.combined_surface_call"
SURFACE_IMPLEMENTATION_ID = "neurodic.ndef.surface_combined/v1"
PRECALC_ACTION_ID = "ndef.precalculation_call"
PRECALC_IMPLEMENTATION_ID = "neurodic.ndef.precalculation/v1"
ROI_ACTION_ID = "ndef.roi.generate_call"
ROI_IMPLEMENTATION_ID = "neurodic.ndef.roi/v1"

_NATIVE_BACKEND_MODULE = "neurodic._neurodic"
_REQUIRED_NATIVE_BACKEND_SYMBOLS = (
    "CameraModel",
    "NDeFProblem",
    "NDeFModelOptions",
    "estimate_ndef_displacement_scale",
    "PhotometricLossType",
    "NDeFSolver",
)

_REQUIRED = (
    "reconstruct/reference_surface.npz", "reconstruct/current_surface.npz",
    "deformation/reference_to_current.npz", "diagnostics/projection.npz",
    "diagnostics/training.npz", "diagnostics/training_history.json",
    "diagnostics/summary.json", "deformation/deformation_field.pt",
    "deformation/deformation_field_best.pt",
    "deformation/deformation_field.metadata.json",
    "deformation/deformation_field_best.metadata.json",
)
_EVAL = ("diagnostics/evaluation.npz", "diagnostics/evaluation.json")
_STRAIN_COMPONENTS = ("E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_xz")
_TRAINING_COLUMNS = ("epoch", "step", "loss", "photometric_loss", "smoothness",
                     "valid_pairs", "supervised_pairs", "displacement_rms")


def _import_native_backend():
    """Import the exact extension consumed by the public deformation API."""
    return importlib.import_module(_NATIVE_BACKEND_MODULE)


def ndef_deformation_backend_capability() -> Mapping[str, Any]:
    """Return an import-only, non-scientific deformation binding capability report.

    Import/loader failures are retained verbatim for control-plane diagnosis.
    Unexpected exceptions are not collapsed into an unavailable result.
    """
    try:
        backend = _import_native_backend()
    except (ImportError, OSError, RuntimeError) as error:
        return {
            "schema_version": "neurodic.ndef.deformation-backend-capability/v1",
            "available": False,
            "module": _NATIVE_BACKEND_MODULE,
            "module_file": None,
            "required_symbols": list(_REQUIRED_NATIVE_BACKEND_SYMBOLS),
            "symbols": {},
            "missing_symbols": list(_REQUIRED_NATIVE_BACKEND_SYMBOLS),
            "exception": {"type": type(error).__name__, "message": str(error)},
        }
    symbols = {
        name: {"present": hasattr(backend, name),
               "type": type(getattr(backend, name, None)).__name__}
        for name in _REQUIRED_NATIVE_BACKEND_SYMBOLS
    }
    missing = [name for name, record in symbols.items() if not record["present"]]
    return {
        "schema_version": "neurodic.ndef.deformation-backend-capability/v1",
        "available": not missing,
        "module": _NATIVE_BACKEND_MODULE,
        "module_file": getattr(backend, "__file__", None),
        "required_symbols": list(_REQUIRED_NATIVE_BACKEND_SYMBOLS),
        "symbols": symbols,
        "missing_symbols": missing,
        "exception": None,
    }


def require_ndef_deformation_backend() -> Mapping[str, Any]:
    """Fail closed before the public scientific callable if bindings are absent."""
    capability = ndef_deformation_backend_capability()
    if not capability["available"]:
        raise ControlPlaneError(ErrorRecord(
            "CAPABILITY.UNSUPPORTED",
            "NDeF deformation native backend preflight failed",
            True,
            stage="ndef.deformation.preflight",
            details={
                "module": capability["module"],
                "missing_symbols": capability["missing_symbols"],
                "import_exception": capability["exception"],
            },
        ))
    return capability


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _fail(path: str, message: str, value: Any = None) -> None:
    raise ControlPlaneError(ErrorRecord("NDEF.DEFORMATION_CONFIG_INVALID", message, True,
                                        stage="ndef.deformation", path=path,
                                        details={"actual_type": type(value).__name__, "actual_value": repr(value)}))


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        _fail(key, f"{key} must be a mapping", value)
    return value


def _int(value: Any, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(path, f"{path} must be an integer >= {minimum}", value)
    return int(value)


def _number(value: Any, path: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        _fail(path, f"{path} must be finite numeric", value)
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(path, f"{path} must be >= {minimum}", value)
    return number


def _camera_records(values: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any], list[Mapping[str, Any]], list[str]]:
    case = _mapping(values, "case")
    if not isinstance(case.get("root"), str) or not isinstance(case.get("calibration"), str):
        raise ValueError("NDEF.CALIBRATION_NOT_MANAGED")
    root = Path(case["root"]).resolve()
    calibration = require_path_within(_path(root, case["calibration"]), root, require_exists=True)
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDEF.CALIBRATION_INVALID") from error
    cameras = payload.get("cameras") or payload.get("scaled_cameras")
    if not isinstance(cameras, list) or len(cameras) < 2 or not all(isinstance(item, Mapping) for item in cameras):
        raise ValueError("NDEF.CALIBRATION_FIELDS_INCOMPLETE")
    names: list[str] = []
    required = ("K", "R", "t", "distortion", "image_width", "image_height")
    for index, camera in enumerate(cameras):
        label = camera.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("NDEF.CAMERA_LABEL_FALLBACK_REJECTED")
        name = camera_name_from_label(label, f"cam_{index}")
        if not name or any(key not in camera for key in required):
            raise ValueError("NDEF.CALIBRATION_FIELDS_INCOMPLETE")
        if not isinstance(camera["distortion"], Sequence) or isinstance(camera["distortion"], (str, bytes)):
            raise ValueError("NDEF.CALIBRATION_FIELDS_INCOMPLETE")
        _int(camera["image_width"], f"calibration.cameras[{index}].image_width", 1)
        _int(camera["image_height"], f"calibration.cameras[{index}].image_height", 1)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("NDEF.CAMERA_ORDER_INVALID")
    return root, calibration, payload, list(cameras), names


def _calibration_package(root: Path, calibration: Path, payload: Mapping[str, Any],
                         cameras: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Mapping[str, Any]:
    pairs = require_path_within(calibration.parent / "camera_pairs.json", root, require_exists=True)
    try:
        pair_data = json.loads(pairs.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDEF.CAMERA_TOPOLOGY_INVALID") from error
    if pair_data.get("camera_names") != list(names):
        raise ValueError("NDEF.CAMERA_ORDER_INVALID")
    neighbours = pair_data.get("neighbors")
    if not isinstance(neighbours, Mapping) or set(neighbours) != set(names):
        raise ValueError("NDEF.CAMERA_TOPOLOGY_INVALID")
    if any(not isinstance(neighbours[name], list) or any(item not in names for item in neighbours[name]) for name in names):
        raise ValueError("NDEF.CAMERA_TOPOLOGY_INVALID")
    camera_models = [{key: camera.get(key) for key in
                      ("label", "K", "R", "t", "distortion", "image_width", "image_height")} for camera in cameras]
    return {
        "path": str(calibration.relative_to(root)), "identity": content_identity(calibration).to_dict(),
        "camera_ids": list(names), "camera_models": camera_models,
        "camera_projection_identity": _digest(camera_models),
        "camera_topology": copy.deepcopy(pair_data),
        "camera_pairs": {"path": str(pairs.relative_to(root)), **content_identity(pairs).to_dict()},
        "coordinate_convention": "calibration_world_frame/v1",
        "scale_identity": {key: payload.get(key) for key in
                           ("sfm_to_world_scale", "world_to_sfm_scale", "sfm_to_world_rotation", "sfm_to_world_translation")},
        "sfm_to_world_scale": payload.get("sfm_to_world_scale", 1.0),
    }


def _images(values: Mapping[str, Any], root: Path, names: Sequence[str]) -> Mapping[str, Any]:
    case = _mapping(values, "case")
    image_root = require_path_within(_path(root, str(case.get("images", "images"))), root, require_exists=True)
    references, current_frames = named_multiview_image_pairs(image_root, list(names))
    frame = case.get("frame", -1)
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise ValueError("NDEF.FRAME_INVALID")
    index = frame if frame >= 0 else len(current_frames) + frame
    if index < 0 or index >= len(current_frames):
        raise ValueError("NDEF.FRAME_INVALID")
    def rec(name: str, path: Path) -> Mapping[str, Any]:
        return {"camera_id": name, "path": str(path.relative_to(root)), "identity": content_identity(path).to_dict()}
    return {"frame": frame, "resolved_index": index,
            "reference_images": [rec(name, path) for name, path in zip(names, references)],
            "current_images": [rec(name, path) for name, path in zip(names, current_frames[index])]}


def _dependency(plan: Mapping[str, Any], dependency_id: str, action: str, implementation: str) -> Mapping[str, Any]:
    matches = [item for item in plan.get("upstream_dependencies", ())
               if isinstance(item, Mapping) and item.get("dependency_id") == dependency_id]
    if len(matches) != 1:
        raise ValueError(f"NDEF.{dependency_id.upper()}_NOT_MANAGED")
    dep = matches[0]; signature = dep.get("producer_signature")
    if (dep.get("producer_action_id") != action or not isinstance(signature, Mapping)
            or signature.get("stage_id") != action
            or signature.get("implementation", {}).get("adapter") != implementation):
        raise ValueError(f"NDEF.{dependency_id.upper()}_PRODUCER_MISMATCH")
    required = dep.get("required_artifacts")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError(f"NDEF.{dependency_id.upper()}_ARTIFACT_MISSING")
    return dep


def _surface_declaration(plan: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    dep = _dependency(plan, "ndef_surface", SURFACE_ACTION_ID, SURFACE_IMPLEMENTATION_ID)
    required = [item for item in dep["required_artifacts"] if isinstance(item, Mapping)]
    selected = [item for item in required if Path(str(item.get("relative_path", ""))).name == "deformation_surface_dataset.npz"]
    if len(selected) != 1 or not isinstance(selected[0].get("identity"), Mapping):
        raise ValueError("NDEF.NDEF_SURFACE_ARTIFACT_MISSING")
    return {"producer_action": SURFACE_ACTION_ID, "implementation": SURFACE_IMPLEMENTATION_ID,
            "producer_signature": dict(dep["producer_signature"]),
            "relative_path": str(selected[0]["relative_path"]), "identity": dict(selected[0]["identity"]),
            "camera_ids": list(names),
            "consumed_fields": ["points", "visibility_mask", "projected_uv", "visible_counts"],
            "consumed_field_identities": {field: {"source_artifact_digest": selected[0]["identity"].get("digest")}
                                           for field in ("points", "visibility_mask", "projected_uv", "visible_counts")},
            "copied_fields": ["normals", "source_camera", "projected_depth", "depth_abs_error"]}


def _precalculation_declaration(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    dep = _dependency(plan, "ndef_precalculation", PRECALC_ACTION_ID, PRECALC_IMPLEMENTATION_ID)
    by_name = {Path(str(item.get("relative_path", ""))).name: item for item in dep["required_artifacts"]
               if isinstance(item, Mapping)}
    if "sparse_tracks.npz" not in by_name or "sparse_scale.json" not in by_name:
        raise ValueError("NDEF.NDEF_PRECALCULATION_ARTIFACT_MISSING")
    precalc = _mapping(values, "precalculation")
    sparse = _mapping(precalc, "sparse")
    key = precalc.get("key", "displacement"); statistic = precalc.get("statistic", "mean")
    mad = precalc.get("mad_threshold", 5.0)
    if not isinstance(key, str) or not key or not isinstance(statistic, str) or statistic not in {"median", "mean", "p75", "p90", "max"}:
        raise ValueError("NDEF.PRECALCULATION_OPTIONS_INVALID")
    _number(mad, "precalculation.mad_threshold", 0.0)
    return {"producer_action": PRECALC_ACTION_ID, "implementation": PRECALC_IMPLEMENTATION_ID,
            "producer_signature": dict(dep["producer_signature"]),
            "tracks": {"relative_path": str(by_name["sparse_tracks.npz"]["relative_path"]), "identity": dict(by_name["sparse_tracks.npz"]["identity"]), "key": key},
            "scale_metadata": {"relative_path": str(by_name["sparse_scale.json"]["relative_path"]), "identity": dict(by_name["sparse_scale.json"]["identity"])},
            "derivation": {"key": key, "statistic": statistic, "mad_threshold": float(mad), "resolved_output_scale": "derived_from_displacement"},
            "sparse_device": sparse.get("device")}


def _roi_declaration(plan: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    dep = _dependency(plan, "ndef_roi", ROI_ACTION_ID, ROI_IMPLEMENTATION_ID)
    by_path = {item.get("relative_path"): item for item in dep["required_artifacts"] if isinstance(item, Mapping)}
    expected = ["roi/mask_meta.json", *[f"roi/per_camera/{name}_mask.npy" for name in names]]
    if "roi/masks.npz" in by_path:
        expected.insert(0, "roi/masks.npz")
    if any(path not in by_path or not isinstance(by_path[path].get("identity"), Mapping) for path in expected):
        raise ValueError("NDEF.ROI_ARTIFACT_MISSING")
    actual_masks = [item.get("relative_path") for item in dep["required_artifacts"]
                    if isinstance(item, Mapping) and str(item.get("relative_path", "")).startswith("roi/per_camera/")]
    if actual_masks != [f"roi/per_camera/{name}_mask.npy" for name in names]:
        raise ValueError("NDEF.ROI_ORDER_INVALID")
    return {"producer_action": ROI_ACTION_ID, "implementation": ROI_IMPLEMENTATION_ID,
            "producer_signature": dict(dep["producer_signature"]), "operational": True,
            "mask_pixels_direct_solver_determinant": False,
            "artifacts": [{"relative_path": path, "identity": dict(by_path[path]["identity"])} for path in expected]}


def _model_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    model = _mapping(values, "deformation_model")
    fourier = model.get("fourier_encoding", {})
    if not isinstance(fourier, Mapping):
        _fail("deformation_model.fourier_encoding", "must be a mapping", fourier)
    for key, default in (("hidden_dim", 32), ("hidden_layers", 5)):
        _int(model.get(key, default), f"deformation_model.{key}", 1)
    _number(model.get("output_scale", 1.0), "deformation_model.output_scale", 0.0)
    if not isinstance(fourier.get("enabled", False), bool) or not isinstance(fourier.get("include_input", True), bool):
        raise ValueError("NDEF.DEFORMATION_MODEL_INVALID")
    _int(fourier.get("num_frequencies", 6), "deformation_model.fourier_encoding.num_frequencies", 0)
    _number(fourier.get("angular_scale", math.pi), "deformation_model.fourier_encoding.angular_scale", 0.0)
    return {"input": {"coordinates": "normalized_xyz", "dimension": 3},
            "hidden_dim": int(model.get("hidden_dim", 32)), "hidden_layers": int(model.get("hidden_layers", 5)),
            "activation": "tanh", "output": {"dimension": 3, "field": "displacement"},
            "fourier_encoding": {key: fourier.get(key, default) for key, default in
                                  (("enabled", False), ("num_frequencies", 6), ("include_input", True), ("angular_scale", math.pi))},
            "initialization": "NDeFInternalModel/v1"}


def _training_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    training = _mapping(values, "deformation_training")
    if any(key in values for key in ("checkpoint", "checkpoint_path", "resume", "warm_start")) or any(key in training for key in ("checkpoint", "checkpoint_path", "resume", "warm_start")):
        raise ValueError("NDEF.RESUME_UNSUPPORTED")
    allowed = {"device", "training_epochs", "batch_size", "auto_batch_start", "auto_batch_max", "memory_fraction",
               "max_steps_per_epoch", "prediction_batch_size", "seed", "random_seed", "photometric_iterations",
               "photometric_sample_count", "photometric_learning_rate", "weight_decay", "smoothness_weight",
               "patch_radius", "min_valid_patch_ratio", "invalid_patch_penalty", "photometric_loss"}
    unknown = sorted(set(training) - allowed)
    if unknown:
        raise ValueError(f"NDEF.DEFORMATION_UNKNOWN_FIELDS:{unknown}")
    device = training.get("device", "cpu")
    if not isinstance(device, str) or not device or device.lower() == "auto":
        raise ValueError("NDEF.DEVICE_UNRESOLVED")
    epochs = _int(training.get("training_epochs", 1), "deformation_training.training_epochs", 0)
    batch = _int(training.get("batch_size", 0), "deformation_training.batch_size", 0)
    if batch < 1:
        raise ValueError("NDEF.AUTO_BATCH_UNRESOLVED")
    for key in ("auto_batch_start", "auto_batch_max", "max_steps_per_epoch", "prediction_batch_size", "photometric_iterations", "photometric_sample_count"):
        if key in training: _int(training[key], f"deformation_training.{key}", 0)
    for key in ("memory_fraction", "photometric_learning_rate", "weight_decay", "smoothness_weight", "min_valid_patch_ratio", "invalid_patch_penalty"):
        if key in training: _number(training[key], f"deformation_training.{key}", 0.0)
    if "min_valid_patch_ratio" in training and float(training["min_valid_patch_ratio"]) > 1.0:
        raise ValueError("NDEF.DEFORMATION_TRAINING_INVALID")
    loss = training.get("photometric_loss", "znssd")
    if not isinstance(loss, str) or loss.lower() not in {"mse", "ssd", "znssd"}:
        raise ValueError("NDEF.LOSS_INVALID")
    seed = training.get("seed", training.get("random_seed", _mapping(values, "runtime").get("random_seed", 23)))
    _int(seed, "deformation_training.resolved_seed", 0)
    sample_count = int(training.get("photometric_sample_count", 0))
    resolved_batch = sample_count if sample_count > 0 else batch
    points = 1  # Actual surface N is bound and validated at execution.
    steps_per_epoch = max(1, (points + resolved_batch - 1) // resolved_batch)
    configured_steps = int(training.get("max_steps_per_epoch", 0))
    if configured_steps > 0: steps_per_epoch = min(steps_per_epoch, configured_steps)
    total_steps = int(training.get("photometric_iterations", 0)) if int(training.get("photometric_iterations", 0)) > 0 else epochs * steps_per_epoch
    if total_steps < 1:
        raise ValueError("NDEF.ZERO_TRAINING_STEPS")
    return {"device": device, "training_epochs": epochs, "batch_size": batch, "resolved_batch_size": resolved_batch,
            "auto_batch": False, "auto_batch_start": training.get("auto_batch_start", 1024),
            "auto_batch_max": training.get("auto_batch_max", 0), "memory_fraction": training.get("memory_fraction", 0.8),
            "max_steps_per_epoch": configured_steps, "prediction_batch_size": training.get("prediction_batch_size", 262144),
            "resolved_seed": int(seed), "photometric_learning_rate": training.get("photometric_learning_rate", 0.003),
            "weight_decay": training.get("weight_decay", 0.0), "smoothness_weight": training.get("smoothness_weight", 0.0),
            "patch_radius": training.get("patch_radius", 2), "min_valid_patch_ratio": training.get("min_valid_patch_ratio", 1.0),
            "invalid_patch_penalty": training.get("invalid_patch_penalty", 0.05), "photometric_loss": loss.lower(),
            "photometric_iterations": int(training.get("photometric_iterations", 0)),
            "photometric_sample_count": int(training.get("photometric_sample_count", 0)),
            "resolved_step_semantics": "photometric_iterations" if int(training.get("photometric_iterations", 0)) > 0 else "training_epochs*steps_per_epoch",
            "resolved_total_steps": total_steps, "determinism": "solver_forces_torch_deterministic_algorithms"}


def deformation_config_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    model = _model_projection(values)
    training = _training_projection(values)
    evaluation = _mapping(values, "evaluation")
    if set(evaluation) - {"enabled", "sample_count", "seed"}:
        raise ValueError("NDEF.EVALUATION_UNKNOWN_FIELDS")
    enabled = evaluation.get("enabled", False)
    if not isinstance(enabled, bool): _fail("evaluation.enabled", "must be boolean", enabled)
    sample_count = _int(evaluation.get("sample_count", 0), "evaluation.sample_count", 0)
    seed = _int(evaluation.get("seed", 0), "evaluation.seed", 0)
    return {"model": model, "normalization": {"contract": "bbox_center_half_range", "half_range_clamp": 1e-8},
            "training": training, "evaluation": {"enabled": enabled, "sample_count": sample_count, "seed": seed},
            "fresh_initialization": {"checkpoint": None, "resume": False},
            "interpolation": "not_consumed_by_solver"}


def validate_ndef_deformation_config(values: Mapping[str, Any]) -> None:
    deformation_config_projection(values)


def managed_deformation_inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    root, calibration, payload, cameras, names = _camera_records(values)
    config = deformation_config_projection(values)
    dependencies = {"surface": _surface_declaration(plan, names),
                    "precalculation": _precalculation_declaration(plan, values),
                    "roi": _roi_declaration(plan, names)}
    return {"schema_version": "neurodic.ndef.deformation-inputs/v1",
            "calibration": _calibration_package(root, calibration, payload, cameras, names),
            "camera_ids": list(names), "coordinate_convention": "calibration_world_frame/v1",
            "images": _images(values, root, names), "dependencies": dependencies,
            "config": config, "output_contract": OUTPUT_CONTRACT}


def deformation_readiness(values: Mapping[str, Any], upstream_dependencies: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    try:
        config = deformation_config_projection(values)
        if config["training"]["device"].lower() != "cpu":
            # CUDA is explicit and allowed by the adapter, but the first smoke
            # is intentionally planned for CPU; no runtime is performed here.
            pass
        managed_deformation_inputs({"upstream_dependencies": tuple(upstream_dependencies)}, values)
    except (ControlPlaneError, OSError, ValueError, KeyError) as error:
        code = error.record.code if isinstance(error, ControlPlaneError) else str(error).split(":", 1)[0]
        return [(code, "NDeF managed deformation inputs/configuration are not ready")]
    return []


def _resolve_inputs(scope: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[Path, Path, Path, list[Path]]:
    deps = scope.get("_managed_dependencies")
    if not isinstance(deps, Mapping): raise ValueError("NDEF.DEFORMATION_DEPENDENCIES_MISSING")
    def dep(name: str) -> Mapping[str, Any]:
        value = deps.get(name)
        if not isinstance(value, Mapping): raise ValueError("NDEF.DEFORMATION_DEPENDENCIES_MISSING")
        return value
    surface_dep, precalc_dep, roi_dep = dep("ndef_surface"), dep("ndef_precalculation"), dep("ndef_roi")
    surface_name = Path(expected["dependencies"]["surface"]["relative_path"]).name
    surface = Path(surface_dep.get("files", {}).get(surface_name, "")).resolve()
    tracks = Path(precalc_dep.get("files", {}).get("sparse_tracks.npz", "")).resolve()
    scale_meta = Path(precalc_dep.get("files", {}).get("sparse_scale.json", "")).resolve()
    if not surface.is_file() or content_identity(surface).to_dict() != expected["dependencies"]["surface"]["identity"]:
        raise ValueError("NDEF.SURFACE_CONTENT_MISMATCH")
    expected_tracks = expected["dependencies"]["precalculation"]["tracks"]["identity"]
    expected_scale = expected["dependencies"]["precalculation"]["scale_metadata"]["identity"]
    if not tracks.is_file() or content_identity(tracks).to_dict() != expected_tracks:
        raise ValueError("NDEF.PRECALCULATION_TRACKS_CONTENT_MISMATCH")
    if not scale_meta.is_file() or content_identity(scale_meta).to_dict() != expected_scale:
        raise ValueError("NDEF.PRECALCULATION_SCALE_CONTENT_MISMATCH")
    masks: list[Path] = []
    artifacts_by_name = {Path(str(artifact["relative_path"])).name.replace("_mask.npy", ""): artifact
                         for artifact in expected["dependencies"]["roi"]["artifacts"]
                         if str(artifact["relative_path"]).startswith("roi/per_camera/")}
    for name in expected["camera_ids"]:
        artifact = artifacts_by_name.get(name)
        if not isinstance(artifact, Mapping): raise ValueError("NDEF.ROI_ORDER_INVALID")
        path = Path(roi_dep.get("files", {}).get(f"{name}_mask.npy", "")).resolve()
        if not path.is_file() or content_identity(path).to_dict() != artifact["identity"]:
            raise ValueError("NDEF.ROI_CONTENT_MISMATCH")
        masks.append(path)
    if len(masks) != len(expected["camera_ids"]): raise ValueError("NDEF.ROI_ORDER_INVALID")
    return surface, tracks, scale_meta, masks


def _load_surface(path: Path, names: Sequence[str], sizes: Sequence[tuple[int, int]]) -> Mapping[str, Any]:
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            if "points" not in data.files or "cam_names" not in data.files:
                raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
            values = {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as error:
        if str(error).startswith("NDEF."): raise
        raise ValueError("NDEF.SURFACE_CONTRACT_INVALID") from error
    points = values["points"]
    if points.ndim != 2 or points.shape[1] != 3 or not np.issubdtype(points.dtype, np.floating) or not np.all(np.isfinite(points)):
        raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
    camera_names = [str(item) for item in values["cam_names"].tolist()]
    if camera_names != list(names): raise ValueError("NDEF.SURFACE_CAMERA_ORDER_INVALID")
    n, views = len(points), len(names)
    visibility = values.get("visibility_mask")
    uv = values.get("projected_uv")
    counts = values.get("visible_counts")
    if visibility is None or uv is None or counts is None or visibility.shape != (n, views) or visibility.dtype != bool or uv.shape != (n, views, 2) or counts.shape != (n,):
        raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
    if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(counts)):
        raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
    for index, (height, width) in enumerate(sizes):
        visible = uv[:, index][visibility[:, index]]
        if len(visible) and (np.any(visible[:, 0] < 0) or np.any(visible[:, 0] >= width) or np.any(visible[:, 1] < 0) or np.any(visible[:, 1] >= height)):
            raise ValueError("NDEF.SURFACE_UV_INVALID")
    return values


def _validate_masks(paths: Sequence[Path], sizes: Sequence[tuple[int, int]]) -> None:
    import numpy as np
    for path, (height, width) in zip(paths, sizes):
        try: mask = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error: raise ValueError("NDEF.ROI_MASK_INVALID") from error
        if mask.dtype != bool or mask.shape != (height, width) or mask.size == 0: raise ValueError("NDEF.ROI_MASK_INVALID")


def _tracks_and_scale(path: Path, scale_path: Path, key: str, statistic: str, mad_threshold: float) -> float:
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files: raise ValueError("NDEF.PRECALCULATION_DISPLACEMENT_KEY_MISSING")
            displacement = np.asarray(data[key])
    except (OSError, ValueError) as error:
        if str(error).startswith("NDEF."): raise
        raise ValueError("NDEF.PRECALCULATION_TRACKS_INVALID") from error
    if displacement.ndim != 2 or displacement.shape[1] != 3 or displacement.shape[0] == 0 or not np.issubdtype(displacement.dtype, np.number) or not np.all(np.isfinite(displacement)):
        raise ValueError("NDEF.PRECALCULATION_DISPLACEMENT_INVALID")
    magnitudes = np.linalg.norm(displacement.astype(np.float64), axis=1)
    median = float(np.median(magnitudes)); mad = float(np.median(np.abs(magnitudes - median)))
    bound = float(mad_threshold) * max(mad, 1e-12)
    selected = magnitudes[np.abs(magnitudes - median) <= bound] if bound > 0 else magnitudes
    if selected.size == 0: raise ValueError("NDEF.PRECALCULATION_SCALE_INVALID")
    values = {"median": float(np.median(selected)), "mean": float(np.mean(selected)),
              "p75": float(np.percentile(selected, 75)), "p90": float(np.percentile(selected, 90)),
              "max": float(np.max(selected))}
    result = values[statistic]
    if not math.isfinite(result) or result < 0: raise ValueError("NDEF.PRECALCULATION_SCALE_INVALID")
    try:
        metadata = json.loads(scale_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping): raise ValueError
    except (OSError, ValueError) as error:
        raise ValueError("NDEF.PRECALCULATION_SCALE_INVALID") from error
    return result


def _sizes(inputs: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [(int(item["image_height"]), int(item["image_width"])) for item in inputs["calibration"]["camera_models"]]


def _checkpoint(path: Path, role: str) -> str:
    if not path.is_file() or path.stat().st_size < 32 or path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("NDEF.CHECKPOINT_INVALID")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if not names or any(".." in Path(name).parts or name.startswith("/") for name in names): raise ValueError
            base_names = {Path(name).name for name in names}
            if "data.pkl" not in base_names or "version" not in base_names or not any("data" in Path(name).parts[:-1] for name in names):
                raise ValueError
            # PyTorch's current zip serializer emits these two structural
            # metadata members beside the weights.  They are not executable
            # payloads and are part of the native checkpoint format.
            allowed = {"data.pkl", "version", "byteorder", ".format_version", ".storage_alignment"}
            if any(Path(name).name not in allowed and not ("data" in Path(name).parts[:-1]) and not (".data" in Path(name).parts[:-1] and Path(name).name == "serialization_id") for name in names): raise ValueError
            if any(info.file_size > 256 * 1024 * 1024 for info in archive.infolist()): raise ValueError
    except (OSError, zipfile.BadZipFile, ValueError, KeyError) as error:
        raise ValueError("NDEF.CHECKPOINT_INVALID") from error
    # Optional safe inspection.  This is deliberately weights-only and CPU
    # mapped; archives that cannot be inspected by this restricted loader are
    # retained under the structural ZIP policy above, never unrestricted
    # pickle deserialization.
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = None
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("NDEF.CHECKPOINT_PAYLOAD_INVALID")
        allowed = {"model_state_dict", "coordinate_center", "coordinate_scale", "output_scale",
                   "sfm_to_world_scale", "batch_size", "steps_per_epoch", "completed_epochs",
                   "random_seed", "camera_names", "best_loss"}
        if set(payload) - allowed or not isinstance(payload.get("model_state_dict"), Mapping):
            raise ValueError("NDEF.CHECKPOINT_KEYS_INVALID")
        for value in payload["model_state_dict"].values():
            if hasattr(value, "is_floating_point") and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("NDEF.CHECKPOINT_TENSOR_INVALID")
    return content_identity(path).digest


def _npz(path: Path, required: set[str]) -> Mapping[str, Any]:
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(required) - set(data.files): raise ValueError("NDEF.OUTPUT_KEYS_INVALID")
            values = {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as error:
        if str(error).startswith("NDEF."): raise
        raise ValueError("NDEF.OUTPUT_NPZ_INVALID") from error
    for value in values.values():
        if np.issubdtype(value.dtype, np.object_): raise ValueError("NDEF.OUTPUT_OBJECT_ARRAY")
    return values


def _finite_arrays(values: Mapping[str, Any], *, allow_bool: set[str] = set()) -> None:
    import numpy as np
    for key, value in values.items():
        if key in allow_bool: continue
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError("NDEF.OUTPUT_NONFINITE")


def _deformation_delta_allclose(lhs: Any, rhs: Any, *, operands: Any) -> bool:
    """Compare a serialized coordinate delta with its independently stored field.

    The producer writes the coordinate operands, their subtraction, and the
    deformation field as float32.  A subtraction of large, nearly equal
    coordinates therefore has a cancellation floor that the ordinary
    displacement-relative tolerance does not cover.  The additional term is
    one unit-roundoff for the subtraction and the independently rounded field,
    scaled by the magnitudes of both serialized operands.
    """
    import numpy as np
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    source = np.asarray(operands)
    if left.shape != right.shape or source.shape != left.shape:
        return False
    dtype = np.result_type(left.dtype, right.dtype, source.dtype)
    if not np.issubdtype(dtype, np.inexact):
        return bool(np.array_equal(left, right))
    epsilon = np.finfo(dtype).eps
    source_scale = np.asarray(source, dtype=dtype)
    tolerance = 1e-7 + 1e-5 * np.abs(right) + 2.0 * epsilon * source_scale
    return bool(np.all(np.abs(left - right) <= tolerance))


def _validate_outputs(root: Path, values: Mapping[str, Any], inputs: Mapping[str, Any], *, checkpoints: bool = True) -> list[ProducedArtifact]:
    import numpy as np
    evaluation_enabled = bool(inputs["config"]["evaluation"]["enabled"])
    required_paths = set(_REQUIRED) | (set(_EVAL) if evaluation_enabled else set())
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if not required_paths <= files or any(path not in required_paths and not path.startswith("visualization/") for path in files):
        raise ValueError("NDEF.DEFORMATION_UNCONTROLLED_OUTPUT")
    names = list(inputs["camera_ids"]); views = len(names)
    reference = _npz(root / "reconstruct/reference_surface.npz", {"points", "points_sfm", "sfm_to_world_scale", "cam_names"})
    current = _npz(root / "reconstruct/current_surface.npz", {"points", "points_sfm", "sfm_to_world_scale", "cam_names"})
    deformation = _npz(root / "deformation/reference_to_current.npz", {"reference_points", "current_points", "displacement", "displacement_magnitude", "strain", "strain_components", "reference_points_sfm", "current_points_sfm", "displacement_sfm", "displacement_magnitude_sfm", "sfm_to_world_scale", "cam_names"})
    _finite_arrays(reference); _finite_arrays(current); _finite_arrays(deformation)
    n = int(reference["points"].shape[0]); scale = float(np.asarray(reference["sfm_to_world_scale"]).reshape(()))
    if n < 1 or reference["points"].shape != (n, 3) or current["points"].shape != (n, 3) or current["points_sfm"].shape != (n, 3) or [str(x) for x in reference["cam_names"].tolist()] != names or [str(x) for x in current["cam_names"].tolist()] != names or [str(x) for x in deformation["cam_names"].tolist()] != names or not math.isfinite(scale) or scale <= 0:
        raise ValueError("NDEF.DEFORMATION_CONTRACT_INVALID")
    if deformation["strain"].shape != (n, 6) or [str(x) for x in deformation["strain_components"].tolist()] != list(_STRAIN_COMPONENTS): raise ValueError("NDEF.STRAIN_CONTRACT_INVALID")
    if (not _deformation_delta_allclose(current["points"] - reference["points"], deformation["displacement"], operands=np.abs(current["points"]) + np.abs(reference["points"]))
            or not np.allclose(np.linalg.norm(deformation["displacement"], axis=1), deformation["displacement_magnitude"], rtol=1e-5, atol=1e-7)
            or not _deformation_delta_allclose(deformation["current_points_sfm"] - deformation["reference_points_sfm"], deformation["displacement_sfm"], operands=np.abs(deformation["current_points_sfm"]) + np.abs(deformation["reference_points_sfm"]))
            or not np.allclose(np.linalg.norm(deformation["displacement_sfm"], axis=1), deformation["displacement_magnitude_sfm"], rtol=1e-5, atol=1e-7)):
        raise ValueError("NDEF.DEFORMATION_INCONSISTENT")
    if not np.allclose(reference["points"], reference["points_sfm"] * scale, rtol=1e-4, atol=1e-6) or not np.allclose(current["points"], current["points_sfm"] * scale, rtol=1e-4, atol=1e-6) or not np.allclose(deformation["reference_points"], deformation["reference_points_sfm"] * scale, rtol=1e-4, atol=1e-6) or not np.allclose(deformation["current_points"], deformation["current_points_sfm"] * scale, rtol=1e-4, atol=1e-6) or not np.isclose(float(np.asarray(current["sfm_to_world_scale"]).reshape(())), scale) or not np.isclose(float(np.asarray(deformation["sfm_to_world_scale"]).reshape(())), scale): raise ValueError("NDEF.WORLD_SFM_INCONSISTENT")
    projection = _npz(root / "diagnostics/projection.npz", {"reference_uv", "current_uv", "reference_depth", "current_depth", "valid"})
    if projection["reference_uv"].shape != (n, views, 2) or projection["current_uv"].shape != (n, views, 2) or projection["valid"].shape != (n, views) or projection["valid"].dtype != bool: raise ValueError("NDEF.PROJECTION_CONTRACT_INVALID")
    _finite_arrays(projection, allow_bool={"valid"})
    training = _npz(root / "diagnostics/training.npz", {"history", "history_columns", "batch_size", "steps_per_epoch", "completed_epochs", "random_seed", "output_scale"})
    if [str(x) for x in training["history_columns"].tolist()] != list(_TRAINING_COLUMNS): raise ValueError("NDEF.TRAINING_COLUMNS_INVALID")
    if (training["history"].ndim != 2 or training["history"].shape[1] != len(_TRAINING_COLUMNS)
            or training["history"].shape[0] < 1 or not np.issubdtype(training["history"].dtype, np.number)):
        raise ValueError("NDEF.TRAINING_HISTORY_EMPTY")
    _finite_arrays(training)
    training_cfg = inputs["config"]["training"]
    expected_batch = int(training_cfg.get("resolved_batch_size", training_cfg["batch_size"]))
    expected_steps = max(1, (n + expected_batch - 1) // expected_batch)
    if int(training_cfg.get("max_steps_per_epoch", 0)) > 0:
        expected_steps = min(expected_steps, int(training_cfg["max_steps_per_epoch"]))
    configured_iterations = int(training_cfg.get("photometric_iterations", 0))
    expected_total = configured_iterations if configured_iterations > 0 else int(training_cfg["training_epochs"]) * expected_steps
    expected_epochs = 0 if expected_total == 0 else (expected_total + expected_steps - 1) // expected_steps
    if int(np.asarray(training["batch_size"]).reshape(())) != expected_batch or int(np.asarray(training["steps_per_epoch"]).reshape(())) != expected_steps or int(np.asarray(training["completed_epochs"]).reshape(())) != expected_epochs or int(np.asarray(training["random_seed"]).reshape(())) != int(training_cfg["resolved_seed"]):
        raise ValueError("NDEF.TRAINING_METADATA_MISMATCH")
    expected_scale = inputs["dependencies"]["precalculation"].get("resolved_output_scale")
    if expected_scale is not None and not np.isclose(float(np.asarray(training["output_scale"]).reshape(())), float(expected_scale)):
        raise ValueError("NDEF.TRAINING_METADATA_MISMATCH")
    try: history = json.loads((root / "diagnostics/training_history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise ValueError("NDEF.TRAINING_HISTORY_INVALID") from error
    if not isinstance(history, list) or len(history) != training["history"].shape[0] or any(not isinstance(row, Mapping) or any(key not in row or not isinstance(row[key], (int, float)) or not math.isfinite(float(row[key])) for key in _TRAINING_COLUMNS) for row in history): raise ValueError("NDEF.TRAINING_HISTORY_INVALID")
    try: summary = json.loads((root / "diagnostics/summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise ValueError("NDEF.SUMMARY_INVALID") from error
    if not isinstance(summary, Mapping): raise ValueError("NDEF.SUMMARY_INVALID")
    if evaluation_enabled:
        _npz(root / _EVAL[0], {"indices", "residual"})
        try: evaluation = json.loads((root / _EVAL[1]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error: raise ValueError("NDEF.EVALUATION_INVALID") from error
        if not isinstance(evaluation, Mapping): raise ValueError("NDEF.EVALUATION_INVALID")
    if checkpoints:
        final_digest = _checkpoint(root / "deformation/deformation_field.pt", "final")
        best_digest = _checkpoint(root / "deformation/deformation_field_best.pt", "best")
        center = ((reference["points"].min(axis=0) + reference["points"].max(axis=0)) * 0.5).tolist()
        coordinate_scale = np.maximum((reference["points"].max(axis=0) - reference["points"].min(axis=0)) * 0.5, 1e-8).tolist()
        expected_scale = inputs["dependencies"]["precalculation"].get("resolved_output_scale")
        losses = [float(row["loss"]) for row in history if isinstance(row, Mapping) and isinstance(row.get("loss"), (int, float))]
        expected_steps = int(np.asarray(training["steps_per_epoch"]).reshape(())); expected_epochs = int(np.asarray(training["completed_epochs"]).reshape(()))
        for rel, role, digest in (("deformation/deformation_field.metadata.json", "final", final_digest), ("deformation/deformation_field_best.metadata.json", "best", best_digest)):
            try: metadata = json.loads((root / rel).read_text(encoding="utf-8"))
            except (OSError, ValueError) as error: raise ValueError("NDEF.CHECKPOINT_METADATA_INVALID") from error
            if (not isinstance(metadata, Mapping) or metadata.get("checkpoint_role") != role
                    or metadata.get("checkpoint_sha256") != digest or metadata.get("implementation") != IMPLEMENTATION_ID
                    or metadata.get("producer_signature") is None or metadata.get("camera_order") != names
                    or metadata.get("model_architecture") != inputs["config"]["model"]
                    or not np.allclose(np.asarray(metadata.get("coordinate_center")), np.asarray(center))
                    or not np.allclose(np.asarray(metadata.get("coordinate_scale", metadata.get("coordinate_half_range"))), np.asarray(coordinate_scale))
                    or (expected_scale is not None and not np.isclose(float(metadata.get("output_scale")), float(expected_scale)))
                    or not np.isclose(float(metadata.get("sfm_to_world_scale")), float(np.asarray(reference["sfm_to_world_scale"]).reshape(())))
                    or int(metadata.get("batch")) != expected_batch or int(metadata.get("steps_per_epoch")) != expected_steps
                    or int(metadata.get("completed_epochs")) != expected_epochs or int(metadata.get("seed")) != int(training_cfg["resolved_seed"])):
                raise ValueError("NDEF.CHECKPOINT_METADATA_INVALID")
            expected_loss = (losses[-1] if role == "final" else min(losses)) if losses else None
            if expected_loss is not None and not np.isclose(float(metadata.get("best_loss")), expected_loss):
                raise ValueError("NDEF.CHECKPOINT_METADATA_INVALID")
    output_files = sorted(files)
    return [ProducedArtifact(path, "ndef_deformation_output", "neurodic.ndef.deformation/v1") for path in output_files]


def validate_ndef_deformation_outputs(root: Path, values: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[ProducedArtifact]:
    """Public native-free output validator used by tests and audit tooling."""
    return _validate_outputs(root, values, inputs)


def _execution_overlay(values: Mapping[str, Any], staging: Path, surface: Path, roi_root: Path,
                       resolved_frame: int, mplconfig: Path) -> dict[str, Any]:
    overlay = copy.deepcopy(dict(values)); case = overlay.setdefault("case", {})
    case.update({"reference_surface": str(surface), "masks": str(roi_root), "frame": resolved_frame})
    output = copy.deepcopy(dict(overlay.get("output", {}))) if isinstance(overlay.get("output"), Mapping) else {}
    output.update({"result": str(staging), "visualization": str(staging / "visualization"), "ndef_subdir": None})
    overlay["output"] = output
    # Matplotlib configuration/cache is process-private runtime state, not a
    # scientific or visualization artifact.  Keep it outside publishable
    # staging so the fail-closed output validator never has to classify it.
    overlay["MPLCONFIGDIR"] = str(mplconfig)
    return overlay


def _run(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    # Import and enumerate the exact native deformation contract before the
    # public ndef_dic boundary. This performs no construction or science.
    require_ndef_deformation_backend()
    expected = scope.get(INPUTS_KEY)
    if not isinstance(expected, Mapping): raise ValueError("NDEF.DEFORMATION_INPUTS_NOT_FROZEN")
    actual = managed_deformation_inputs({"upstream_dependencies": scope.get("_planned_dependencies", ())}, values)
    if actual != expected: raise ValueError("NDEF.DEFORMATION_INPUTS_CHANGED")
    surface, tracks, scale_meta, masks = _resolve_inputs(scope, actual)
    sizes = _sizes(actual); surface_payload = _load_surface(surface, actual["camera_ids"], sizes); _validate_masks(masks, sizes)
    derivation = actual["dependencies"]["precalculation"]["derivation"]
    output_scale = _tracks_and_scale(tracks, scale_meta, derivation["key"], derivation["statistic"], float(derivation["mad_threshold"]))
    import numpy as np
    center = (surface_payload["points"].min(axis=0) + surface_payload["points"].max(axis=0)) * 0.5
    half = np.maximum((surface_payload["points"].max(axis=0) - surface_payload["points"].min(axis=0)) * 0.5, 1e-8)
    batch = int(actual["config"]["training"].get("resolved_batch_size", actual["config"]["training"]["batch_size"]))
    steps_per_epoch = max(1, (len(surface_payload["points"]) + batch - 1) // batch)
    if int(actual["config"]["training"].get("max_steps_per_epoch", 0)) > 0:
        steps_per_epoch = min(steps_per_epoch, int(actual["config"]["training"]["max_steps_per_epoch"]))
    iterations = int(actual["config"]["training"].get("photometric_iterations", 0))
    total_steps = iterations if iterations > 0 else int(actual["config"]["training"]["training_epochs"]) * steps_per_epoch
    completed_epochs = 0 if total_steps == 0 else (total_steps + steps_per_epoch - 1) // steps_per_epoch
    with tempfile.TemporaryDirectory(prefix=f"neurodic-{staging.name}-") as private_runtime:
        mplconfig = Path(private_runtime) / "mplconfig"
        mplconfig.mkdir()
        overlay = _execution_overlay(values, staging, surface, masks[0].parent,
                                     int(actual["images"]["resolved_index"]), mplconfig)
        overlay.setdefault("deformation_model", {})["output_scale"] = output_scale
        overlay.setdefault("precalculation", {}).update({
            "displacement": str(tracks), "key": derivation["key"],
            "statistic": derivation["statistic"], "mad_threshold": derivation["mad_threshold"]})
        previous_mpl = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = str(mplconfig)
        try:
            from ...api.ndef_dic import ndef_dic
            ndef_dic(overlay, write_case_artifacts=True)
        finally:
            if previous_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mpl
    # Add control-layer sidecars only after the public export has completed.
    actual["dependencies"]["precalculation"]["resolved_output_scale"] = output_scale
    try:
        history_payload = json.loads((staging / "diagnostics/training_history.json").read_text(encoding="utf-8"))
        losses = [float(row["loss"]) for row in history_payload if isinstance(row, Mapping) and isinstance(row.get("loss"), (int, float)) and math.isfinite(float(row["loss"]))]
    except (OSError, ValueError, KeyError):
        losses = []
    best_loss = min(losses) if losses else None
    final_loss = losses[-1] if losses else None
    for rel, role, checkpoint_name, loss in (("deformation/deformation_field.metadata.json", "final", "deformation/deformation_field.pt", final_loss), ("deformation/deformation_field_best.metadata.json", "best", "deformation/deformation_field_best.pt", best_loss)):
        checkpoint_digest = content_identity(staging / checkpoint_name).digest
        metadata = {"checkpoint_role": role, "implementation": IMPLEMENTATION_ID,
                    "producer_signature": scope.get("_producer_signature"), "checkpoint_sha256": checkpoint_digest,
                    "model_architecture": actual["config"]["model"], "coordinate_center": center.tolist(),
                    "coordinate_scale": half.tolist(), "coordinate_half_range": half.tolist(), "output_scale": output_scale,
                    "sfm_to_world_scale": actual["calibration"]["sfm_to_world_scale"],
                    "camera_order": actual["camera_ids"], "seed": actual["config"]["training"]["resolved_seed"],
                    "batch": batch, "steps_per_epoch": steps_per_epoch,
                    "completed_epochs": completed_epochs, "best_loss": loss}
        (staging / rel).write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return _validate_outputs(staging, values, actual)


def _input_identities(plan: Mapping[str, Any], _values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = plan.get("scope", {}).get(INPUTS_KEY)
    if not isinstance(frozen, Mapping): raise ValueError("NDEF deformation signature requires frozen managed inputs")
    return {"managed_ndef_deformation_inputs": frozen}


def deformation_output_paths(signature: Any) -> Sequence[str]:
    enabled = bool(signature.input_identities.get("managed_ndef_deformation_inputs", {}).get("config", {}).get("evaluation", {}).get("enabled", False))
    return (*_REQUIRED, *(_EVAL if enabled else ()), "visualization/**")


def guarded_ndef_deformation_action() -> TrustedAction:
    return TrustedAction(ACTION_ID, _run, IMPLEMENTATION_ID, output_contract=OUTPUT_CONTRACT,
                         input_identities=_input_identities, config_projection=deformation_config_projection,
                         output_paths_resolver=deformation_output_paths)
