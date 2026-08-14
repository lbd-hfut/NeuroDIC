"""Guarded managed NDeF sparse-precalculation execution.

This adapter is deliberately a control-plane boundary.  It binds the managed
surface, ROI, calibration, image/frame, and sparse-option identities before
the public ``ndef_sparse_precalculation`` API is imported.  No native/Torch
module is imported while planning or validating outputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...case_io import image_files, named_multiview_image_pairs
from ...ndef_paths import camera_name_from_label
from ..artifacts import content_identity, require_path_within
from ..errors import ControlPlaneError, ErrorRecord
from ..execution import ProducedArtifact, TrustedAction
from ..schemas import canonical_json


ACTION_ID = "ndef.precalculation_call"
IMPLEMENTATION_ID = "neurodic.ndef.precalculation/v1"
OUTPUT_CONTRACT = "neurodic.ndef.precalculation-artifacts/v1"
INPUTS_KEY = "ndef_precalculation_inputs"
SURFACE_ACTION_ID = "ndef.combined_surface_call"
SURFACE_IMPLEMENTATION_ID = "neurodic.ndef.surface_combined/v1"
ROI_ACTION_ID = "ndef.roi.generate_call"
ROI_IMPLEMENTATION_ID = "neurodic.ndef.roi/v1"

_TRACKS = "precalculation/sparse_tracks.npz"
_SCALE = "precalculation/sparse_scale.json"
_TRACK_KEYS = {
    "source_camera", "source_uv", "reference_points", "current_points",
    "displacement", "displacement_magnitude", "camera_count",
    "reference_reprojection_error", "current_reprojection_error",
    "mean_match_score", "inlier_mask",
}
_SPARSE_KEYS = (
    "points_per_camera", "neighbors_per_camera", "patch_radius",
    "cross_search_radius", "temporal_search_radius", "cross_ncc_threshold",
    "temporal_ncc_threshold", "min_texture_std", "max_reprojection_error",
    "displacement_mad_threshold", "match_batch_size", "device", "random_seed",
)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _error(path: str, message: str, value: Any) -> None:
    raise ControlPlaneError(ErrorRecord("NDEF.PRECALCULATION_CONFIG_INVALID", message, True,
                                        stage="ndef.precalculation", path=path,
                                        details={"actual_type": type(value).__name__, "actual_value": repr(value)}))


def _int(value: Any, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _error(path, f"{path} must be an integer >= {minimum}", value)
    return int(value)


def _number(value: Any, path: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        _error(path, f"{path} must be finite numeric", value)
    number = float(value)
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        _error(path, f"{path} is outside the production range", value)
    return number


def _camera_records(values: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any], list[Mapping[str, Any]], list[str]]:
    case = values.get("case", {})
    if not isinstance(case, Mapping) or not isinstance(case.get("root"), str) or not isinstance(case.get("calibration"), str):
        raise ValueError("NDeF precalculation requires explicit case.root and case.calibration")
    root = Path(case["root"]).resolve()
    calibration = require_path_within(_path(root, case["calibration"]), root, require_exists=True)
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("NDeF calibration is not valid JSON") from exc
    cameras = payload.get("cameras") or payload.get("scaled_cameras")
    if not isinstance(cameras, list) or len(cameras) < 2 or not all(isinstance(item, Mapping) for item in cameras):
        raise ValueError("NDeF calibration lacks an ordered camera array")
    names: list[str] = []
    required = ("K", "R", "t", "distortion", "image_width", "image_height")
    for index, camera in enumerate(cameras):
        raw_label = camera.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("NDEF.CAMERA_LABEL_FALLBACK_REJECTED")
        name = camera_name_from_label(raw_label, f"cam_{index}")
        if not name:
            raise ValueError("NDEF.CAMERA_LABEL_FALLBACK_REJECTED")
        if any(key not in camera for key in required):
            raise ValueError("NDEF.CALIBRATION_FIELDS_INCOMPLETE")
        if not isinstance(camera["distortion"], Sequence) or isinstance(camera["distortion"], (str, bytes)):
            raise ValueError("NDEF.CALIBRATION_DISTORTION_MISSING")
        _int(camera["image_width"], f"calibration.cameras[{index}].image_width", 1)
        _int(camera["image_height"], f"calibration.cameras[{index}].image_height", 1)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("NDEF.CAMERA_ORDER_INVALID")
    return root, calibration, payload, list(cameras), names


def _calibration_package(values: Mapping[str, Any], root: Path, calibration: Path,
                         payload: Mapping[str, Any], cameras: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Mapping[str, Any]:
    package: dict[str, Any] = {
        "path": str(calibration.relative_to(root)),
        "identity": content_identity(calibration).to_dict(),
        "camera_ids": list(names),
        "camera_models": [{key: camera.get(key) for key in ("label", "K", "R", "t", "distortion", "image_width", "image_height")} for camera in cameras],
    }
    package["camera_projection_identity"] = _digest(package["camera_models"])
    package["scale_identity"] = {key: payload.get(key) for key in (
        "sfm_to_world_scale", "world_to_sfm_scale", "sfm_to_world_rotation", "sfm_to_world_translation")}
    package["coordinate_convention"] = "calibration_world_frame/v1"
    pairs = require_path_within(calibration.parent / "camera_pairs.json", root, require_exists=True)
    try:
        pair_data = json.loads(pairs.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("NDEF camera topology is not valid JSON") from exc
    if pair_data.get("camera_names") != list(names):
        raise ValueError("NDEF.CAMERA_ORDER_INVALID")
    neighbors = pair_data.get("neighbors")
    if not isinstance(neighbors, Mapping) or set(neighbors) != set(names) or any(
            not isinstance(neighbors[name], list) or any(item not in names for item in neighbors[name]) for name in names):
        raise ValueError("NDEF.CAMERA_TOPOLOGY_INVALID")
    package["camera_topology"] = copy.deepcopy(pair_data)
    package["camera_pairs"] = {"path": str(pairs.relative_to(root)), **content_identity(pairs).to_dict()}
    observations = require_path_within(calibration.parent / "observations.npz", root, require_exists=True)
    package["observations"] = {"path": str(observations.relative_to(root)), **content_identity(observations).to_dict()}
    return package


def _image_binding(values: Mapping[str, Any], root: Path, names: Sequence[str]) -> Mapping[str, Any]:
    case = values["case"]
    image_root = require_path_within(_path(root, str(case.get("images", "images"))), root, require_exists=True)
    references, deformed = named_multiview_image_pairs(image_root, list(names))
    frame = case.get("frame", -1)
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise ValueError("NDEF.FRAME_INVALID")
    if not deformed:
        raise ValueError("NDEF.FRAME_INVALID")
    index = frame if frame >= 0 else len(deformed) + frame
    if index < 0 or index >= len(deformed):
        raise ValueError("NDEF.FRAME_INVALID")
    def record(name: str, path: Path) -> Mapping[str, Any]:
        return {"camera_id": name, "path": str(path.relative_to(root)), "identity": content_identity(path).to_dict()}
    return {"frame": frame, "resolved_index": index,
            "reference_images": [record(name, path) for name, path in zip(names, references)],
            "current_images": [record(name, path) for name, path in zip(names, deformed[index])]}


def _surface_declaration(plan: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    matches = [item for item in plan.get("upstream_dependencies", ())
               if isinstance(item, Mapping) and item.get("dependency_id") == "ndef_surface"]
    if len(matches) != 1:
        raise ValueError("NDEF.SURFACE_NOT_MANAGED")
    dep = matches[0]; signature = dep.get("producer_signature")
    if (dep.get("producer_action_id") != SURFACE_ACTION_ID or not isinstance(signature, Mapping)
            or signature.get("stage_id") != SURFACE_ACTION_ID
            or signature.get("implementation", {}).get("adapter") != SURFACE_IMPLEMENTATION_ID):
        raise ValueError("NDEF.SURFACE_PRODUCER_MISMATCH")
    required = dep.get("required_artifacts")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError("NDEF.SURFACE_ARTIFACT_MISSING")
    allowed = {"scientific/surface/deformation_surface_dataset.npz", "surface/deformation_surface_dataset.npz"}
    selected = [item for item in required if isinstance(item, Mapping) and item.get("relative_path") in allowed]
    if len(selected) != 1 or not isinstance(selected[0].get("identity"), Mapping):
        raise ValueError("NDEF.SURFACE_ARTIFACT_MISSING")
    return {"producer_action": SURFACE_ACTION_ID, "implementation": SURFACE_IMPLEMENTATION_ID,
            "producer_signature": dict(signature), "relative_path": selected[0]["relative_path"],
            "identity": dict(selected[0]["identity"]), "camera_ids": list(names)}


def _roi_declaration(plan: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    matches = [item for item in plan.get("upstream_dependencies", ())
               if isinstance(item, Mapping) and item.get("dependency_id") == "ndef_roi"]
    if len(matches) != 1:
        raise ValueError("NDEF.ROI_NOT_MANAGED")
    dep = matches[0]; signature = dep.get("producer_signature")
    if (dep.get("producer_action_id") != ROI_ACTION_ID or not isinstance(signature, Mapping)
            or signature.get("stage_id") != ROI_ACTION_ID
            or signature.get("implementation", {}).get("adapter") != ROI_IMPLEMENTATION_ID):
        raise ValueError("NDEF.ROI_PRODUCER_MISMATCH")
    required = dep.get("required_artifacts")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError("NDEF.ROI_ARTIFACT_MISSING")
    by_path = {item.get("relative_path"): item for item in required if isinstance(item, Mapping)}
    masks = [f"roi/per_camera/{name}_mask.npy" for name in names]
    if any(path not in by_path or not isinstance(by_path[path].get("identity"), Mapping) for path in masks):
        raise ValueError("NDEF.ROI_ARTIFACT_MISSING")
    declared_masks = [item.get("relative_path") for item in required
                      if isinstance(item, Mapping) and str(item.get("relative_path", "")).startswith("roi/per_camera/")]
    if declared_masks != masks:
        raise ValueError("NDEF.ROI_ORDER_INVALID")
    return {"producer_action": ROI_ACTION_ID, "implementation": ROI_IMPLEMENTATION_ID,
            "producer_signature": dict(signature),
            "artifacts": [{"relative_path": path, "identity": dict(by_path[path]["identity"])} for path in masks]}


def _config_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = values.get("runtime", {}); case = values.get("case", {}); surface = values.get("surface", {})
    precalc = values.get("precalculation", {})
    sparse = precalc.get("sparse", {}) if isinstance(precalc, Mapping) else None
    if not isinstance(runtime, Mapping) or not isinstance(case, Mapping) or not isinstance(surface, Mapping) or not isinstance(sparse, Mapping):
        raise ValueError("NDEF.PRECALCULATION_CONFIG_INVALID")
    unknown = sorted(set(sparse) - set(_SPARSE_KEYS))
    if unknown:
        raise ValueError(f"NDEF.PRECALCULATION_UNKNOWN_FIELDS:{unknown}")
    required = ("points_per_camera", "neighbors_per_camera", "patch_radius", "cross_search_radius",
                "temporal_search_radius", "cross_ncc_threshold", "temporal_ncc_threshold", "min_texture_std",
                "max_reprojection_error", "displacement_mad_threshold", "match_batch_size", "device")
    if any(key not in sparse for key in required):
        raise ValueError("NDEF.PRECALCULATION_OPTIONS_INCOMPLETE")
    for key in ("points_per_camera", "neighbors_per_camera", "patch_radius", "cross_search_radius", "temporal_search_radius", "match_batch_size"):
        _int(sparse[key], f"precalculation.sparse.{key}", 0)
    for key in ("cross_ncc_threshold", "temporal_ncc_threshold"):
        _number(sparse[key], f"precalculation.sparse.{key}", minimum=0.0, maximum=1.0)
    for key in ("min_texture_std", "max_reprojection_error", "displacement_mad_threshold"):
        _number(sparse[key], f"precalculation.sparse.{key}", minimum=0.0)
    if not isinstance(sparse["device"], str) or not sparse["device"]:
        _error("precalculation.sparse.device", "device must be an explicit string", sparse["device"])
    if sparse["device"].lower() == "auto":
        raise ValueError("NDEF.PRECALCULATION_DEVICE_UNRESOLVED")
    runtime_seed = runtime.get("random_seed", 23)
    if "random_seed" in runtime:
        _int(runtime_seed, "runtime.random_seed", 0)
    if "deterministic" in runtime and not isinstance(runtime["deterministic"], bool):
        _error("runtime.deterministic", "runtime.deterministic must be boolean", runtime["deterministic"])
    sparse_seed = sparse.get("random_seed", runtime_seed)
    _int(sparse_seed, "precalculation.sparse.random_seed", 0)
    frame = case.get("frame", -1)
    if not isinstance(frame, int) or isinstance(frame, bool):
        _error("case.frame", "case.frame must be integer", frame)
    max_points = surface.get("max_points")
    _int(max_points, "surface.max_points", 1)
    return {"runtime": {"random_seed": runtime_seed, "deterministic": runtime.get("deterministic", False)},
            "case": {"frame": frame}, "surface": {"max_points": max_points},
            "precalculation": {"sparse": {key: sparse.get(key) for key in _SPARSE_KEYS if key in sparse},
                                "resolved_random_seed": sparse_seed, "resolved_device": sparse["device"]}}


def precalculation_config_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Public name for the exact scientific projection used in signatures."""
    return _config_projection(values)


def validate_ndef_precalculation_config(values: Mapping[str, Any]) -> None:
    """Validate consumed sparse options without importing the scientific API."""
    _config_projection(values)


def managed_precalculation_inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    root, calibration, payload, cameras, names = _camera_records(values)
    config = _config_projection(values)
    return {"schema_version": "neurodic.ndef.precalculation-inputs/v1",
            "calibration": _calibration_package(values, root, calibration, payload, cameras, names),
            "camera_ids": list(names), "coordinate_convention": "calibration_world_frame/v1",
            "scale_identity": {key: payload.get(key) for key in ("sfm_to_world_scale", "world_to_sfm_scale", "sfm_to_world_rotation", "sfm_to_world_translation")},
            "surface": _surface_declaration(plan, names), "roi": _roi_declaration(plan, names),
            "images": _image_binding(values, root, names), "config": config}


def precalculation_readiness(values: Mapping[str, Any], upstream_dependencies: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    try:
        _config_projection(values)
    except (ControlPlaneError, OSError, ValueError, KeyError) as exc:
        return [(str(exc).split(":", 1)[0], "NDeF precalculation configuration or resolved device is not ready")]
    try:
        managed_precalculation_inputs({"upstream_dependencies": tuple(upstream_dependencies)}, values)
    except (OSError, ValueError, KeyError, ControlPlaneError) as exc:
        return [(str(exc).split(":", 1)[0], "NDeF precalculation managed inputs are not complete")]
    return []


def _resolved_inputs(scope: Mapping[str, Any], names: Sequence[str], expected: Mapping[str, Any]) -> tuple[Path, list[Path]]:
    dependencies = scope.get("_managed_dependencies", {})
    if not isinstance(dependencies, Mapping):
        raise ValueError("NDEF.PRECALCULATION_DEPENDENCIES_MISSING")
    surface = dependencies.get("ndef_surface"); roi = dependencies.get("ndef_roi")
    if not isinstance(surface, Mapping) or not isinstance(roi, Mapping):
        raise ValueError("NDEF.PRECALCULATION_DEPENDENCIES_MISSING")
    sf = surface.get("files", {}); rf = roi.get("files", {})
    surface_name = Path(expected["surface"]["relative_path"]).name
    surface_path = Path(sf.get(surface_name, "")).resolve()
    if not surface_path.is_file() or content_identity(surface_path).to_dict() != expected["surface"]["identity"]:
        raise ValueError("NDEF.SURFACE_CONTENT_MISMATCH")
    masks: list[Path] = []
    for name, expected_artifact in zip(names, expected["roi"]["artifacts"]):
        key = f"{name}_mask.npy"; path = Path(rf.get(key, "")).resolve()
        if not path.is_file() or content_identity(path).to_dict() != expected_artifact["identity"]:
            raise ValueError("NDEF.ROI_CONTENT_MISMATCH")
        masks.append(path)
    if len({item.parent for item in masks}) != 1:
        raise ValueError("NDEF.ROI_ORDER_INVALID")
    return surface_path, masks


def _validate_surface(path: Path, names: Sequence[str], image_sizes: Sequence[tuple[int, int]]) -> None:
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"points", "visibility_mask", "projected_uv", "cam_names"}
            if required - set(data.files):
                raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
            points = np.asarray(data["points"]); visibility = np.asarray(data["visibility_mask"])
            uv = np.asarray(data["projected_uv"]); cam_names = [str(x) for x in np.asarray(data["cam_names"]).tolist()]
            if points.ndim != 2 or points.shape[1] != 3 or not np.issubdtype(points.dtype, np.floating) or not np.all(np.isfinite(points)):
                raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
            if visibility.shape != (len(points), len(names)) or visibility.dtype != bool:
                raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
            if uv.shape != (len(points), len(names), 2) or not np.issubdtype(uv.dtype, np.floating) or not np.all(np.isfinite(uv)):
                raise ValueError("NDEF.SURFACE_CONTRACT_INVALID")
            if cam_names != list(names):
                raise ValueError("NDEF.SURFACE_CAMERA_ORDER_INVALID")
            for index, (height, width) in enumerate(image_sizes):
                visible = uv[:, index][visibility[:, index]]
                if len(visible) and (np.any(visible[:, 0] < 0) or np.any(visible[:, 0] >= width)
                                     or np.any(visible[:, 1] < 0) or np.any(visible[:, 1] >= height)):
                    raise ValueError("NDEF.SURFACE_UV_INVALID")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("NDEF."):
            raise
        raise ValueError("NDEF.SURFACE_CONTRACT_INVALID") from exc


def _validate_masks(paths: Sequence[Path], image_sizes: Sequence[tuple[int, int]]) -> None:
    import numpy as np
    arrays = []
    for path, (height, width) in zip(paths, image_sizes):
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError("NDEF.ROI_MASK_INVALID") from exc
        if array.dtype != bool or array.shape != (height, width) or not np.all(np.isfinite(array.astype(np.float32))):
            raise ValueError("NDEF.ROI_MASK_INVALID")
        arrays.append(array)
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("NDEF.ROI_MASK_INVALID")


def _image_sizes(inputs: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [(int(camera["image_height"]), int(camera["image_width"])) for camera in inputs["calibration"]["camera_models"]]


def _outputs() -> list[ProducedArtifact]:
    return [ProducedArtifact(_TRACKS, "ndef_sparse_tracks", "neurodic.ndef.precalculation-tracks/v1"),
            ProducedArtifact(_SCALE, "ndef_sparse_scale", "neurodic.ndef.precalculation-scale/v1")]


def validate_ndef_precalculation_outputs(root: Path, values: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[ProducedArtifact]:
    import numpy as np
    outputs = _outputs()
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if files != sorted(item.path for item in outputs):
        raise ValueError("NDEF.PRECALCULATION_UNCONTROLLED_OUTPUT")
    for item in outputs:
        path = require_path_within(root / item.path, root, require_exists=True)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("NDEF.PRECALCULATION_OUTPUT_MISSING")
    names = list(inputs["camera_ids"]); cameras = len(names); sizes = _image_sizes(inputs)
    config = inputs["config"]; sparse = config["precalculation"]["sparse"]
    try:
        with np.load(root / _TRACKS, allow_pickle=False) as data:
            if set(data.files) != _TRACK_KEYS:
                raise ValueError("NDEF.PRECALCULATION_TRACK_KEYS_INVALID")
            arrays = {key: np.asarray(data[key]) for key in _TRACK_KEYS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("NDEF."):
            raise
        raise ValueError("NDEF.PRECALCULATION_TRACKS_INVALID") from exc
    n = len(arrays["source_camera"])
    if n < 1:
        raise ValueError("NDEF.PRECALCULATION_TRACKS_EMPTY")
    expected_shapes = {"source_camera": (n,), "source_uv": (n, 2), "reference_points": (n, 3), "current_points": (n, 3),
                       "displacement": (n, 3), "displacement_magnitude": (n,), "camera_count": (n,),
                       "reference_reprojection_error": (n,), "current_reprojection_error": (n,),
                       "mean_match_score": (n,), "inlier_mask": (n,)}
    if any(arrays[key].shape != shape for key, shape in expected_shapes.items()):
        raise ValueError("NDEF.PRECALCULATION_TRACK_SHAPES_INVALID")
    for key in ("source_camera", "camera_count"):
        if arrays[key].dtype != np.dtype("int64"):
            raise ValueError("NDEF.PRECALCULATION_TRACK_DTYPE_INVALID")
    if arrays["inlier_mask"].dtype != bool:
        raise ValueError("NDEF.PRECALCULATION_TRACK_DTYPE_INVALID")
    for key in ("source_uv", "reference_points", "current_points", "displacement", "displacement_magnitude",
                "reference_reprojection_error", "current_reprojection_error", "mean_match_score"):
        if arrays[key].dtype != np.dtype("float64") or not np.all(np.isfinite(arrays[key])):
            raise ValueError("NDEF.PRECALCULATION_TRACK_VALUES_INVALID")
    if np.any(arrays["source_camera"] < 0) or np.any(arrays["source_camera"] >= cameras) or np.any(arrays["camera_count"] < 2) or np.any(arrays["camera_count"] > cameras):
        raise ValueError("NDEF.PRECALCULATION_TRACK_CAMERA_INVALID")
    if np.any(arrays["source_uv"][:, 0] < 0) or np.any(arrays["source_uv"][:, 0] >= np.asarray([item[1] for item in sizes])[arrays["source_camera"]]) or np.any(arrays["source_uv"][:, 1] < 0) or np.any(arrays["source_uv"][:, 1] >= np.asarray([item[0] for item in sizes])[arrays["source_camera"]]):
        raise ValueError("NDEF.PRECALCULATION_TRACK_UV_INVALID")
    if np.any(arrays["reference_reprojection_error"] < 0) or np.any(arrays["current_reprojection_error"] < 0) or np.any(arrays["displacement_magnitude"] < 0):
        raise ValueError("NDEF.PRECALCULATION_TRACK_VALUES_INVALID")
    if not np.allclose(arrays["current_points"] - arrays["reference_points"], arrays["displacement"], rtol=1e-7, atol=1e-8):
        raise ValueError("NDEF.PRECALCULATION_DISPLACEMENT_INCONSISTENT")
    if not np.allclose(np.linalg.norm(arrays["displacement"], axis=1), arrays["displacement_magnitude"], rtol=1e-6, atol=1e-8):
        raise ValueError("NDEF.PRECALCULATION_MAGNITUDE_INCONSISTENT")
    try:
        metadata = json.loads((root / _SCALE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("NDEF.PRECALCULATION_SCALE_INVALID") from exc
    if (not isinstance(metadata, Mapping) or not isinstance(metadata.get("n_tracks"), int) or isinstance(metadata.get("n_tracks"), bool)
            or not isinstance(metadata.get("n_inliers"), int) or isinstance(metadata.get("n_inliers"), bool)
            or metadata.get("n_tracks") != n or metadata.get("n_inliers") != int(arrays["inlier_mask"].sum())):
        raise ValueError("NDEF.PRECALCULATION_SCALE_COUNTS_INVALID")
    stats = metadata.get("scale_stats")
    if not isinstance(stats, Mapping) or any(key not in stats or not isinstance(stats[key], (int, float)) or not math.isfinite(float(stats[key])) or float(stats[key]) < 0 for key in ("median", "mean", "p75", "p90", "max")):
        raise ValueError("NDEF.PRECALCULATION_SCALE_STATS_INVALID")
    per_camera = metadata.get("per_camera")
    if (not isinstance(per_camera, list) or len(per_camera) != cameras
            or not all(isinstance(item, Mapping) for item in per_camera)
            or [item.get("camera") for item in per_camera] != names):
        raise ValueError("NDEF.PRECALCULATION_SCALE_CAMERA_INVALID")
    if any((not isinstance(item.get("requested_seeds"), int) or isinstance(item.get("requested_seeds"), bool)
            or not isinstance(item.get("triangulated_tracks"), int) or isinstance(item.get("triangulated_tracks"), bool)
            or item.get("requested_seeds") != sparse["points_per_camera"]
            or item.get("triangulated_tracks") != int(np.count_nonzero(arrays["source_camera"] == index)))
           for index, item in enumerate(per_camera)):
        raise ValueError("NDEF.PRECALCULATION_SCALE_COUNTS_INVALID")
    sampling = metadata.get("sampling")
    if (not isinstance(sampling, Mapping) or not isinstance(sampling.get("method"), str) or not sampling["method"]
            or sampling.get("random_seed") != config["precalculation"]["resolved_random_seed"]
            or sampling.get("min_texture_std") != sparse["min_texture_std"]
            or sampling.get("without_replacement") is not True):
        raise ValueError("NDEF.PRECALCULATION_SCALE_SAMPLING_INVALID")
    if not isinstance(metadata.get("coordinate_unit"), str) or not metadata["coordinate_unit"]:
        raise ValueError("NDEF.PRECALCULATION_SCALE_INVALID")
    return outputs


def _execution_overlay(values: Mapping[str, Any], staging: Path, surface_path: Path, roi_root: Path) -> dict[str, Any]:
    overlay = copy.deepcopy(dict(values)); case = overlay.setdefault("case", {})
    case["reference_surface"] = str(surface_path)
    case["masks"] = str(roi_root)
    output = copy.deepcopy(dict(overlay.get("output", {}))) if isinstance(overlay.get("output"), Mapping) else {}
    output.update({"result": str(staging), "visualization": str(staging / "visualization"), "ndef_subdir": None})
    overlay["output"] = output
    return overlay


def _run_ndef_precalculation(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    expected = scope.get(INPUTS_KEY)
    if not isinstance(expected, Mapping):
        raise ValueError("NDEF.PRECALCULATION_INPUTS_NOT_FROZEN")
    actual = managed_precalculation_inputs({"upstream_dependencies": scope.get("_planned_dependencies", ())}, values)
    if actual != expected:
        raise ValueError("NDEF.PRECALCULATION_INPUTS_CHANGED")
    names = list(actual["camera_ids"])
    surface_path, masks = _resolved_inputs(scope, names, actual)
    image_sizes = _image_sizes(actual)
    _validate_surface(surface_path, names, image_sizes)
    _validate_masks(masks, image_sizes)
    overlay = _execution_overlay(values, staging, surface_path, masks[0].parent)
    # This is the sole scientific boundary.  It is intentionally after every
    # managed-input and output-routing check above.
    from ...api.ndef_dic import ndef_sparse_precalculation
    ndef_sparse_precalculation(overlay, write_case_artifacts=True)
    return validate_ndef_precalculation_outputs(staging, values, actual)


def _input_identities(plan: Mapping[str, Any], _values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = plan.get("scope", {}).get(INPUTS_KEY)
    if not isinstance(frozen, Mapping):
        raise ValueError("NDEF precalculation signature requires frozen managed inputs")
    return {"managed_ndef_precalculation_inputs": frozen}


def guarded_ndef_precalculation_action() -> TrustedAction:
    return TrustedAction(ACTION_ID, _run_ndef_precalculation, IMPLEMENTATION_ID,
                         output_contract=OUTPUT_CONTRACT, input_identities=_input_identities,
                         config_projection=precalculation_config_projection, output_paths=(_TRACKS, _SCALE))
