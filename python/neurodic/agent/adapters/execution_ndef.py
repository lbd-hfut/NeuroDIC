"""Guarded combined NDeF reference-surface execution.

The public NDeF surface API owns sparse training, dense continuation, fusion,
and serialization as one scientific operation.  This adapter therefore binds
all inputs before entering that API and publishes only the complete result.
It deliberately contains no native/Torch imports at module import time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...case_io import image_files
from ...ndef_paths import camera_name_from_label
from ..artifacts import content_identity, require_path_within
from ..errors import ControlPlaneError, ErrorRecord
from ..execution import ProducedArtifact, TrustedAction
from ..schemas import canonical_json


ACTION_ID = "ndef.combined_surface_call"
IMPLEMENTATION_ID = "neurodic.ndef.surface_combined/v1"
ROI_ACTION_ID = "ndef.roi.generate_call"
ROI_IMPLEMENTATION_ID = "neurodic.ndef.roi/v1"
INPUTS_KEY = "ndef_surface_inputs"

_PRETRAIN = "scientific/pretrain/surface"
_SURFACE = "scientific/surface"


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _calibration(values: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]], list[str]]:
    case = values.get("case", {})
    if not isinstance(case, Mapping) or not isinstance(case.get("root"), str) or not isinstance(case.get("calibration"), str):
        raise ValueError("NDeF surface requires a case root and explicit calibration path")
    root = Path(case["root"]).resolve()
    calibration = require_path_within(_path(root, case["calibration"]), root, require_exists=True)
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDeF calibration is not valid JSON") from error
    cameras = payload.get("cameras") or payload.get("scaled_cameras")
    points = payload.get("points3d") or payload.get("scaled_points3d")
    if not isinstance(cameras, list) or not cameras or not isinstance(points, list) or not points:
        raise ValueError("NDeF calibration lacks coherent cameras/points3d evidence")
    if not all(isinstance(camera, Mapping) for camera in cameras) or not all(isinstance(point, Mapping) for point in points):
        raise ValueError("NDeF calibration camera/point records are invalid")
    names = [camera_name_from_label(str(camera.get("label", "")), f"cam_{index}") for index, camera in enumerate(cameras)]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("NDeF calibration camera labels are not a unique ordered camera identity")
    return root, calibration, payload, list(cameras), list(points), names


def _roi_declaration(plan: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    matches = [item for item in plan.get("upstream_dependencies", ())
               if isinstance(item, Mapping) and item.get("dependency_id") == "ndef_roi"]
    if len(matches) != 1:
        raise ValueError("NDeF surface requires exactly one managed ndef_roi dependency")
    dependency = matches[0]
    signature = dependency.get("producer_signature", {})
    if (dependency.get("producer_action_id") != ROI_ACTION_ID or not isinstance(signature, Mapping)
            or signature.get("stage_id") != ROI_ACTION_ID
            or signature.get("implementation", {}).get("adapter") != ROI_IMPLEMENTATION_ID):
        raise ValueError("NDeF ROI dependency is not the required managed ROI producer")
    required = dependency.get("required_artifacts")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError("NDeF ROI dependency lacks required artifact declarations")
    by_path = {item.get("relative_path"): item for item in required if isinstance(item, Mapping)}
    expected = ["roi/mask_meta.json", *[f"roi/per_camera/{name}_mask.npy" for name in names]]
    # D2 ROI producers publish a camera-order bundle as a required scientific
    # artifact.  Keep compatibility with the D1 dependency fixtures that
    # predate that bundle while binding it whenever the producer declares it.
    if "roi/masks.npz" in by_path:
        expected.insert(0, "roi/masks.npz")
    if any(path not in by_path or not isinstance(by_path[path].get("identity"), Mapping) for path in expected):
        raise ValueError("NDeF ROI dependency does not bind every ordered per-camera mask and metadata")
    return {"producer_action": ROI_ACTION_ID, "implementation": ROI_IMPLEMENTATION_ID,
            "producer_signature": signature, "scope": dependency.get("scope", {}),
            "artifacts": [{"relative_path": path, "identity": by_path[path]["identity"]} for path in expected]}


def surface_config_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Only scientific values actually consumed by the combined public API."""
    model = values.get("surface_model", {})
    sparse = values.get("surface_training", {})
    dense = values.get("surface_dense_training", {})
    surface = values.get("surface", {})
    if not all(isinstance(item, Mapping) for item in (model, sparse, dense, surface)):
        raise ValueError("NDeF surface configuration blocks must be mappings")
    dense_enabled = bool(dense.get("enabled", int(dense.get("iterations", 0)) > 0))
    return {
        "model": {key: model.get(key) for key in ("hidden_dim", "pixel_layers", "camera_layers", "trunk_layers", "camera_embedding_dim", "positional_encoding_enabled", "positional_encoding_num_frequencies")},
        "sparse": {key: sparse.get(key) for key in ("pretrain_iterations", "pretrain_learning_rate", "weight_decay", "device")},
        "sparse_filter": {key: surface.get("sparse_filter", {}).get(key) for key in ("min_track_length", "max_reprojection_error", "radius_mad_thresh", "knn_k", "knn_mad_thresh")},
        "dense": {"enabled": dense_enabled, **{key: dense.get(key) for key in ("epochs", "samples_per_camera", "auto_batch", "auto_batch_start", "memory_fraction", "spacing_px", "patch_radius", "learning_rate", "anchor_weight", "min_valid_patch_ratio", "seed", "prediction_batch_size")}},
        "fusion": {key: surface.get(key, default) for key, default in (("fusion_relative_sample_spacing", 0.006), ("fusion_depth_tolerance_factor", 1.0), ("fusion_min_visible_cameras", 2), ("fusion_max_points", 100000), ("fusion_candidate_spacing_factor", 0.5), ("fusion_max_candidate_points", 1200000), ("fusion_seed", 17))},
        "hard_coded": {"positive_depth_mad_multiplier": 6.0, "query_grid_stride": 1, "calibration_reprojection_p95_px": surface.get("max_reprojection_p95_px", 5.0), "fresh_initialization": {"checkpoint": None}},
    }


def _config_type_error(path: str, expected: str, value: Any, *, reason: str | None = None) -> None:
    details = {"expected": expected, "actual_type": type(value).__name__, "actual_value": repr(value)}
    if reason is not None:
        details["reason"] = reason
    raise ControlPlaneError(ErrorRecord("NDEF.CONFIG_TYPE_INVALID", "NDeF surface configuration has an invalid scalar type or value",
                                        True, stage="ndef.surface", path=path, details=details))


def _config_mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        _config_type_error(key, "mapping", value)
    return value


def _config_int(path: str, value: Any, *, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        _config_type_error(path, "integer", value)
    if minimum is not None and value < minimum:
        _config_type_error(path, f"integer >= {minimum}", value, reason="production range")


def _config_number(path: str, value: Any, *, minimum: float | None = None,
                   maximum: float | None = None, exclusive_minimum: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _config_type_error(path, "finite numeric", value)
    numeric = float(value)
    if not math.isfinite(numeric):
        _config_type_error(path, "finite numeric", value)
    if minimum is not None and (numeric <= minimum if exclusive_minimum else numeric < minimum):
        operator = ">" if exclusive_minimum else ">="
        _config_type_error(path, f"finite numeric {operator} {minimum}", value, reason="production range")
    if maximum is not None and numeric > maximum:
        _config_type_error(path, f"finite numeric <= {maximum}", value, reason="production range")


def _config_bool(path: str, value: Any) -> None:
    if not isinstance(value, bool):
        _config_type_error(path, "boolean", value)


def _optional_int(section: Mapping[str, Any], section_name: str, key: str, **kwargs: Any) -> None:
    if key in section:
        _config_int(f"{section_name}.{key}", section[key], **kwargs)


def _optional_number(section: Mapping[str, Any], section_name: str, key: str, **kwargs: Any) -> None:
    if key in section:
        _config_number(f"{section_name}.{key}", section[key], **kwargs)


def validate_ndef_surface_config(values: Mapping[str, Any]) -> None:
    """Fail closed on values that would otherwise leak into native setters.

    This owns only the public combined-surface configuration contract.  It
    preserves numeric values exactly; in particular it never coerces strings
    such as ``"1e-06"`` into scientific settings.
    """
    model = _config_mapping(values, "surface_model")
    sparse = _config_mapping(values, "surface_training")
    dense = _config_mapping(values, "surface_dense_training")
    surface = _config_mapping(values, "surface")
    sparse_filter = surface.get("sparse_filter", {})
    if not isinstance(sparse_filter, Mapping):
        _config_type_error("surface.sparse_filter", "mapping", sparse_filter)

    for key in ("hidden_dim", "pixel_layers", "camera_layers", "trunk_layers", "camera_embedding_dim", "positional_encoding_num_frequencies"):
        _optional_int(model, "surface_model", key)
    if "positional_encoding_enabled" in model:
        _config_bool("surface_model.positional_encoding_enabled", model["positional_encoding_enabled"])

    _optional_int(sparse, "surface_training", "pretrain_iterations", minimum=0)
    _optional_number(sparse, "surface_training", "pretrain_learning_rate", minimum=0.0, exclusive_minimum=True)
    _optional_number(sparse, "surface_training", "weight_decay", minimum=0.0)
    _optional_number(sparse, "surface_training", "smoothness_weight", minimum=0.0)
    _optional_int(sparse, "surface_training", "smooth_samples_per_camera")
    if "device" in sparse and not isinstance(sparse["device"], str):
        _config_type_error("surface_training.device", "string", sparse["device"])

    if "enabled" in dense:
        _config_bool("surface_dense_training.enabled", dense["enabled"])
        dense_enabled = dense["enabled"]
    else:
        _optional_int(dense, "surface_dense_training", "iterations", minimum=0)
        dense_enabled = bool(dense.get("iterations", 0))
    if dense_enabled:
        _optional_int(dense, "surface_dense_training", "epochs", minimum=1)
        _optional_int(dense, "surface_dense_training", "samples_per_camera", minimum=1)
        if "auto_batch" in dense:
            _config_bool("surface_dense_training.auto_batch", dense["auto_batch"])
        _optional_int(dense, "surface_dense_training", "auto_batch_start", minimum=1)
        _optional_number(dense, "surface_dense_training", "memory_fraction", minimum=0.0, maximum=1.0, exclusive_minimum=True)
        _optional_int(dense, "surface_dense_training", "spacing_px", minimum=1)
        _optional_int(dense, "surface_dense_training", "patch_radius", minimum=0)
        _optional_number(dense, "surface_dense_training", "learning_rate", minimum=0.0, exclusive_minimum=True)
        _optional_number(dense, "surface_dense_training", "anchor_weight", minimum=0.0)
        _optional_number(dense, "surface_dense_training", "min_valid_patch_ratio", minimum=0.0, maximum=1.0, exclusive_minimum=True)
        _optional_int(dense, "surface_dense_training", "seed")
        _optional_int(dense, "surface_dense_training", "prediction_batch_size", minimum=1)

    _optional_int(sparse_filter, "surface.sparse_filter", "min_track_length")
    if "max_reprojection_error" in sparse_filter and sparse_filter["max_reprojection_error"] is not None:
        _config_number("surface.sparse_filter.max_reprojection_error", sparse_filter["max_reprojection_error"])
    for key in ("radius_mad_thresh", "knn_mad_thresh"):
        _optional_number(sparse_filter, "surface.sparse_filter", key)
    _optional_int(sparse_filter, "surface.sparse_filter", "knn_k")

    _optional_int(surface, "surface", "max_points", minimum=0)
    _optional_number(surface, "surface", "max_reprojection_p95_px")
    for key in ("fusion_relative_sample_spacing", "fusion_depth_tolerance_factor", "fusion_candidate_spacing_factor"):
        _optional_number(surface, "surface", key)
    _optional_int(surface, "surface", "fusion_min_visible_cameras", minimum=1)
    _optional_int(surface, "surface", "fusion_max_points", minimum=0)
    _optional_int(surface, "surface", "fusion_max_candidate_points", minimum=0)
    _optional_int(surface, "surface", "fusion_seed")


def managed_surface_inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze exact production inputs without using a filesystem camera sort."""
    root, calibration, payload, cameras, points, names = _calibration(values)
    calibration_dir = calibration.parent
    observations = require_path_within(calibration_dir / "observations.npz", root, require_exists=True)
    pairs = require_path_within(calibration_dir / "camera_pairs.json", root, require_exists=True)
    try:
        pair_data = json.loads(pairs.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDeF camera_pairs evidence is not valid JSON") from error
    if pair_data.get("camera_names") != names:
        raise ValueError("NDeF camera_pairs order does not equal calibration order")
    image_root = require_path_within(root / str(values.get("case", {}).get("images", "images")), root, require_exists=True)
    references = []
    for name in names:
        reference = require_path_within(image_files(image_root / name)[0], root, require_exists=True)
        references.append({"camera_id": name, "path": str(reference.relative_to(root)), **content_identity(reference).to_dict()})
    camera_models = [{key: camera.get(key) for key in ("label", "K", "R", "t", "distortion", "image_width", "image_height")} for camera in cameras]
    scale = {key: payload.get(key) for key in ("sfm_to_world_scale", "world_to_sfm_scale", "sfm_to_world_rotation", "sfm_to_world_translation")}
    dense = values.get("surface_dense_training", {})
    if not isinstance(dense, Mapping):
        raise ValueError("NDeF dense configuration is invalid")
    dense_enabled = bool(dense.get("enabled", int(dense.get("iterations", 0)) > 0))
    if dense_enabled and bool(dense.get("auto_batch", True)):
        raise ValueError("NDEF.AUTO_BATCH_UNRESOLVED")
    batch = {"mode": "manual", "samples_per_camera": int(dense.get("samples_per_camera", 10000))} if dense_enabled else {"mode": "not_applicable"}
    if dense_enabled and batch["samples_per_camera"] < 1:
        raise ValueError("NDeF manual dense batch must be positive")
    return {
        "schema_version": "neurodic.ndef.surface-inputs/v1",
        "calibration": {"path": str(calibration.relative_to(root)), **content_identity(calibration).to_dict()},
        "points3d_identity": {"algorithm": "sha256-canonical-json/v1", "digest": _digest(points)},
        "observations": {"path": str(observations.relative_to(root)), **content_identity(observations).to_dict()},
        "camera_pairs": {"path": str(pairs.relative_to(root)), **content_identity(pairs).to_dict()},
        "camera_ids": list(names), "camera_models_identity": {"algorithm": "sha256-canonical-json/v1", "digest": _digest(camera_models)},
        "reference_images": references, "coordinate_convention": "calibration_world_frame/v1",
        "embedded_scale_identity": {"algorithm": "sha256-canonical-json/v1", "digest": _digest(scale)},
        "roi_dependency": _roi_declaration(plan, names), "batch_contract": batch,
    }


def surface_readiness(values: Mapping[str, Any], upstream_dependencies: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Native-free fail-closed readiness used by planning and current-case checks."""
    issues: list[tuple[str, str]] = []
    case = values.get("case", {})
    root = Path(str(case.get("root", "."))).resolve()
    calibration_value = case.get("calibration")
    if not isinstance(calibration_value, str) or not _path(root, calibration_value).is_file():
        issues.append(("NDEF.CALIBRATION_NOT_MANAGED", "Configured NDeF calibration evidence is unavailable"))
    if not any(isinstance(item, Mapping) and item.get("dependency_id") == "ndef_roi" for item in upstream_dependencies):
        issues.append(("NDEF.ROI_NOT_MANAGED", "NDeF surface requires an explicit managed ROI dependency"))
    dense = values.get("surface_dense_training", {})
    enabled = isinstance(dense, Mapping) and bool(dense.get("enabled", int(dense.get("iterations", 0)) > 0))
    if enabled and bool(dense.get("auto_batch", True)):
        issues.append(("NDEF.AUTO_BATCH_UNRESOLVED", "Managed NDeF surface execution requires an explicit manual dense batch"))
    if root.name == "CylinderDIC" and enabled:
        issues.append(("NDEF.REAL_SMOKE_UNBOUNDED", "Current CylinderDIC full-ROI stride-one surface workload is not a bounded smoke"))
    return issues


def _outputs() -> list[ProducedArtifact]:
    return [ProducedArtifact(f"{_PRETRAIN}/surface_pretrain.npz", "ndef_surface_pretrain", "neurodic.ndef.surface-pretrain/v1"),
            ProducedArtifact(f"{_PRETRAIN}/surface_pretrain_meta.json", "ndef_surface_pretrain_metadata", "json/v1"),
            ProducedArtifact(f"{_SURFACE}/surface_dense_samples.npz", "ndef_surface_dense_samples", "neurodic.ndef.surface-dense-samples/v1"),
            ProducedArtifact(f"{_SURFACE}/surface_dense_field.npz", "ndef_surface_dense_field", "neurodic.ndef.surface-dense-field/v1"),
            ProducedArtifact(f"{_SURFACE}/deformation_surface_dataset.npz", "ndef_deformation_surface", "neurodic.ndef.deformation-surface/v1"),
            ProducedArtifact(f"{_SURFACE}/surface_dense_meta.json", "ndef_surface_dense_metadata", "json/v1")]


def _npz(path: Path):
    import numpy as np
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"NDeF surface NPZ is invalid: {path.name}") from error


def _finite(value: Any) -> bool:
    import numpy as np
    return np.issubdtype(np.asarray(value).dtype, np.number) and bool(np.all(np.isfinite(value)))


def validate_ndef_surface_outputs(root: Path, values: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[ProducedArtifact]:
    """Validate the complete public serializer contract before publication."""
    import numpy as np
    outputs = _outputs()
    for artifact in outputs:
        path = require_path_within(root / artifact.path, root, require_exists=True)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"NDeF surface required output is missing or empty: {artifact.path}")
    names = list(inputs["camera_ids"]); cameras = len(names)
    with _npz(root / outputs[0].path) as data:
        required = {"sparse_uv", "sparse_camera", "sparse_depth", "sparse_prediction", "query_uv", "query_camera", "query_depth", "roi_uv_bounds", "depth_mean", "depth_std"}
        if required - set(data.files): raise ValueError("NDeF pretrain NPZ lacks required keys")
        sparse = len(data["sparse_camera"]); query = len(data["query_camera"])
        if (np.asarray(data["sparse_uv"]).shape != (sparse, 2) or np.asarray(data["sparse_depth"]).shape != (sparse,)
                or np.asarray(data["sparse_prediction"]).shape != (sparse,) or np.asarray(data["query_uv"]).shape != (query, 2)
                or np.asarray(data["query_depth"]).shape != (query,) or np.asarray(data["roi_uv_bounds"]).shape != (cameras, 4)
                or not _finite(data["sparse_uv"]) or not _finite(data["sparse_depth"]) or not _finite(data["sparse_prediction"])
                or not _finite(data["query_uv"]) or not _finite(data["query_depth"]) or not _finite(data["roi_uv_bounds"])
                or not np.issubdtype(data["sparse_camera"].dtype, np.integer) or not np.issubdtype(data["query_camera"].dtype, np.integer)
                or np.any((data["sparse_camera"] < 0) | (data["sparse_camera"] >= cameras)) or np.any((data["query_camera"] < 0) | (data["query_camera"] >= cameras))
                or np.asarray(data["depth_mean"]).shape != () or np.asarray(data["depth_std"]).shape != () or not _finite(data["depth_mean"]) or not _finite(data["depth_std"]) or float(data["depth_std"]) <= 0):
            raise ValueError("NDeF pretrain NPZ violates the output contract")
    with _npz(root / outputs[2].path) as data:
        required = {"uv", "camera", "targets", "depth", "world", "history", "history_columns", "roi_uv_bounds", "depth_mean", "depth_std"}
        if required - set(data.files): raise ValueError("NDeF dense samples NPZ lacks required keys")
        count = len(data["camera"]); history = np.asarray(data["history"])
        if (np.asarray(data["uv"]).shape != (count, 2) or np.asarray(data["targets"]).shape != (count, 2) or np.asarray(data["depth"]).shape != (count,) or np.asarray(data["world"]).shape != (count, 3)
                or history.ndim != 2 or history.shape[1] != 3 or not all(_finite(data[key]) for key in ("uv", "depth", "world", "history", "roi_uv_bounds", "depth_mean", "depth_std"))
                or not np.issubdtype(data["camera"].dtype, np.integer) or np.any((data["camera"] < 0) | (data["camera"] >= cameras))
                or np.asarray(data["roi_uv_bounds"]).shape != (cameras, 4) or float(data["depth_std"]) <= 0):
            raise ValueError("NDeF dense samples NPZ violates the output contract")
    with _npz(root / outputs[3].path) as data:
        required = {"uv", "camera", "depth", "world", "grid_stride", "roi_uv_bounds", "depth_mean", "depth_std"}
        if required - set(data.files): raise ValueError("NDeF dense field NPZ lacks required keys")
        count = len(data["camera"])
        if (np.asarray(data["uv"]).shape != (count, 2) or np.asarray(data["depth"]).shape != (count,) or np.asarray(data["world"]).shape != (count, 3)
                or int(np.asarray(data["grid_stride"])) != 1 or np.asarray(data["roi_uv_bounds"]).shape != (cameras, 4)
                or not all(_finite(data[key]) for key in ("uv", "depth", "world", "roi_uv_bounds", "depth_mean", "depth_std"))
                or not np.issubdtype(data["camera"].dtype, np.integer) or np.any((data["camera"] < 0) | (data["camera"] >= cameras)) or float(data["depth_std"]) <= 0):
            raise ValueError("NDeF dense field NPZ violates the output contract")
    with _npz(root / outputs[4].path) as data:
        required = {"points", "normals", "source_camera", "visibility_mask", "projected_uv", "projected_depth", "depth_abs_error", "visible_counts", "cam_names"}
        if required - set(data.files): raise ValueError("NDeF final surface NPZ lacks required keys")
        points, normals, source, visible = (np.asarray(data[key]) for key in ("points", "normals", "source_camera", "visibility_mask"))
        count = len(points)
        if (count < 1 or points.dtype != np.float32 or points.shape != (count, 3) or normals.shape != (count, 3)
                or source.shape != (count,) or visible.shape != (count, cameras) or np.asarray(data["projected_uv"]).shape != (count, cameras, 2)
                or np.asarray(data["projected_depth"]).shape != (count, cameras) or np.asarray(data["depth_abs_error"]).shape != (count, cameras)
                or np.asarray(data["visible_counts"]).shape != (count,) or list(np.asarray(data["cam_names"]).astype(str)) != names
                or not all(_finite(data[key]) for key in ("points", "normals", "projected_uv"))
                or not np.issubdtype(source.dtype, np.integer) or np.any((source < 0) | (source >= cameras))
                or not np.array_equal(np.asarray(data["visible_counts"]), visible.sum(axis=1))):
            raise ValueError("NDeF final surface NPZ violates the output contract")
        minimum = int(surface_config_projection(values)["fusion"]["fusion_min_visible_cameras"])
        if np.any(np.asarray(data["visible_counts"]) < minimum): raise ValueError("NDeF final surface has insufficient visible cameras")
        if np.any(np.asarray(data["depth_abs_error"])[visible] < 0) or not np.all(np.isfinite(np.asarray(data["depth_abs_error"])[visible])) or not np.all(np.isfinite(np.asarray(data["projected_depth"])[visible])):
            raise ValueError("NDeF final surface visible projection evidence is invalid")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-8) or np.any(lengths > 1.001): raise ValueError("NDeF final surface normals are implausible")
        maximum = int(surface_config_projection(values)["fusion"]["fusion_max_points"])
        if count > maximum: raise ValueError("NDeF final surface point count exceeds configured maximum")
    for artifact in (outputs[1], outputs[5]):
        try: metadata = json.loads((root / artifact.path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error: raise ValueError(f"NDeF metadata is invalid: {artifact.path}") from error
        if not isinstance(metadata, Mapping): raise ValueError(f"NDeF metadata is not an object: {artifact.path}")
    return outputs


def _resolved_roi(scope: Mapping[str, Any], names: Sequence[str]) -> Path:
    dependencies = scope.get("_managed_dependencies", {})
    dependency = dependencies.get("ndef_roi") if isinstance(dependencies, Mapping) else None
    if not isinstance(dependency, Mapping): raise ValueError("NDeF surface did not receive its managed ROI dependency")
    signature = dependency.get("producer_signature", {})
    if signature.get("stage_id") != ROI_ACTION_ID or signature.get("implementation", {}).get("adapter") != ROI_IMPLEMENTATION_ID:
        raise ValueError("NDeF surface ROI producer is not trusted")
    files = dependency.get("files", {})
    if not isinstance(files, Mapping) or "mask_meta.json" not in files or any(f"{name}_mask.npy" not in files for name in names):
        raise ValueError("NDeF surface ROI dependency does not contain every required mask")
    masks = [Path(files[f"{name}_mask.npy"]).resolve() for name in names]
    if len({path.parent for path in masks}) != 1 or any(not path.is_file() for path in masks):
        raise ValueError("NDeF surface managed ROI masks do not share one explicit directory")
    return masks[0].parent


def _execution_overlay(values: Mapping[str, Any], staging: Path, roi_root: Path) -> dict[str, Any]:
    overlay = copy.deepcopy(dict(values)); case = overlay.setdefault("case", {})
    case["masks"] = str(roi_root)
    overlay["output"] = {"result": str(staging / "scientific"), "visualization": str(staging / "visualization"), "ndef_subdir": None}
    return overlay


def _run_ndef_surface(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    # Keep malformed configuration out of the public scientific API and its
    # pybind setters.  This is intentionally before any native/Torch import.
    validate_ndef_surface_config(values)
    expected = scope.get(INPUTS_KEY)
    if not isinstance(expected, Mapping): raise ValueError("NDeF surface execution lacks a frozen managed input contract")
    # The re-derived content snapshot makes a post-plan content change fail
    # before the public scientific boundary is imported.
    actual = managed_surface_inputs({"upstream_dependencies": scope.get("_planned_dependencies", ())}, values)
    if actual != expected: raise ValueError("NDeF surface managed inputs changed after planning")
    names = actual["camera_ids"]; roi_root = _resolved_roi(scope, names)
    overlay = _execution_overlay(values, staging, roi_root)
    from ...api.ndef_surface import pretrain_ndef_surface
    pretrain_ndef_surface(overlay)
    return validate_ndef_surface_outputs(staging, values, actual)


def _input_identities(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = plan.get("scope", {}).get(INPUTS_KEY)
    if not isinstance(frozen, Mapping): raise ValueError("NDeF surface signature requires frozen managed inputs")
    return {"managed_surface_inputs": frozen}


def guarded_ndef_surface_action() -> TrustedAction:
    return TrustedAction(ACTION_ID, _run_ndef_surface, IMPLEMENTATION_ID,
                         output_contract="neurodic.ndef.surface-combined-artifacts/v1",
                         input_identities=_input_identities, config_projection=surface_config_projection)
