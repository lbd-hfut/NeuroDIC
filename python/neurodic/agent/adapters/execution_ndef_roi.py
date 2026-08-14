"""Guarded managed NDeF ROI generation.

The ROI algorithm remains in :mod:`neurodic.ndef_roi`; this adapter owns only
the explicit input binding, staging, structural validation, and provenance
boundary required by the control plane.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...case_io import image_files
from ...ndef_paths import camera_name_from_label
from ...ndef_preflight import inspect_ndef_preflight, ndef_reprojection_gate
from ...ndef_roi import NDeFROIOptions
from ..artifacts import content_identity, require_path_within
from ..execution import ProducedArtifact, TrustedAction


ACTION_ID = "ndef.roi.generate_call"
IMPLEMENTATION_ID = "neurodic.ndef.roi/v1"
OUTPUT_CONTRACT = "neurodic.ndef.roi-artifacts/v1"
INPUTS_KEY = "ndef_roi_inputs"

_REQUIRED = (
    ProducedArtifact("roi/masks.npz", "ndef_roi_bundle", "neurodic.ndef.roi.bundle/v1"),
    ProducedArtifact("roi/mask_meta.json", "ndef_roi_metadata", "json/v1"),
)


def _root_and_calibration(values: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any], list[str]]:
    case = values.get("case", {})
    if not isinstance(case, Mapping) or not isinstance(case.get("root"), str):
        raise ValueError("NDeF ROI requires an explicit case root")
    root = Path(case["root"]).resolve()
    calibration_value = case.get("calibration", "result/calibration/calibration_result_scaled.json")
    calibration = Path(str(calibration_value))
    calibration = calibration if calibration.is_absolute() else root / calibration
    calibration = require_path_within(calibration, root, require_exists=True)
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDeF ROI calibration is not valid JSON") from error
    cameras = payload.get("cameras") or payload.get("scaled_cameras")
    if not isinstance(cameras, list) or len(cameras) < 2 or not all(isinstance(item, Mapping) for item in cameras):
        raise ValueError("NDeF ROI calibration lacks a camera array")
    names = [camera_name_from_label(str(item.get("label", "")), f"cam_{index}")
             for index, item in enumerate(cameras)]
    if not names or len(set(names)) != len(names):
        raise ValueError("NDeF ROI calibration camera identities are not unique")
    return root, calibration, payload, names


def _options(values: Mapping[str, Any]) -> NDeFROIOptions:
    configured = values.get("ndef_roi", values.get("roi", {}))
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ValueError("NDeF ROI options must be a mapping")
    allowed = set(NDeFROIOptions.__dataclass_fields__)
    unknown = sorted(set(configured) - allowed)
    if unknown:
        raise ValueError(f"NDeF ROI options contain unsupported fields: {unknown}")
    try:
        return NDeFROIOptions(**{key: configured[key] for key in configured})
    except (TypeError, ValueError) as error:
        raise ValueError("NDeF ROI options are invalid") from error


def roi_config_projection(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze every option consumed by ``generate_ndef_roi``."""
    return {"options": copy.deepcopy(vars(_options(values)))}


def _ordered_inputs(values: Mapping[str, Any]) -> dict[str, Any]:
    root, calibration, payload, names = _root_and_calibration(values)
    calibration_dir = calibration.parent
    observations = require_path_within(calibration_dir / "observations.npz", root, require_exists=True)
    pairs = require_path_within(calibration_dir / "camera_pairs.json", root, require_exists=True)
    summary = require_path_within(calibration_dir / "summary.json", root, require_exists=True)
    try:
        pair_data = json.loads(pairs.read_text(encoding="utf-8"))
        summary_data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDeF ROI calibration package metadata is invalid") from error
    if pair_data.get("camera_names") != names:
        raise ValueError("NDeF ROI camera_pairs.json order does not equal calibration order")
    neighbors = pair_data.get("neighbors")
    if not isinstance(neighbors, Mapping) or set(neighbors) != set(names):
        raise ValueError("NDeF ROI camera topology is not complete")
    if any(not isinstance(neighbors[name], list) or not neighbors[name] for name in names):
        raise ValueError("NDeF ROI camera topology has an empty neighbor list")
    if any(item not in names for listed in neighbors.values() for item in listed):
        raise ValueError("NDeF ROI camera topology references an unknown camera")
    image_root = require_path_within(root / str(values.get("case", {}).get("images", "images")), root, require_exists=True)
    references: list[dict[str, Any]] = []
    summary_paths = summary_data.get("image_paths")
    if not isinstance(summary_paths, list):
        raise ValueError("NDeF ROI calibration summary lacks image_paths")
    summary_by_name = {Path(str(item)).parent.name: Path(str(item)) for item in summary_paths}
    for name in names:
        candidates = image_files(image_root / name)
        reference = candidates[0]
        # The public ROI function consumes the summary-selected reference. It
        # must agree with the deterministic image resolver used by the agent.
        summary_reference = summary_by_name.get(name)
        if summary_reference is None or summary_reference.resolve() != reference.resolve():
            raise ValueError(f"NDeF ROI summary reference does not match ordered image resolver for {name}")
        references.append({"camera_id": name, "path": str(reference.relative_to(root)),
                           "identity": content_identity(reference).to_dict()})
    return {
        "calibration": {"path": str(calibration.relative_to(root)), **content_identity(calibration).to_dict()},
        "observations": {"path": str(observations.relative_to(root)), **content_identity(observations).to_dict()},
        "camera_pairs": {"path": str(pairs.relative_to(root)), **content_identity(pairs).to_dict()},
        "camera_ids": list(names), "camera_topology": copy.deepcopy(pair_data),
        "reference_images": references,
        "options": roi_config_projection(values)["options"],
        "coordinate_convention": "calibration_world_frame/v1",
        "scale_identity": {key: payload.get(key) for key in ("sfm_to_world_scale", "world_to_sfm_scale",
                                                               "sfm_to_world_rotation", "sfm_to_world_translation")},
    }


def managed_roi_inputs(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return _ordered_inputs(values)


def roi_readiness(values: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Native-free readiness and the existing calibration reprojection gate."""
    issues: list[tuple[str, str]] = []
    try:
        preflight = inspect_ndef_preflight(values)
        if not preflight.get("ready_for_roi"):
            issues.append(("NDEF.ROI_INPUTS_NOT_READY", "NDeF ROI calibration package, observations, topology, or references are not ready"))
        gate = ndef_reprojection_gate(values)
        if not gate.get("pass"):
            issues.append(("NDEF.CALIBRATION_REPROJECTION_GATE", "NDeF calibration reprojection gate did not pass"))
    except (OSError, ValueError, KeyError):
        issues.append(("NDEF.ROI_INPUTS_NOT_READY", "NDeF ROI preflight could not resolve the explicit calibration package"))
    try:
        managed_roi_inputs(values)
    except (OSError, ValueError):
        issues.append(("NDEF.ROI_INPUTS_NOT_READY", "NDeF ROI managed input identities are incomplete"))
    return issues


def _required_outputs(names: Sequence[str]) -> list[ProducedArtifact]:
    return [*_REQUIRED, *[
        ProducedArtifact(f"roi/per_camera/{name}_mask.npy", f"ndef_roi_mask.{name}", "npy/v1")
        for name in names
    ]]


def validate_ndef_roi_outputs(root: Path, values: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[ProducedArtifact]:
    """Validate production ROI serialization without accepting legacy fallback."""
    names = list(inputs["camera_ids"])
    outputs = _required_outputs(names)
    roi = root / "roi"
    try:
        with np.load(roi / "masks.npz", allow_pickle=False) as payload:
            required = {"cam_names", "masks"}
            if required - set(payload.files):
                raise ValueError("ROI bundle lacks cam_names or masks")
            cam_names = [str(item) for item in np.asarray(payload["cam_names"]).tolist()]
            bundle = np.asarray(payload["masks"])
    except (OSError, ValueError) as error:
        raise ValueError(f"NDeF ROI bundle is invalid: {error}") from error
    if cam_names != names or bundle.ndim != 3 or bundle.shape[0] != len(names) or bundle.dtype != np.dtype(bool):
        raise ValueError("NDeF ROI bundle camera order/count/dtype is invalid")
    if not np.isfinite(bundle.astype(np.float32)).all():
        raise ValueError("NDeF ROI bundle contains non-finite values")
    for index, name in enumerate(names):
        path = roi / "per_camera" / f"{name}_mask.npy"
        try:
            mask = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"NDeF ROI per-camera mask is invalid: {name}") from error
        if mask.dtype != np.dtype(bool) or mask.shape != bundle.shape[1:] or not np.array_equal(mask, bundle[index]):
            raise ValueError(f"NDeF ROI per-camera mask does not match bundle slice: {name}")
    try:
        metadata = json.loads((roi / "mask_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("NDeF ROI metadata is invalid") from error
    def finite_json(value: Any) -> bool:
        if isinstance(value, float):
            return bool(np.isfinite(value))
        if isinstance(value, Mapping):
            return all(finite_json(key) and finite_json(item) for key, item in value.items())
        if isinstance(value, list):
            return all(finite_json(item) for item in value)
        return True

    records = metadata.get("cameras") if isinstance(metadata, Mapping) else None
    if (not isinstance(metadata, Mapping) or metadata.get("schema_version") != 1
            or not isinstance(records, list) or [item.get("camera_name") for item in records] != names
            or any(item.get("camera_index") != index for index, item in enumerate(records))
            or not finite_json(metadata)):
        raise ValueError("NDeF ROI metadata camera order/count is invalid")
    for item in records:
        for key in ("mask_pixels", "image_pixels", "shared_observation_union", "points_after_outlier_filter"):
            if not isinstance(item.get(key), int) or item[key] < 0:
                raise ValueError("NDeF ROI metadata contains invalid counts")
        for key in ("mask_fraction",):
            if not isinstance(item.get(key), (int, float)) or not np.isfinite(item[key]) or not 0 <= item[key] <= 1:
                raise ValueError("NDeF ROI metadata contains invalid scalar evidence")
    return outputs


def _run_ndef_roi(values: Mapping[str, Any], staging: Path, _scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    root, _calibration, _payload, names = _root_and_calibration(values)
    inputs = managed_roi_inputs(values)
    expected = _scope.get(INPUTS_KEY)
    if not isinstance(expected, Mapping) or expected != inputs:
        raise ValueError("NDeF ROI managed inputs changed after planning")
    gate = ndef_reprojection_gate(values)
    if not gate.get("pass"):
        raise ValueError("NDEF.CALIBRATION_REPROJECTION_GATE")
    from ...ndef_roi import generate_ndef_roi
    generate_ndef_roi(root, options=_options(values), result_root=staging / "roi", visualization_root=staging / "visualization")
    return validate_ndef_roi_outputs(staging, values, inputs)


def _input_identities(_plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _plan.get("scope", {}).get(INPUTS_KEY)
    if not isinstance(frozen, Mapping):
        raise ValueError("NDeF ROI signature requires frozen managed inputs")
    return {"managed_ndef_roi_inputs": frozen}


def guarded_ndef_roi_action() -> TrustedAction:
    return TrustedAction(ACTION_ID, _run_ndef_roi, IMPLEMENTATION_ID,
                         output_contract=OUTPUT_CONTRACT, input_identities=_input_identities,
                         config_projection=roi_config_projection)
