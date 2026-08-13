"""Read-only input-contract checks for a multi-view NDeF run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_config
from .case_io import named_multiview_image_pairs
from .ndef_paths import camera_name_from_label, ndef_run_roots


def _mapping(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else config


def inspect_ndef_preflight(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Return a non-mutating readiness report before ROI or GPU work starts.

    The surface stage deliberately requires the calibration export produced by
    :func:`run_multiview_case`: camera directory labels, structured sparse
    points, observations, summary, and camera-pair topology must agree.
    """
    values = _mapping(config)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    calibration_path = Path(case.get("calibration", "result/calibration/calibration_result_scaled.json"))
    calibration_path = calibration_path if calibration_path.is_absolute() else root / calibration_path
    calibration_dir = calibration_path.parent
    result_root, visualization_root = ndef_run_roots(root, values)
    report: dict[str, Any] = {
        "case_root": str(root), "calibration": str(calibration_path),
        "run_result_root": str(result_root), "run_visualization_root": str(visualization_root),
        "checks": [], "ready_for_roi": False, "ready_for_surface": False,
    }

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    check("case_root", root.is_dir(), str(root))
    check("calibration_file", calibration_path.is_file(), str(calibration_path))
    if not calibration_path.is_file():
        return report
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        check("calibration_json", False, str(error))
        return report
    cameras = calibration.get("cameras", calibration.get("scaled_cameras", []))
    points = calibration.get("points3d", calibration.get("scaled_points3d", []))
    check("camera_count", len(cameras) >= 2, f"count={len(cameras)}")
    labels = [str(item.get("label", "")) for item in cameras]
    names = [camera_name_from_label(label, f"cam_{index}") for index, label in enumerate(labels)]
    labels_are_camera_dirs = bool(labels) and all(label and name for label, name in zip(labels, names))
    check("camera_labels", labels_are_camera_dirs,
          "camera names resolved as " + repr(names[:2]))
    structured_points = bool(points) and all(isinstance(point, dict) and "xyz" in point for point in points)
    check("sparse_point_schema", structured_points,
          "points3d entries must contain xyz, observations, and reprojection_error")
    if structured_points:
        complete_points = all("observations" in point and "reprojection_error" in point for point in points)
        check("sparse_point_diagnostics", complete_points, f"count={len(points)}")
    else:
        check("sparse_point_diagnostics", False, "not checked because point schema is invalid")
    for filename in ("observations.npz", "summary.json", "camera_pairs.json"):
        path = calibration_dir / filename
        check(filename, path.is_file(), str(path))
    observation_path = calibration_dir / "observations.npz"
    if observation_path.is_file():
        try:
            with np.load(observation_path) as payload:
                required = {"point_indices", "cam_indices", "uv"}
                check("observation_schema", required <= set(payload.files),
                      f"fields={sorted(payload.files)}")
        except (OSError, ValueError) as error:
            check("observation_schema", False, str(error))
    image_root = root / case.get("images", "images")
    try:
        if labels_are_camera_dirs:
            named_multiview_image_pairs(image_root, names)
        image_ok = labels_are_camera_dirs
    except (FileNotFoundError, ValueError):
        image_ok = False
    check("reference_images", image_ok, str(image_root))
    mask_value = case.get("masks")
    mask_root = (Path(mask_value) if mask_value is not None else result_root / "roi" / "per_camera")
    if not mask_root.is_absolute(): mask_root = root / mask_root
    masks_ok = labels_are_camera_dirs and all((mask_root / f"{name}_mask.npy").is_file() for name in names)
    check("roi_masks", masks_ok, str(mask_root))
    report["ready_for_roi"] = all(item["ok"] for item in report["checks"]
                                   if item["name"] in {"case_root", "camera_count", "camera_labels", "reference_images", "observations.npz", "summary.json", "camera_pairs.json", "observation_schema"})
    report["ready_for_surface"] = report["ready_for_roi"] and structured_points and all(
        item["ok"] for item in report["checks"] if item["name"] in {"sparse_point_diagnostics", "roi_masks"})
    return report
