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


def ndef_reprojection_gate(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Compute the read-only production calibration reprojection gate.

    This is the same camera/point/observation evidence consumed by the public
    surface entry point.  It performs no calibration, optimization, or output
    writes; callers receive the exact mean/median/p95 values and the configured
    p95 threshold.
    """
    values = _mapping(config)
    case = values.get("case", {})
    root = Path(str(case.get("root", "."))).resolve()
    calibration_value = Path(str(case.get("calibration", "result/calibration/calibration_result_scaled.json")))
    calibration = calibration_value if calibration_value.is_absolute() else root / calibration_value
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    cameras = payload.get("cameras") or payload.get("scaled_cameras")
    points = payload.get("points3d") or payload.get("scaled_points3d")
    observations_path = calibration.parent / "observations.npz"
    if not isinstance(cameras, list) or not isinstance(points, list) or not observations_path.is_file():
        return {"pass": False, "reason": "incomplete calibration package", "threshold_p95_px": float(values.get("surface", {}).get("max_reprojection_p95_px", 5.0))}
    with np.load(observations_path, allow_pickle=False) as observed:
        required = {"point_indices", "cam_indices", "uv"}
        if required - set(observed.files):
            return {"pass": False, "reason": "observations.npz lacks point_indices/cam_indices/uv",
                    "threshold_p95_px": float(values.get("surface", {}).get("max_reprojection_p95_px", 5.0))}
        point_indices = np.asarray(observed["point_indices"], dtype=np.int64)
        camera_indices = np.asarray(observed["cam_indices"], dtype=np.int64)
        uv = np.asarray(observed["uv"], dtype=np.float64)
    xyz = np.asarray([point["xyz"] for point in points], dtype=np.float64)
    if (point_indices.ndim != 1 or camera_indices.shape != point_indices.shape or uv.shape != (len(point_indices), 2)
            or np.any(point_indices < 0) or np.any(point_indices >= len(xyz))
            or np.any(camera_indices < 0) or np.any(camera_indices >= len(cameras))):
        return {"pass": False, "reason": "observation arrays are incoherent",
                "threshold_p95_px": float(values.get("surface", {}).get("max_reprojection_p95_px", 5.0))}
    import cv2
    errors: list[np.ndarray] = []
    for camera_index, camera in enumerate(cameras):
        selected = np.flatnonzero(camera_indices == camera_index)
        if not len(selected):
            continue
        rotation = np.asarray(camera["R"], dtype=np.float64)
        translation = np.asarray(camera["t"], dtype=np.float64).reshape(3, 1)
        intrinsic = np.asarray(camera["K"], dtype=np.float64)
        distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
        projected, _ = cv2.projectPoints(xyz[point_indices[selected]].reshape(-1, 1, 3),
                                         cv2.Rodrigues(rotation)[0], translation, intrinsic, distortion)
        errors.append(np.linalg.norm(projected.reshape(-1, 2) - uv[selected], axis=1))
    threshold = float(values.get("surface", {}).get("max_reprojection_p95_px", 5.0))
    if not errors:
        return {"pass": False, "reason": "no observations for any camera", "threshold_p95_px": threshold}
    error = np.concatenate(errors)
    metrics = {"mean": float(error.mean()), "median": float(np.median(error)), "p95": float(np.percentile(error, 95))}
    return {**metrics, "threshold_p95_px": threshold, "pass": bool(np.isfinite(error).all() and metrics["p95"] <= threshold)}


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
