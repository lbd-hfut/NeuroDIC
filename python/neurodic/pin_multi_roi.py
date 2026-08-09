"""Pairwise SIFT ROI generation for the independent ``pin_multi_slover`` route.

This module stays independent of :mod:`neurodic.ndef_roi`.  Inputs are the
reference-time images of selected camera pairs; outputs are pair-local masks
and diagnostics under ``result/pin_multi_slover/pair_roi/<pair_id>/``.
NDeF masks under ``result/mask/per_camera`` are never read or written here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class PINMultiPairROIOptions:
    """Configuration contract for reference-time pairwise SIFT ROI generation."""

    feature_method: Literal["sift"] = "sift"
    max_features: int = 12_000
    match_ratio: float = 0.75
    mutual_check: bool = True
    ransac_reprojection_threshold_px: float = 3.0
    min_matches: int = 20
    support: Literal["convex_hull", "alpha_shape"] = "convex_hull"
    alpha_radius_scale: float = 8.0
    erode_pixels: int = 0
    min_mask_area_ratio: float = 0.01


@dataclass(frozen=True)
class PINMultiPairSelectionOptions:
    """Camera-pair selection for the pairwise multi-camera PIN route."""

    mode: Literal["auto_spatial_neighbors", "manual"] = "auto_spatial_neighbors"
    wrap: bool = True
    manual: tuple[tuple[str, str], ...] = ()
    camera_pairs_json: str | Path | None = None


@dataclass
class PINMultiROIResult:
    """Batch pair-ROI generation result; skipped pairs stay structured."""

    output_root: Path
    pairs: list[tuple[str, str]]
    pair_ids: list[str]
    results: list[dict[str, Any]]
    manifest_path: Path

    @property
    def skipped(self) -> list[dict[str, Any]]:
        return [item for item in self.results if item.get("status") != "ok"]


def pair_id_for(left: str, right: str) -> str:
    """Stable pair label, for example ``cam_0__cam_1``."""
    return f"{left}__{right}"


def camera_name_from_label(label: str) -> str:
    """Extract the camera directory name from a calibration label path."""
    parent = Path(label).parent.name
    return parent if parent and parent not in {".", ".."} else label


def _camera_centers(calibration: Mapping[str, Any]) -> list[np.ndarray]:
    centers: list[np.ndarray] = []
    for camera in calibration.get("cameras", []):
        center = camera.get("camera_center")
        if center is None and "R" in camera and "t" in camera:
            rotation = np.asarray(camera["R"], dtype=np.float64)
            translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
            center = -(rotation.T @ translation)
        centers.append(np.asarray(center, dtype=np.float64).reshape(3))
    return centers


def _camera_names(calibration: Mapping[str, Any]) -> list[str]:
    return [camera_name_from_label(str(camera.get("label", f"cam_{index}")))
            for index, camera in enumerate(calibration.get("cameras", []))]


def _load_camera_pairs_topology(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _fallback_adjacent_pairs(names: list[str], wrap: bool) -> list[tuple[str, str]]:
    pairs = [(names[index], names[index + 1]) for index in range(max(0, len(names) - 1))]
    if wrap and len(names) > 2:
        pairs.append((names[-1], names[0]))
    return pairs


def select_pin_multi_pairs(
    calibration: Mapping[str, Any],
    options: PINMultiPairSelectionOptions = PINMultiPairSelectionOptions(),
) -> list[tuple[str, str, dict[str, Any]]]:
    """Select ordered camera pairs for pairwise reconstruction.

    ``auto_spatial_neighbors`` prefers the neighbor topology written by the
    calibration pipeline (``camera_pairs.json``); when unavailable it falls
    back to 3D camera-center nearest neighbors with optional ring closure.
    ``manual`` uses the explicit pair list only.
    """
    names = _camera_names(calibration)
    mode = options.mode.lower()
    if mode == "manual":
        selected = [(str(left), str(right), {}) for left, right in options.manual]
        return selected
    if mode != "auto_spatial_neighbors":
        raise ValueError(
            f"camera_pairs.selection must be 'auto_spatial_neighbors' or 'manual', got {mode!r}")

    topology: dict[str, Any] | None = None
    if options.camera_pairs_json is not None:
        path = Path(options.camera_pairs_json)
        if not path.exists():
            raise FileNotFoundError(f"camera_pairs JSON not found: {path}")
        topology = _load_camera_pairs_topology(path)

    ordered = names
    neighbors: Mapping[str, list[str]] | None = None
    if topology is not None:
        ordered = [str(name) for name in topology.get("ordered_camera_names", names)]
        neighbors = topology.get("neighbors")

    candidates = _fallback_adjacent_pairs(ordered, options.wrap)
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for left, right in candidates:
        diagnostic: dict[str, Any] = {}
        if neighbors is not None:
            if right not in neighbors.get(left, []):
                continue
            diagnostic["neighbor_geometry"] = "topology"
            for pair in topology.get("pairs", []):
                cameras = [str(name) for name in pair.get("cameras", [])]
                if cameras == [left, right] or cameras == [right, left]:
                    diagnostic["inlier_match_count"] = int(pair.get("inlier_match_count", 0))
                    break
        selected.append((left, right, diagnostic))
    if not selected and neighbors is None:
        raise ValueError("camera pair selection produced no pairs; check calibration camera order and wrap setting")
    return selected


def _image_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return arr.astype(np.uint8)
    arr = arr.astype(np.float32, copy=False)
    if float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _match_pair_features(
    left: np.ndarray, right: np.ndarray, options: PINMultiPairROIOptions
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2

    if options.feature_method != "sift":
        raise ValueError(f"pair_roi.feature_method only supports 'sift', got {options.feature_method!r}")
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("The current OpenCV build does not provide cv2.SIFT_create().")
    sift = cv2.SIFT_create(nfeatures=max(1, int(options.max_features)))
    keypoints_left, desc_left = sift.detectAndCompute(_image_u8(left), None)
    keypoints_right, desc_right = sift.detectAndCompute(_image_u8(right), None)
    meta: dict[str, Any] = {
        "feature_method": "sift",
        "max_features": int(options.max_features),
        "match_ratio": float(options.match_ratio),
        "mutual_check": bool(options.mutual_check),
        "ransac_reprojection_threshold_px": float(options.ransac_reprojection_threshold_px),
        "min_matches": int(options.min_matches),
        "keypoints_left": int(len(keypoints_left)),
        "keypoints_right": int(len(keypoints_right)),
        "ratio_matches": 0,
        "mutual_matches": 0,
        "ransac_matches": 0,
    }
    empty = np.zeros((0, 2), dtype=np.float64)
    if desc_left is None or desc_right is None or not keypoints_left or not keypoints_right:
        return empty, empty, meta

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ratio = float(options.match_ratio)
    forward = matcher.knnMatch(desc_left, desc_right, k=2)
    ratio_matches = [pair[0] for pair in forward if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance]
    meta["ratio_matches"] = int(len(ratio_matches))

    matches = ratio_matches
    if options.mutual_check and ratio_matches:
        reverse = matcher.knnMatch(desc_right, desc_left, k=2)
        reverse_best: dict[int, int] = {}
        for pair in reverse:
            if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
                reverse_best[int(pair[0].queryIdx)] = int(pair[0].trainIdx)
        matches = [match for match in ratio_matches if reverse_best.get(int(match.trainIdx)) == int(match.queryIdx)]
    meta["mutual_matches"] = int(len(matches))

    left_uv = np.asarray([keypoints_left[match.queryIdx].pt for match in matches], dtype=np.float64).reshape((-1, 2))
    right_uv = np.asarray([keypoints_right[match.trainIdx].pt for match in matches], dtype=np.float64).reshape((-1, 2))
    if len(left_uv) >= 8 and options.ransac_reprojection_threshold_px > 0.0:
        _, inlier_mask = cv2.findFundamentalMat(
            left_uv.astype(np.float32), right_uv.astype(np.float32), cv2.FM_RANSAC,
            float(options.ransac_reprojection_threshold_px), 0.99)
        if inlier_mask is not None:
            keep = inlier_mask.reshape(-1).astype(bool)
            if int(keep.sum()) >= 3:
                left_uv = left_uv[keep]
                right_uv = right_uv[keep]
    meta["ransac_matches"] = int(len(left_uv))
    return left_uv, right_uv, meta


def _alpha_shape_mask(points: np.ndarray, shape: tuple[int, int], radius_scale: float) -> np.ndarray:
    from scipy.spatial import Delaunay

    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < 3:
        return mask
    try:
        triangulation = Delaunay(points)
    except Exception:
        return mask
    edge_lengths: list[float] = []
    for triangle in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_lengths.append(float(np.linalg.norm(points[triangle[first]] - points[triangle[second]])))
    threshold = float(radius_scale) * float(np.mean(edge_lengths)) if edge_lengths else 0.0
    for triangle in triangulation.simplices:
        keep = True
        for first, second in ((0, 1), (1, 2), (2, 0)):
            if np.linalg.norm(points[triangle[first]] - points[triangle[second]]) > threshold:
                keep = False
                break
        if keep:
            corners = points[triangle].astype(np.int64)
            cv2_polygon = corners.reshape((-1, 1, 2))
            mask = _cv2_fill_poly(mask, cv2_polygon)
    return mask


def _cv2_fill_poly(mask: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    import cv2

    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def _support_mask(points: np.ndarray, shape: tuple[int, int], options: PINMultiPairROIOptions) -> np.ndarray:
    import cv2

    height, width = shape
    if options.support == "alpha_shape":
        mask = _alpha_shape_mask(points, (height, width), options.alpha_radius_scale)
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 3:
            hull = cv2.convexHull(points.astype(np.float32))
            mask = _cv2_fill_poly(mask, hull)
    if int(options.erode_pixels) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(options.erode_pixels), int(options.erode_pixels)))
        mask = cv2.erode(mask, kernel)
    return mask


def _save_overlay(path: Path, left: np.ndarray, right: np.ndarray, left_uv: np.ndarray, right_uv: np.ndarray,
                  left_mask: np.ndarray, right_mask: np.ndarray) -> None:
    import cv2

    left_image = cv2.cvtColor(_image_u8(left), cv2.COLOR_GRAY2BGR)
    right_image = cv2.cvtColor(_image_u8(right), cv2.COLOR_GRAY2BGR)
    height = max(left_image.shape[0], right_image.shape[0])
    canvas = np.zeros((height, left_image.shape[1] + right_image.shape[1], 3), dtype=np.uint8)
    canvas[: left_image.shape[0], : left_image.shape[1]] = left_image
    canvas[: right_image.shape[0], left_image.shape[1]:] = right_image
    offset = left_image.shape[1]
    for left_point, right_point in zip(left_uv, right_uv):
        canvas = cv2.line(canvas, tuple(int(v) for v in left_point), (int(right_point[0]) + offset, int(right_point[1])),
                          (0, 0, 255), 1)
    for mask_image, image_half in ((left_mask, 0), (right_mask, offset)):
        contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2, offset=(image_half, 0))
    cv2.imwrite(str(path), canvas)


def generate_pin_multi_pair_roi(
    left_reference: str | Path,
    right_reference: str | Path,
    output_dir: str | Path,
    *,
    options: PINMultiPairROIOptions | None = None,
) -> dict[str, Any]:
    """Generate pair-local ROI masks from reference-time SIFT matches.

    Matches SIFT only between ``left_reference(t0)`` and ``right_reference(t0)``,
    keeps robust geometric inliers, forms the left and right image support
    masks, and saves masks/diagnostics under ``output_dir``.  Failing pairs
    return a structured skipped record instead of raising or falling back to
    NDeF or full-image masks.
    """
    import cv2

    opts = options if options is not None else PINMultiPairROIOptions()
    output = Path(output_dir)
    left = cv2.imread(str(left_reference), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_reference), cv2.IMREAD_GRAYSCALE)
    if left is None:
        raise FileNotFoundError(f"Unable to read left reference image: {left_reference}")
    if right is None:
        raise FileNotFoundError(f"Unable to read right reference image: {right_reference}")
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)

    left_uv, right_uv, meta = _match_pair_features(left, right, opts)
    if len(left_uv) < opts.min_matches:
        return {
            "status": "skipped", "reason": "min_matches", "match_count": int(len(left_uv)),
            "diagnostics": meta, "output_dir": str(output),
        }
    left_mask = _support_mask(left_uv, left.shape, opts)
    right_mask = _support_mask(right_uv, right.shape, opts)
    area_ratio = float(left_mask.mean())
    if area_ratio < opts.min_mask_area_ratio:
        return {
            "status": "skipped", "reason": "mask_area_too_small", "match_count": int(len(left_uv)),
            "mask_area_ratio": area_ratio, "diagnostics": meta, "output_dir": str(output),
        }

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "left_mask.npy", left_mask)
    np.save(output / "right_mask.npy", right_mask)
    cv2.imwrite(str(output / "left_mask.png"), (left_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(output / "right_mask.png"), (right_mask > 0).astype(np.uint8) * 255)
    np.savez(output / "matches.npz", left_uv=left_uv, right_uv=right_uv,
             match_count=np.asarray(len(left_uv), dtype=np.int64))
    _save_overlay(output / "overlay.png", left, right, left_uv, right_uv, left_mask, right_mask)
    meta["support"] = opts.support
    meta["mask_area_ratio"] = float(area_ratio)
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "status": "ok", "match_count": int(len(left_uv)), "mask_area_ratio": float(area_ratio),
        "diagnostics": meta, "output_dir": str(output),
    }


def _options_from_config(values: Mapping[str, Any]) -> tuple[PINMultiPairSelectionOptions, PINMultiPairROIOptions]:
    pair_roi = values.get("pair_roi", {})
    roi_options = PINMultiPairROIOptions(
        feature_method=str(pair_roi.get("feature_method", "sift")),
        max_features=int(pair_roi.get("max_features", 12_000)),
        match_ratio=float(pair_roi.get("match_ratio", 0.75)),
        mutual_check=bool(pair_roi.get("mutual_check", True)),
        ransac_reprojection_threshold_px=float(pair_roi.get("ransac_reprojection_threshold_px", 3.0)),
        min_matches=int(pair_roi.get("min_matches", 20)),
        support=str(pair_roi.get("support", "convex_hull")),  # type: ignore[arg-type]
        alpha_radius_scale=float(pair_roi.get("alpha_radius_scale", 8.0)),
        erode_pixels=int(pair_roi.get("erode_pixels", 0)),
    )
    pairs_config = values.get("camera_pairs", {})
    manual = [(str(item[0]), str(item[1])) for item in pairs_config.get("manual", [])]
    selection = PINMultiPairSelectionOptions(
        mode=str(pairs_config.get("selection", "auto_spatial_neighbors")),  # type: ignore[arg-type]
        wrap=bool(pairs_config.get("wrap", True)),
        manual=tuple(manual),
        camera_pairs_json=pairs_config.get("camera_pairs_json"),
    )
    return selection, roi_options


def pin_multi_pair_roi(config: str | Path | Mapping[str, Any]) -> PINMultiROIResult:
    """Generate pairwise SIFT ROIs for every selected camera pair.

    Writes ``<case root>/result/pin_multi_slover/pair_roi/<pair_id>/`` plus the
    root ``manifest.json``.  Never reads or writes ``result/mask/per_camera``.
    """
    from .config import load_config

    values = load_config(config) if isinstance(config, (str, Path)) else config
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    selection, roi_options = _options_from_config(values)
    calibration_path = Path(case.get("calibration", "result/calibration/calibration_result_scaled.json"))
    calibration_path = calibration_path if calibration_path.is_absolute() else root / calibration_path
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    pair_roi_config = values.get("pair_roi", {})
    output_root = Path(pair_roi_config.get("output", "result/pin_multi_slover/pair_roi"))
    if not output_root.is_absolute():
        output_root = root / output_root

    camera_pairs_json = selection.camera_pairs_json
    if camera_pairs_json is not None and not Path(camera_pairs_json).is_absolute():
        camera_pairs_json = root / camera_pairs_json
    selection = PINMultiPairSelectionOptions(
        mode=selection.mode, wrap=selection.wrap, manual=selection.manual,
        camera_pairs_json=camera_pairs_json)

    selected = select_pin_multi_pairs(calibration, selection)
    image_root = root / str(case.get("images", "images"))
    reference_frame = str(case.get("reference_frame", "001.bmp"))

    results: list[dict[str, Any]] = []
    for left, right, pair_diagnostics in selected:
        pair_id = pair_id_for(left, right)
        left_path = image_root / left / reference_frame
        right_path = image_root / right / reference_frame
        if not left_path.exists() or not right_path.exists():
            results.append({"pair_id": pair_id, "left": left, "right": right, "status": "skipped",
                            "reason": "reference_image_missing", "diagnostics": pair_diagnostics,
                            "output_dir": str(output_root / pair_id)})
            continue
        result = generate_pin_multi_pair_roi(left_path, right_path, output_root / pair_id,
                                             options=roi_options)
        results.append({"pair_id": pair_id, "left": left, "right": right, **result})

    manifest_path = output_root.parent / "manifest.json"
    manifest = {
        "schema_version": 1,
        "route": "pin_multi_slover",
        "stage": "pair_roi",
        "pair_selection": {"mode": selection.mode, "wrap": selection.wrap,
                           "manual": [list(pair) for pair in selection.manual]},
        "pair_roi_output": str(output_root),
        "pairs": results,
        "skipped": [item for item in results if item.get("status") != "ok"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PINMultiROIResult(
        output_root=output_root, pairs=[(str(item["left"]), str(item["right"])) for item in results],
        pair_ids=[str(item["pair_id"]) for item in results], results=results, manifest_path=manifest_path)
