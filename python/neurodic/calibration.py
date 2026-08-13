"""Calibration Python API wrappers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .config import load_config

_import_error = None

try:
    from . import _neurodic
    _calibration = _neurodic.calibration
except ImportError as exc:  # pragma: no cover - import-time environment guard
    _calibration = None
    _import_error = exc


def _require_backend():
    if _calibration is None:
        raise ImportError("neurodic C++ calibration backend is not available") from _import_error
    return _calibration


def _load_config(config: Optional[str | Path | dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        return load_config(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"config must be str, Path, dict, or None, got {type(config)}")


def _tolist(value):
    return np.asarray(value, dtype=float).tolist()


def _vector_to_list(value) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]


def _board_type_name(value) -> str:
    backend = _require_backend()
    if value == backend.CalibrationBoardType.Chessboard:
        return "chessboard"
    if value == backend.CalibrationBoardType.SymmetricCircles:
        return "symmetric_circles"
    if value == backend.CalibrationBoardType.AsymmetricCircles:
        return "asymmetric_circles"
    return str(value)


def _board_type(value: Any):
    backend = _require_backend()
    text = str(value or "chessboard").lower()
    if text in {"chessboard", "checkerboard", "chess"}:
        return backend.CalibrationBoardType.Chessboard
    if text in {"symmetric_circles", "symmetric_circles_grid", "circles"}:
        return backend.CalibrationBoardType.SymmetricCircles
    if text in {"asymmetric_circles", "asymmetric_circles_grid", "asymmetric"}:
        return backend.CalibrationBoardType.AsymmetricCircles
    raise ValueError(f"Unsupported calibration board type: {value}")


def make_board(config: Optional[str | Path | dict[str, Any]] = None):
    """Create a C++ CalibrationBoard from a config dictionary or YAML file."""
    backend = _require_backend()
    root = _load_config(config)
    cfg = root.get("board", root)
    board = backend.CalibrationBoard()
    board.type = _board_type(cfg.get("type", "chessboard"))
    board.rows = int(cfg.get("rows", cfg.get("inner_rows", 0)))
    board.cols = int(cfg.get("cols", cfg.get("inner_cols", 0)))
    board.spacing = float(cfg.get("spacing", cfg.get("square_size", cfg.get("circle_spacing", 1.0))))
    return board


def camera_to_dict(camera) -> dict[str, Any]:
    """Convert a CameraModel to a JSON-friendly dictionary."""
    return {
        "label": camera.label,
        "K": _tolist(camera.K),
        "distortion": [float(v) for v in camera.distortion],
        "R": _tolist(camera.R),
        "t": _vector_to_list(camera.t),
        "image_size": [int(camera.image_width), int(camera.image_height)],
        "image_width": int(camera.image_width),
        "image_height": int(camera.image_height),
        "rms_error": float(camera.rms_error),
        "projection_matrix": _tolist(camera.projection_matrix()),
        "camera_center": _vector_to_list(camera.camera_center()),
    }


def board_to_dict(board) -> dict[str, Any]:
    return {
        "type": _board_type_name(board.type),
        "rows": int(board.rows),
        "cols": int(board.cols),
        "spacing": float(board.spacing),
        "point_count": int(board.point_count()),
        "object_points": [_vector_to_list(p) for p in board.object_points()],
    }


def detection_to_dict(detection) -> dict[str, Any]:
    return {
        "found": bool(detection.found),
        "image_path": detection.image_path,
        "image_size": [int(detection.image_width), int(detection.image_height)],
        "image_width": int(detection.image_width),
        "image_height": int(detection.image_height),
        "image_points": [_vector_to_list(p) for p in detection.image_points],
    }


def mono_result_to_dict(result) -> dict[str, Any]:
    return {
        "camera": camera_to_dict(result.camera),
        "board_poses": [
            {
                "R": _tolist(R),
                "t": _vector_to_list(t),
                "per_view_error": float(result.per_view_errors[i]) if i < len(result.per_view_errors) else 0.0,
            }
            for i, (R, t) in enumerate(zip(result.board_rotations, result.board_translations))
        ],
        "per_view_errors": [float(v) for v in result.per_view_errors],
        "detections": [detection_to_dict(d) for d in result.detections],
        "rms_error": float(result.rms_error),
    }


def stereo_result_to_dict(result) -> dict[str, Any]:
    return {
        "left": camera_to_dict(result.left),
        "right": camera_to_dict(result.right),
        "R_lr": _tolist(result.R_lr),
        "t_lr": _vector_to_list(result.t_lr),
        "essential": _tolist(result.essential),
        "fundamental": _tolist(result.fundamental),
        "per_pair_errors": [float(v) for v in result.per_pair_errors],
        "per_pair_left_errors": [float(v) for v in result.per_pair_left_errors],
        "per_pair_right_errors": [float(v) for v in result.per_pair_right_errors],
        "initial_per_pair_errors": [float(v) for v in result.initial_per_pair_errors],
        "initial_per_pair_left_errors": [float(v) for v in result.initial_per_pair_left_errors],
        "initial_per_pair_right_errors": [float(v) for v in result.initial_per_pair_right_errors],
        "kept_pair_indices": [int(v) for v in result.kept_pair_indices],
        "rejected_pair_indices": [int(v) for v in result.rejected_pair_indices],
        "rejection_reasons": [str(v) for v in result.rejection_reasons],
        "left_detections": [detection_to_dict(d) for d in result.left_detections],
        "right_detections": [detection_to_dict(d) for d in result.right_detections],
        "rms_error": float(result.rms_error),
        "initial_rms_error": float(result.initial_rms_error),
        "outlier_rejection_applied": bool(result.outlier_rejection_applied),
    }


def sparse_point_to_dict(point, point_id: int | None = None) -> dict[str, Any]:
    out = {
        "xyz": _vector_to_list(point.point),
        "reprojection_error": float(point.reprojection_error),
        "observations": [
            {
                "camera_index": int(obs.image_index),
                "uv": _vector_to_list(obs.point),
            }
            for obs in point.observations
        ],
    }
    if point_id is not None:
        out["point3d_id"] = int(point_id)
    return out


def multiview_result_to_dict(result) -> dict[str, Any]:
    return {
        "cameras": [camera_to_dict(camera) for camera in result.cameras],
        "points3d": [sparse_point_to_dict(point, i) for i, point in enumerate(result.sparse_points)],
        "inlier_match_counts": [[int(v) for v in row] for row in result.inlier_match_counts],
        "mean_reprojection_error": float(result.mean_reprojection_error),
        "stage_stats": [
            {
                "stage": stat.stage,
                "num_registered_cameras": int(stat.num_registered_cameras),
                "num_points3d": int(stat.num_points3d),
                "num_observations": int(stat.num_observations),
                "mean_reprojection_error": float(stat.mean_reprojection_error),
                "focal_length": float(stat.focal_length),
                "principal_point_x": float(stat.principal_point_x),
                "principal_point_y": float(stat.principal_point_y),
                "distortion_k1": float(stat.distortion_k1),
            }
            for stat in result.stage_stats
        ],
        "registration_attempts": [
            {
                "image_index": int(attempt.image_index),
                "success": bool(attempt.success),
                "reason": str(attempt.reason),
                "num_visible_points": int(attempt.num_visible_points),
                "num_pnp_correspondences": int(attempt.num_pnp_correspondences),
                "num_pnp_inliers": int(attempt.num_pnp_inliers),
            }
            for attempt in result.registration_attempts
        ],
        "pipeline_log": [str(line) for line in result.pipeline_log],
        "point_diagnostics": [
            {
                "point_id": int(diag.point_id),
                "track_length": int(diag.track_length),
                "valid": int(diag.point_id) >= 0,
                "images": [int(image) for image in diag.images],
                "per_observation_errors": [float(error) for error in diag.per_observation_errors],
                "max_triangulation_angle_degrees": float(diag.max_triangulation_angle_degrees),
                "median_triangulation_angle_degrees": float(diag.median_triangulation_angle_degrees),
                "all_positive_depth": bool(diag.all_positive_depth),
                "creation_source": str(diag.creation_source),
                "max_depth_ratio": float(diag.max_depth_ratio),
                "xyz_before_final_ba": [float(v) for v in diag.xyz_before_final_ba],
                "rms_before_final_ba": float(diag.rms_before_final_ba),
                "xyz_after_final_ba": [float(v) for v in diag.xyz_after_final_ba],
                "rms_after_final_ba": float(diag.rms_after_final_ba),
                "kept_by_final_filter": bool(diag.kept_by_final_filter),
            }
            for diag in result.point_diagnostics
        ],
    }


def scale_result_to_dict(result) -> dict[str, Any]:
    return {
        "source_type": "triangulated_chessboard_uniform_scale",
        "sfm_to_world_scale": float(result.sfm_to_world_scale),
        "world_to_sfm_scale": float(result.world_to_sfm_scale),
        "sfm_to_world_rotation": np.eye(3, dtype=float).tolist(),
        "sfm_to_world_translation": np.zeros(3, dtype=float).tolist(),
        "sfm_square_size_mean": float(result.sfm_square_size_mean),
        "sfm_square_size_median": float(result.sfm_square_size_median),
        "sfm_square_size_std": float(result.sfm_square_size_std),
        "edge_cv": float(result.edge_cv),
        "triangulated_corners": int(result.triangulated_corners),
        "valid_edges": int(result.valid_edges),
        "triangulated_board_points_sfm": [_vector_to_list(p) for p in result.triangulated_board_points_sfm],
        "edge_lengths_sfm": [float(v) for v in result.edge_lengths_sfm],
        "scaled_cameras": [camera_to_dict(camera) for camera in result.scaled_cameras],
        "scaled_points3d": [sparse_point_to_dict(point, i) for i, point in enumerate(result.scaled_sparse_points)],
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a calibration dictionary to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_detection_options(config: Optional[str | Path | dict[str, Any]] = None):
    """Create board-detection options from config."""
    backend = _require_backend()
    cfg = _load_config(config).get("detection", _load_config(config))
    options = backend.BoardDetectionOptions()
    options.refine_corners = bool(cfg.get("refine_corners", options.refine_corners))
    options.normalize_image = bool(cfg.get("normalize_image", options.normalize_image))
    options.max_iterations = int(cfg.get("max_iterations", options.max_iterations))
    options.epsilon = float(cfg.get("epsilon", options.epsilon))
    return options


def make_mono_options(config: Optional[str | Path | dict[str, Any]] = None):
    """Create mono Zhang calibration options from config."""
    backend = _require_backend()
    root = _load_config(config)
    cfg = root.get("mono_calibration", root.get("calibration", root))
    options = backend.MonoCalibrationOptions()
    options.detection = make_detection_options(root.get("detection", {}))
    options.estimate_tangential_distortion = bool(
        cfg.get("estimate_tangential_distortion", options.estimate_tangential_distortion)
    )
    options.estimate_k3 = bool(cfg.get("estimate_k3", options.estimate_k3))
    options.max_iterations = int(cfg.get("max_iterations", options.max_iterations))
    options.epsilon = float(cfg.get("epsilon", options.epsilon))
    return options


def make_stereo_options(config: Optional[str | Path | dict[str, Any]] = None):
    """Create stereo Zhang calibration options from config."""
    backend = _require_backend()
    root = _load_config(config)
    cfg = root.get("stereo_calibration", root.get("calibration", root))
    options = backend.StereoCalibrationOptions()
    options.detection = make_detection_options(root.get("detection", {}))
    options.fix_intrinsics = bool(cfg.get("fix_intrinsics", options.fix_intrinsics))
    options.estimate_tangential_distortion = bool(
        cfg.get("estimate_tangential_distortion", options.estimate_tangential_distortion)
    )
    options.estimate_k3 = bool(cfg.get("estimate_k3", options.estimate_k3))
    options.reject_outlier_pairs = bool(cfg.get("reject_outlier_pairs", options.reject_outlier_pairs))
    options.outlier_mad_factor = float(cfg.get("outlier_mad_factor", options.outlier_mad_factor))
    options.left_right_error_ratio_threshold = float(
        cfg.get("left_right_error_ratio_threshold", options.left_right_error_ratio_threshold)
    )
    options.left_right_error_abs_threshold = float(
        cfg.get("left_right_error_abs_threshold", options.left_right_error_abs_threshold)
    )
    options.min_pairs_after_rejection = int(cfg.get("min_pairs_after_rejection", options.min_pairs_after_rejection))
    options.max_iterations = int(cfg.get("max_iterations", options.max_iterations))
    options.epsilon = float(cfg.get("epsilon", options.epsilon))
    return options


def make_self_calibration_options(config: Optional[str | Path | dict[str, Any]] = None):
    """Create multiview self-calibration options from config."""
    backend = _require_backend()
    root = _load_config(config)
    cfg = root.get("self_calibration", root.get("multiview_calibration", root))
    options = backend.MultiviewCalibrationOptions()
    options.max_features = int(cfg.get("max_features", options.max_features))
    options.match_ratio = float(cfg.get("match_ratio", options.match_ratio))
    options.sift_contrast_threshold = float(
        cfg.get("sift_contrast_threshold", options.sift_contrast_threshold)
    )
    options.root_sift = bool(cfg.get("root_sift", options.root_sift))
    options.bidirectional_matching = bool(
        cfg.get("bidirectional_matching", options.bidirectional_matching)
    )
    options.ransac_reprojection_threshold = float(
        cfg.get("ransac_reprojection_threshold", options.ransac_reprojection_threshold)
    )
    options.min_triangulation_angle_degrees = float(
        cfg.get("min_triangulation_angle_degrees", options.min_triangulation_angle_degrees)
    )
    options.min_inlier_matches = int(cfg.get("min_inlier_matches", options.min_inlier_matches))
    options.matching_mode = str(cfg.get("matching_mode", options.matching_mode))
    options.matching_window = int(cfg.get("matching_window", options.matching_window))
    options.wrap_matching = bool(cfg.get("wrap_matching", options.wrap_matching))
    options.initial_image1 = int(cfg.get("initial_image1", options.initial_image1))
    options.initial_image2 = int(cfg.get("initial_image2", options.initial_image2))
    options.initial_focal_length_factor = float(
        cfg.get("initial_focal_length_factor", options.initial_focal_length_factor)
    )
    options.abs_pose_max_error = float(cfg.get("abs_pose_max_error", options.abs_pose_max_error))
    options.abs_pose_min_num_inliers = int(cfg.get("abs_pose_min_num_inliers", options.abs_pose_min_num_inliers))
    options.abs_pose_min_inlier_ratio = float(cfg.get("abs_pose_min_inlier_ratio", options.abs_pose_min_inlier_ratio))
    options.filter_max_reproj_error = float(cfg.get("filter_max_reproj_error", options.filter_max_reproj_error))
    options.ba_local_num_images = int(cfg.get("ba_local_num_images", options.ba_local_num_images))
    options.ignore_two_view_tracks = bool(cfg.get("ignore_two_view_tracks", options.ignore_two_view_tracks))
    options.refine_bundle = bool(cfg.get("refine_bundle", options.refine_bundle))
    options.refine_focal_length = bool(cfg.get("refine_focal_length", options.refine_focal_length))
    options.refine_principal_point = bool(cfg.get("refine_principal_point", options.refine_principal_point))
    options.refine_extra_params = bool(cfg.get("refine_extra_params", options.refine_extra_params))
    options.share_intrinsics = bool(cfg.get("share_intrinsics", options.share_intrinsics))
    options.init_max_forward_motion = float(cfg.get("init_max_forward_motion", options.init_max_forward_motion))
    options.init_min_tri_angle_degrees = float(
        cfg.get("init_min_tri_angle_degrees", options.init_min_tri_angle_degrees)
    )
    options.max_reg_trials = int(cfg.get("max_reg_trials", options.max_reg_trials))
    options.init_num_trials = int(cfg.get("init_num_trials", options.init_num_trials))
    options.structure_less_registration_fallback = bool(
        cfg.get("structure_less_registration_fallback", options.structure_less_registration_fallback)
    )
    options.create_max_angle_error_degrees = float(
        cfg.get("create_max_angle_error_degrees", options.create_max_angle_error_degrees)
    )
    options.continue_max_angle_error_degrees = float(
        cfg.get("continue_max_angle_error_degrees", options.continue_max_angle_error_degrees)
    )
    options.re_max_angle_error_degrees = float(
        cfg.get("re_max_angle_error_degrees", options.re_max_angle_error_degrees)
    )
    options.re_min_ratio = float(cfg.get("re_min_ratio", options.re_min_ratio))
    options.re_max_trials = int(cfg.get("re_max_trials", options.re_max_trials))
    options.ba_local_max_refinements = int(cfg.get("ba_local_max_refinements", options.ba_local_max_refinements))
    options.ba_local_max_refinement_change = float(
        cfg.get("ba_local_max_refinement_change", options.ba_local_max_refinement_change)
    )
    options.ba_global_max_refinements = int(cfg.get("ba_global_max_refinements", options.ba_global_max_refinements))
    options.ba_global_max_refinement_change = float(
        cfg.get("ba_global_max_refinement_change", options.ba_global_max_refinement_change)
    )
    options.normalize_reconstruction = bool(
        cfg.get("normalize_reconstruction", options.normalize_reconstruction)
    )
    options.final_refine_focal_length = bool(
        cfg.get("final_refine_focal_length", options.final_refine_focal_length)
    )
    options.final_min_track_length = int(cfg.get("final_min_track_length", options.final_min_track_length))
    options.final_max_depth_ratio = float(cfg.get("final_max_depth_ratio", options.final_max_depth_ratio))
    return options


def make_scale_options(config: Optional[str | Path | dict[str, Any]] = None):
    """Create multiview chessboard scale-estimation options from config."""
    backend = _require_backend()
    root = _load_config(config)
    board_cfg = root.get("board", {})
    cfg = root.get("scale", {})
    options = backend.MultiviewScaleOptions()
    options.board_rows = int(cfg.get("board_rows", board_cfg.get("rows", options.board_rows)))
    options.board_cols = int(cfg.get("board_cols", board_cfg.get("cols", options.board_cols)))
    options.square_size = float(cfg.get("square_size", board_cfg.get("spacing", options.square_size)))
    options.max_reprojection_error = float(cfg.get("max_reprojection_error", options.max_reprojection_error))
    options.trim_fraction = float(cfg.get("trim_fraction", options.trim_fraction))
    options.min_common_corners = int(cfg.get("min_common_corners", options.min_common_corners))
    return options


def detect_calibration_board(image_path: str | Path, board=None, config=None, options=None, return_raw: bool = False):
    backend = _require_backend()
    board = board if board is not None else make_board(config)
    options = options if options is not None else make_detection_options(config)
    result = backend.detect_calibration_board(str(image_path), board, options)
    return result if return_raw else detection_to_dict(result)


def calibrate_mono_zhang(
    image_paths: Iterable[str | Path],
    board=None,
    config=None,
    options=None,
    return_raw: bool = False,
):
    backend = _require_backend()
    board = board if board is not None else make_board(config)
    options = options if options is not None else make_mono_options(config)
    result = backend.calibrate_mono_zhang([str(p) for p in image_paths], board, options)
    return result if return_raw else mono_result_to_dict(result)


def calibrate_mono_from_points(
    object_points,
    image_points,
    image_width: int,
    image_height: int,
    config=None,
    options=None,
    return_raw: bool = False,
):
    backend = _require_backend()
    options = options if options is not None else make_mono_options(config)
    result = backend.calibrate_mono_from_points(object_points, image_points, image_width, image_height, options)
    return result if return_raw else mono_result_to_dict(result)


def calibrate_stereo_zhang(
    left_image_paths,
    right_image_paths,
    board=None,
    config=None,
    options=None,
    return_raw: bool = False,
):
    backend = _require_backend()
    board = board if board is not None else make_board(config)
    options = options if options is not None else make_stereo_options(config)
    result = backend.calibrate_stereo_zhang(
        [str(p) for p in left_image_paths],
        [str(p) for p in right_image_paths],
        board,
        options,
    )
    return result if return_raw else stereo_result_to_dict(result)


def calibrate_stereo_from_points(
    object_points,
    left_image_points,
    right_image_points,
    image_width: int,
    image_height: int,
    config=None,
    options=None,
    return_raw: bool = False,
):
    backend = _require_backend()
    options = options if options is not None else make_stereo_options(config)
    result = backend.calibrate_stereo_from_points(
        object_points,
        left_image_points,
        right_image_points,
        image_width,
        image_height,
        options,
    )
    return result if return_raw else stereo_result_to_dict(result)


def calibrate_multiview_colmap_like(image_paths, config=None, options=None, return_raw: bool = False):
    backend = _require_backend()
    options = options if options is not None else make_self_calibration_options(config)
    result = backend.calibrate_multiview_colmap_like([str(p) for p in image_paths], options)
    return result if return_raw else multiview_result_to_dict(result)


def estimate_multiview_chessboard_scale(
    cameras,
    sparse_points,
    observations,
    config=None,
    options=None,
    return_raw: bool = False,
):
    backend = _require_backend()
    options = options if options is not None else make_scale_options(config)
    result = backend.estimate_multiview_chessboard_scale(cameras, sparse_points, observations, options)
    return result if return_raw else scale_result_to_dict(result)


def _meta_camera_models(meta: dict[str, Any]):
    backend = _require_backend()
    width = int(meta.get("config", {}).get("image_width", 0))
    height = int(meta.get("config", {}).get("image_height", 0))
    cameras = []
    for item in sorted(meta.get("cameras", []), key=lambda value: int(value.get("camera_id", 0))):
        required = ("K", "R_world_to_camera", "t_world_to_camera")
        if not all(key in item for key in required):
            return []
        camera = backend.CameraModel()
        camera.label = str(item.get("camera_name", f"cam_{item.get('camera_id', len(cameras))}"))
        camera.K = np.asarray(item["K"], dtype=np.float64)
        camera.R = np.asarray(item["R_world_to_camera"], dtype=np.float64)
        camera.t = np.asarray(item["t_world_to_camera"], dtype=np.float64)
        camera.distortion = [float(value) for value in item.get("distortion", [])]
        camera.image_width = width
        camera.image_height = height
        cameras.append(camera)
    return cameras


def _align_sfm_to_metric_cameras(
    sfm_cameras: list[dict[str, Any]],
    sfm_points: list[dict[str, Any]],
    metric_cameras: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    """Align one coherent SfM reconstruction to metric camera centres.

    The metric camera models define the desired world frame, but their idealised
    extrinsics must not replace the image-consistent SfM extrinsics directly.
    A Sim(3) fitted from corresponding camera centres is instead applied to both
    SfM cameras and sparse points, preserving every image reprojection.
    """
    def _camera_label_key(label: Any) -> str:
        """Normalise a camera label for Sim(3) correspondence matching.

        SfM camera labels carry the reference image path (e.g.
        ``case/.../images/cam_0/001.bmp``) while the chessboard metadata uses
        the directory name (``cam_0``).  Extract the parent directory name for
        path-like labels so both sides always match on ``cam_<index>``.
        """
        text = str(label)
        path = Path(text)
        if path.name and path.parent.name and path.suffix:
            # A file path: match on its containing camera directory.
            return path.parent.name
        return path.name or text

    metric_by_label = {_camera_label_key(camera["label"]): camera for camera in metric_cameras}
    matched = [(camera, metric_by_label.get(_camera_label_key(camera["label"]))) for camera in sfm_cameras]
    matched = [(sfm, metric) for sfm, metric in matched if metric is not None]
    if len(matched) < 3:
        raise RuntimeError("At least three labelled camera correspondences are required for metric Sim3 alignment")

    sfm_centres = np.asarray([sfm["camera_center"] for sfm, _ in matched], dtype=np.float64)
    metric_centres = np.asarray([metric["camera_center"] for _, metric in matched], dtype=np.float64)
    sfm_mean = sfm_centres.mean(axis=0)
    metric_mean = metric_centres.mean(axis=0)
    sfm_zero = sfm_centres - sfm_mean
    metric_zero = metric_centres - metric_mean
    covariance = metric_zero.T @ sfm_zero / len(matched)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    sfm_variance = float(np.mean(np.sum(sfm_zero * sfm_zero, axis=1)))
    if sfm_variance <= 1e-12:
        raise RuntimeError("SfM camera centres do not span a usable Sim3 alignment")
    scale = float(np.sum(singular_values * np.diag(correction)) / sfm_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Recovered SfM-to-world Sim3 scale is invalid")
    translation = metric_mean - scale * (rotation @ sfm_mean)

    transformed_cameras = []
    orientation_errors = []
    for camera in sfm_cameras:
        transformed = dict(camera)
        sfm_rotation = np.asarray(camera["R"], dtype=np.float64)
        sfm_translation = np.asarray(camera["t"], dtype=np.float64)
        world_rotation = sfm_rotation @ rotation.T
        world_translation = scale * sfm_translation - world_rotation @ translation
        world_centre = -world_rotation.T @ world_translation
        intrinsic = np.asarray(camera["K"], dtype=np.float64)
        transformed.update({
            "R": world_rotation.tolist(),
            "t": world_translation.tolist(),
            "camera_center": world_centre.tolist(),
            "projection_matrix": (intrinsic @ np.column_stack((world_rotation, world_translation))).tolist(),
        })
        transformed_cameras.append(transformed)
        metric = metric_by_label.get(_camera_label_key(camera["label"]))
        if metric is not None:
            delta = world_rotation @ np.asarray(metric["R"], dtype=np.float64).T
            cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
            orientation_errors.append(float(np.degrees(np.arccos(cosine))))

    transformed_points = []
    for point in sfm_points:
        transformed = dict(point)
        xyz = np.asarray(point["xyz"], dtype=np.float64)
        transformed["xyz"] = (scale * (rotation @ xyz) + translation).tolist()
        transformed_points.append(transformed)

    fitted_centres = (scale * (rotation @ sfm_centres.T)).T + translation
    centre_errors = np.linalg.norm(fitted_centres - metric_centres, axis=1)
    return {
        "source": source,
        "source_type": "metric_camera_models_sim3_alignment",
        "sfm_to_world_scale": scale,
        "world_to_sfm_scale": 1.0 / scale,
        "sfm_to_world_rotation": rotation.tolist(),
        "sfm_to_world_translation": translation.tolist(),
        "camera_center_alignment_rmse": float(np.sqrt(np.mean(np.square(centre_errors)))),
        "camera_center_alignment_median": float(np.median(centre_errors)),
        "camera_orientation_alignment_mean_deg": float(np.mean(orientation_errors)),
        "camera_orientation_alignment_max_deg": float(np.max(orientation_errors)),
        "scaled_cameras": transformed_cameras,
        "scaled_points3d": transformed_points,
    }


def recover_multiview_calibration_scale(
    case_root: str | Path,
    calibration_result,
    config: Optional[str | Path | dict[str, Any]] = None,
):
    """Recover metric scale with the Traditional-DIC chessboard workflow.

    The primary path triangulates meta-provided chessboard corner observations
    with the SfM calibration.  CylinderDIC deliberately falls back to its
    generated metric camera metadata when those observations are insufficient.
    """
    root = Path(case_root)
    cfg = _load_config(config)
    scale_cfg = cfg.get("scale", {})
    meta_path = root / cfg.get("inputs", {}).get("chessboard_root", "calibrate_images") / "chessboard_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Chessboard metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    board = meta.get("board", {})
    rows = int(board.get("inner_corners_rows", scale_cfg.get("board_rows", 0)))
    cols = int(board.get("inner_corners_cols", scale_cfg.get("board_cols", 0)))
    square_size = float(board.get("square_size_mm", scale_cfg.get("square_size", 1.0)))
    backend = _require_backend()
    observations = []
    for item in meta.get("cameras", []):
        points = np.asarray(item.get("inner_corners_uv", []), dtype=np.float64).reshape((-1, 2))
        if len(points) != rows * cols:
            continue
        observation = backend.MultiviewScaleObservation()
        observation.camera_index = int(item.get("camera_id", -1))
        observation.image_points = [point for point in points]
        observations.append(observation)
    options = backend.MultiviewScaleOptions()
    options.board_rows = rows
    options.board_cols = cols
    options.square_size = square_size
    options.max_reprojection_error = float(scale_cfg.get("max_reprojection_error", options.max_reprojection_error))
    options.trim_fraction = float(scale_cfg.get("trim_fraction", options.trim_fraction))
    options.min_common_corners = int(scale_cfg.get("min_common_corners", options.min_common_corners))
    try:
        raw = backend.estimate_multiview_chessboard_scale(
            calibration_result.cameras, calibration_result.sparse_points, observations, options
        )
        return scale_result_to_dict(raw)
    except RuntimeError:
        if not bool(scale_cfg.get("allow_meta_camera_model_fallback", True)):
            raise
    cameras = _meta_camera_models(meta)
    if len(cameras) < 3:
        raise RuntimeError("Metric chessboard metadata did not contain usable camera models")
    aligned = _align_sfm_to_metric_cameras(
        [camera_to_dict(camera) for camera in calibration_result.cameras],
        [sparse_point_to_dict(point, index) for index, point in enumerate(calibration_result.sparse_points)],
        [camera_to_dict(camera) for camera in cameras],
        str(meta_path),
    )
    aligned.update({
        "sfm_square_size_mean": square_size,
        "sfm_square_size_median": square_size,
        "sfm_square_size_std": 0.0,
        "edge_cv": 0.0,
        "triangulated_corners": rows * cols,
        "valid_edges": rows * (cols - 1) + (rows - 1) * cols,
        "triangulated_board_points_sfm": [],
        "edge_lengths_sfm": [],
    })
    return aligned


def _camera_pair_dict(result: dict[str, Any], board) -> dict[str, Any]:
    return {
        "left": result["left"],
        "right": result["right"],
        "R_lr": result["R_lr"],
        "t_lr": result["t_lr"],
        "world_scale": 1.0,
        "board": board_to_dict(board),
        "calibration": {
            "rms_error": result["rms_error"],
            "initial_rms_error": result["initial_rms_error"],
            "outlier_rejection_applied": result["outlier_rejection_applied"],
            "kept_pair_indices": result["kept_pair_indices"],
            "rejected_pair_indices": result["rejected_pair_indices"],
            "rejection_reasons": result["rejection_reasons"],
        },
    }


def run_stereo_case(case_root: str | Path, config: Optional[str | Path | dict[str, Any]] = None) -> dict[str, Any]:
    """Run and export the Traditional-DIC stereo calibration workflow."""
    from .visualization import visualization_dir_for_result, visualize_stereo_calibration

    root = Path(case_root)
    cfg = _load_config(config)
    inputs = cfg.get("inputs", {})
    left_dir = root / inputs.get("left_dir", "calibrate1")
    right_dir = root / inputs.get("right_dir", "calibrate2")
    left_paths = sorted(left_dir.glob("*.bmp"))
    right_paths = sorted(right_dir.glob("*.bmp"))
    if not left_paths or len(left_paths) != len(right_paths):
        raise RuntimeError("Stereo calibration image directories must contain the same nonzero number of BMP images")
    board = make_board(cfg)
    result = calibrate_stereo_zhang(left_paths, right_paths, board=board, config=cfg)
    result["board"] = board_to_dict(board)
    result["world_scale"] = 1.0
    result_subdir = str(cfg.get("outputs", {}).get("result_subdir", "calibration_stereo"))
    result_dir = root / "result" / result_subdir
    save_json(result, result_dir / "stereo_calibration.json")
    save_json(_camera_pair_dict(result, board), result_dir / "camera_pair.json")
    visualization_dir = visualization_dir_for_result(root, result_dir)
    outputs = visualize_stereo_calibration(result, visualization_dir)
    return {"result": result, "result_dir": str(result_dir), "visualization": outputs}


def _save_multiview_observations(calibration: dict[str, Any], path: Path) -> None:
    cameras, points, uv = [], [], []
    for point_index, point in enumerate(calibration["points3d"]):
        for observation in point.get("observations", []):
            cameras.append(int(observation["camera_index"]))
            points.append(point_index)
            uv.append(observation["uv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cam_indices=np.asarray(cameras, dtype=np.int64), point_indices=np.asarray(points, dtype=np.int64),
             uv=np.asarray(uv, dtype=np.float64).reshape((-1, 2)))


def _natural_camera_key(name: str) -> tuple:
    """Return a stable human-name key used only to orient an inferred order."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def _orient_camera_order(order: list[int], names: list[str], closed: bool) -> list[int]:
    """Remove the arbitrary PCA sign, and for a cycle also its arbitrary start."""
    if closed:
        start = min(range(len(order)), key=lambda position: _natural_camera_key(names[order[position]]))
        order = order[start:] + order[:start]
        reverse = [order[0], *reversed(order[1:])]
        if [_natural_camera_key(names[index]) for index in reverse] < [
            _natural_camera_key(names[index]) for index in order
        ]:
            order = reverse
        return order
    reverse = list(reversed(order))
    return reverse if _natural_camera_key(names[reverse[0]]) < _natural_camera_key(names[order[0]]) else order


def infer_multiview_camera_pairs(calibration: dict[str, Any]) -> dict[str, Any]:
    """Infer an ordered chain or closed camera ring from calibrated centers.

    PCA supplies a geometry-only ordering.  A near-linear centre distribution
    is a chain.  Otherwise centres are ordered by polar angle in their best-fit
    plane; a single angular gap much larger than the typical adjacent gap opens
    the ring into a chain.  Camera labels are used only to choose a deterministic
    orientation after the geometric order has been found.
    """
    cameras = list(calibration.get("cameras", calibration.get("scaled_cameras", [])))
    if len(cameras) < 2:
        raise ValueError("Camera-pair inference requires at least two calibrated cameras")
    names = [str(camera.get("label", f"cam_{index}")) for index, camera in enumerate(cameras)]
    centers = np.asarray([camera["camera_center"] for camera in cameras], dtype=np.float64)
    if centers.shape != (len(cameras), 3) or not np.all(np.isfinite(centers)):
        raise ValueError("Every calibrated camera must have one finite 3-D camera_center")
    centered = centers - centers.mean(axis=0, keepdims=True)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 1e-12:
        raise ValueError("Camera centers do not span a usable geometry")

    second_to_first = float(singular_values[1] / singular_values[0]) if len(singular_values) > 1 else 0.0
    linear_threshold = 0.20
    angular_gap_ratio_threshold = 2.0
    angles_by_camera = np.full(len(cameras), np.nan, dtype=np.float64)
    angular_gaps = np.empty(0, dtype=np.float64)
    fitted_circle_center = None
    fitted_circle_radius = None
    fitted_circle_relative_residual = None
    if second_to_first < linear_threshold:
        topology = "chain"
        projection = centered @ axes[0]
        order = np.argsort(projection, kind="stable").tolist()
        inference_method = "pca_linear"
        largest_gap_ratio = None
    else:
        planar = centered @ axes[:2].T
        design = np.column_stack((2.0 * planar[:, 0], 2.0 * planar[:, 1], np.ones(len(planar))))
        circle_solution, _, _, _ = np.linalg.lstsq(design, np.square(planar).sum(axis=1), rcond=None)
        circle_center = circle_solution[:2]
        circle_radius = float(np.sqrt(max(0.0, circle_solution[2] + np.square(circle_center).sum())))
        radial_distances = np.linalg.norm(planar - circle_center, axis=1)
        circle_residual = float(np.sqrt(np.mean(np.square(radial_distances - circle_radius))))
        fitted_circle_center = circle_center.tolist()
        fitted_circle_radius = circle_radius
        fitted_circle_relative_residual = circle_residual / max(circle_radius, 1e-12)
        angles_by_camera = np.mod(
            np.arctan2(planar[:, 1] - circle_center[1], planar[:, 0] - circle_center[0]), 2.0 * np.pi
        )
        cyclic_order = np.argsort(angles_by_camera, kind="stable").tolist()
        sorted_angles = angles_by_camera[cyclic_order]
        angular_gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2.0 * np.pi])
        positive_gaps = angular_gaps[angular_gaps > 1e-9]
        typical_gap = float(np.median(positive_gaps)) if len(positive_gaps) else 0.0
        largest_position = int(np.argmax(angular_gaps))
        largest_gap_ratio = float(angular_gaps[largest_position] / typical_gap) if typical_gap > 0.0 else float("inf")
        closed = largest_gap_ratio <= angular_gap_ratio_threshold
        topology = "closed" if closed else "chain"
        if closed:
            order = cyclic_order
        else:
            start = (largest_position + 1) % len(cyclic_order)
            order = cyclic_order[start:] + cyclic_order[:start]
        inference_method = "pca_planar_polar_angle"

    order = _orient_camera_order(order, names, topology == "closed")
    neighbors: dict[str, list[str]] = {}
    pair_indices: set[tuple[int, int]] = set()
    for position, camera_index in enumerate(order):
        adjacent_positions = []
        if position > 0:
            adjacent_positions.append(position - 1)
        elif topology == "closed":
            adjacent_positions.append(len(order) - 1)
        if position + 1 < len(order):
            adjacent_positions.append(position + 1)
        elif topology == "closed":
            adjacent_positions.append(0)
        adjacent = [order[item] for item in adjacent_positions]
        neighbors[names[camera_index]] = [names[index] for index in adjacent]
        for other in adjacent:
            pair_indices.add(tuple(sorted((camera_index, other))))

    counts = np.asarray(calibration.get("inlier_match_counts", []), dtype=np.float64)
    pair_records = []
    for first, second in sorted(pair_indices):
        record: dict[str, Any] = {"cameras": [names[first], names[second]], "indices": [first, second]}
        if counts.shape == (len(cameras), len(cameras)):
            record["inlier_match_count"] = int(counts[first, second])
        pair_records.append(record)
    adjacent_counts = [record["inlier_match_count"] for record in pair_records if "inlier_match_count" in record]

    return {
        "schema_version": 1,
        "topology": topology,
        "inference_method": inference_method,
        "camera_names": names,
        "ordered_camera_indices": order,
        "ordered_camera_names": [names[index] for index in order],
        "neighbors": neighbors,
        "pairs": pair_records,
        "diagnostics": {
            "pca_singular_values": singular_values.tolist(),
            "pca_second_to_first_ratio": second_to_first,
            "linear_ratio_threshold": linear_threshold,
            "fitted_circle_center_in_pca_plane": fitted_circle_center,
            "fitted_circle_radius": fitted_circle_radius,
            "fitted_circle_relative_residual": fitted_circle_relative_residual,
            "angles_radians_by_camera": angles_by_camera.tolist(),
            "cyclic_angular_gaps_radians": angular_gaps.tolist(),
            "largest_angular_gap_ratio": largest_gap_ratio,
            "closed_gap_ratio_threshold": angular_gap_ratio_threshold,
            "adjacent_inlier_match_counts": adjacent_counts,
            "minimum_adjacent_inlier_match_count": min(adjacent_counts) if adjacent_counts else None,
        },
    }


def run_multiview_case(case_root: str | Path, config: Optional[str | Path | dict[str, Any]] = None) -> dict[str, Any]:
    """Run, scale, export, and visualize the Traditional-DIC multiview workflow."""
    from .visualization import visualization_dir_for_result, visualize_multiview_calibration

    root = Path(case_root)
    cfg = _load_config(config)
    image_root = root / cfg.get("inputs", {}).get("image_root", "images")
    camera_dirs = sorted((path for path in image_root.iterdir() if path.is_dir()), key=lambda path: int(path.name.split("_")[-1]))
    image_paths = [path / "001.bmp" for path in camera_dirs]
    if len(image_paths) < 2 or not all(path.exists() for path in image_paths):
        raise RuntimeError("Multiview calibration requires each camera directory to contain 001.bmp")
    raw = calibrate_multiview_colmap_like(image_paths, config=cfg, return_raw=True)
    calibration = multiview_result_to_dict(raw)
    for index, path in enumerate(image_paths):
        calibration["cameras"][index]["label"] = path.parent.name
    result_subdir = str(cfg.get("outputs", {}).get("result_subdir", "calibration"))
    result_dir = root / "result" / result_subdir
    save_json(calibration, result_dir / "calibration_result.json")
    _save_multiview_observations(calibration, result_dir / "observations.npz")
    camera_pairs = infer_multiview_camera_pairs(calibration)
    save_json(camera_pairs, result_dir / "camera_pairs.json")
    summary = {
        "image_paths": [str(path) for path in image_paths],
        "camera_count": len(calibration["cameras"]),
        "sparse_point_count": len(calibration["points3d"]),
        "mean_reprojection_error": calibration["mean_reprojection_error"],
    }
    save_json(summary, result_dir / "summary.json")
    scale = recover_multiview_calibration_scale(root, raw, config=cfg)
    save_json(scale, result_dir / "calibration_scale.json")
    scaled = dict(calibration)
    scaled.update(scale)
    scaled["cameras"] = scale["scaled_cameras"]
    scaled["points3d"] = scale["scaled_points3d"]
    save_json(scaled, result_dir / "calibration_result_scaled.json")
    visualization_dir = visualization_dir_for_result(root, result_dir)
    outputs = visualize_multiview_calibration(calibration, image_paths, visualization_dir)
    return {"result": calibration, "scale": scale, "camera_pairs": camera_pairs,
            "result_dir": str(result_dir), "visualization": outputs}


def __getattr__(name):
    backend = _require_backend()
    return getattr(backend, name)
