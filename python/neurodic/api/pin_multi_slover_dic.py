"""Pairwise multi-camera PIN-DIC workflow assembly.

Runs one independent pipeline per selected camera pair: reference-time SIFT
pair ROIs, three planar PIN fields (A0->B0, A0->Ak, A0->Bk) solved in C++,
and per-pair 3D reconstruction of X0/Xk/dX.  Pair products are saved under
``result/pin_multi_slover/pairs/<pair_id>/``; the fused stage stays disabled.
"""

from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..models import _require_backend
from ..pin_multi_quality import compute_pair_reason_codes, pair_quality_summary
from ..case_io import multiview_image_pairs, named_multiview_image_pairs
from ..pin_multi_roi import camera_name_from_label, pin_multi_pair_roi
from ..runtime import configure_runtime
from .pin_dic import _build_problem, _mapping
from .pin_stereo_dic import _camera, _field_image, _read_gray, _save_pair_visualization


def _cameras_by_name(calibration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cameras: dict[str, dict[str, Any]] = {}
    for camera in calibration.get("cameras", []):
        cameras[camera_name_from_label(str(camera.get("label", "")))] = dict(camera)
    return cameras


def _save_scatter(path: Path, points: np.ndarray, valid: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    rendered = axis.scatter(points[valid, 0], points[valid, 1], c=points[valid, 2], s=1, cmap="turbo")
    axis.set_aspect("equal")
    axis.set_title(title)
    figure.colorbar(rendered, ax=axis, label="Z")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_reconstruction_roi_fields(directory: Path, xy: np.ndarray, reference: np.ndarray,
                                    current: np.ndarray, valid: np.ndarray, *,
                                    roi_mask: np.ndarray | None, image_size: tuple[int, int] | None,
                                    pair_id: str) -> None:
    """Render reconstructed world coordinates on the left-camera ROI grid."""
    import matplotlib.pyplot as plt

    width, height = image_size if image_size is not None else (
        int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1)
    def field(values: np.ndarray, component: int) -> np.ndarray:
        image = np.full((height, width), np.nan, dtype=np.float64)
        inside = valid & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        image[xy[inside, 1], xy[inside, 0]] = values[inside, component]
        if roi_mask is not None:
            image[~roi_mask] = np.nan
        return image
    for name, points, state in (("reference", reference, "reference"), ("current", current, "current")):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for axis, component, label in zip(axes, range(3), ("X", "Y", "Z")):
            rendered = axis.imshow(field(points, component), cmap="turbo")
            axis.set_title(f"{pair_id} {state} reconstruction: {label} (world)")
            axis.set_axis_off(); figure.colorbar(rendered, ax=axis, label=f"{label} (world)")
        figure.savefig(directory / f"{name}_roi_xyz.png", dpi=170); plt.close(figure)
        figure = plt.figure(figsize=(8, 7), constrained_layout=True)
        axis = figure.add_subplot(projection="3d")
        plotted = axis.scatter(points[valid, 0], points[valid, 1], points[valid, 2],
                               c=points[valid, 2], s=0.5, cmap="turbo")
        axis.set(xlabel="X (world)", ylabel="Y (world)", zlabel="Z (world)",
                 title=f"{pair_id} {state} reconstructed surface")
        axis.set_box_aspect(np.maximum(np.ptp(points[valid], axis=0), 1e-8))
        figure.colorbar(plotted, ax=axis, shrink=.72, label="Z (world)")
        figure.savefig(directory / f"{name}_surface_3d.png", dpi=170); plt.close(figure)


def _save_pair_result(pair_result, result_root: Path, visualization_root: Path, *,
                      roi_mask: np.ndarray | None = None, image_size: tuple[int, int] | None = None,
                      max_reprojection_error_px: float = 5.0) -> dict[str, Any]:
    pair_id = str(pair_result.pair_id)
    result = pair_result.result
    disp_dir, reconstruct_dir, deformation_dir = (result_root / "pairs" / pair_id / name
                                                  for name in ("disp", "reconstruct", "deformation"))
    quality_dir = result_root / "pairs" / pair_id / "quality"
    vis_disp_dir, vis_reconstruct_dir, vis_deformation_dir = (
        visualization_root / "pairs" / pair_id / name
        for name in ("disp", "reconstruct", "deformation"))
    for directory in (disp_dir, reconstruct_dir, deformation_dir, quality_dir, vis_disp_dir,
                      vis_reconstruct_dir, vis_deformation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, field, title in (
        ("reference_disparity", result.reference_disparity, "L0 to R0 reference disparity"),
        ("left_temporal", result.left_temporal, "L0 to Llast temporal displacement"),
        ("deformed_disparity", result.deformed_disparity, "L0 to Rlast deformed disparity"),
    ):
        np.savez(disp_dir / f"{name}.npz", coordinates=field.displacement.coordinates.numpy(),
                 displacement=field.displacement.values.numpy(), iterations=field.diagnostics.iterations,
                 final_loss=field.diagnostics.final_loss)
        _save_pair_visualization(vis_disp_dir / f"{name}.png", field, title)
    valid = result.valid.numpy().astype(bool)
    reference = result.reference_points.numpy()
    current = result.current_points.numpy()
    np.savez(reconstruct_dir / "reference.npz", left_coordinates=result.left_reference_coordinates.numpy(),
             right_coordinates=result.right_reference_coordinates.numpy(), points=reference, valid=valid,
             reprojection_error=result.reference_reprojection_error.numpy())
    np.savez(reconstruct_dir / "current.npz", left_coordinates=result.left_current_coordinates.numpy(),
             right_coordinates=result.right_current_coordinates.numpy(), points=current, valid=valid,
             reprojection_error=result.current_reprojection_error.numpy())
    _save_scatter(vis_reconstruct_dir / "reference.png", reference, valid, f"{pair_id} reference shape (Z)")
    _save_scatter(vis_reconstruct_dir / "current.png", current, valid, f"{pair_id} current shape (Z)")
    _save_reconstruction_roi_fields(vis_reconstruct_dir, result.left_reference_coordinates.numpy().astype(np.int64),
                                    reference, current, valid, roi_mask=roi_mask, image_size=image_size,
                                    pair_id=pair_id)
    displacement = result.displacement_3d.numpy()
    import torch
    closure = _require_backend().compute_local_displacement_consistency(
        torch.as_tensor(reference, dtype=torch.float64), torch.as_tensor(displacement, dtype=torch.float64),
        torch.as_tensor(valid, dtype=torch.bool), 16, 5.0)
    closure_residual = closure.residual.numpy()
    closure_inlier = closure.inlier_mask.numpy().astype(bool)
    filtered_valid = valid & closure_inlier
    np.savez(deformation_dir / "initial_to_current.npz", coordinates=result.left_reference_coordinates.numpy(),
             reference_points=reference, current_points=current, displacement=displacement, valid=valid)
    np.savez(deformation_dir / "closure_quality.npz", coordinates=result.left_reference_coordinates.numpy(),
             residual=closure_residual, predicted_displacement=closure.predicted_displacement.numpy(),
             valid=valid, inlier=closure_inlier, filtered_valid=filtered_valid,
             residual_median=closure.residual_median, residual_mad=closure.residual_mad,
             residual_threshold=closure.residual_threshold, k_neighbors=np.asarray(16), mad_factor=np.asarray(5.0))
    np.savez(deformation_dir / "initial_to_current_filtered.npz", coordinates=result.left_reference_coordinates.numpy(),
             reference_points=reference, current_points=current, displacement=displacement,
             valid=filtered_valid, closure_residual=closure_residual)
    import matplotlib.pyplot as plt

    xy = result.left_reference_coordinates.numpy().astype(np.int64)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, index, label in zip(axes, range(3), ("U", "V", "W")):
        rendered = axis.imshow(_field_image(xy, displacement, valid, index), cmap="turbo")
        axis.set_title(f"3D displacement {label}")
        figure.colorbar(rendered, ax=axis)
    figure.savefig(vis_deformation_dir / "initial_to_current.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    residual_image = _field_image(xy, closure_residual[:, None], valid, 0)
    rendered = axes[0].imshow(residual_image, cmap="magma")
    axes[0].set_title("Local 3D closure residual"); figure.colorbar(rendered, ax=axes[0], label="world displacement residual")
    mask_image = _field_image(xy, filtered_valid[:, None].astype(np.float64), valid, 0)
    axes[1].imshow(mask_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Closure inlier mask (white = retained)")
    figure.savefig(vis_deformation_dir / "closure_quality.png", dpi=170)
    plt.close(figure)
    count = int(valid.sum())
    stats: dict[str, Any] = {
        "pair_id": pair_id,
        "total_points": int(valid.size),
        "valid_points": count,
        "valid_ratio": float(count / valid.size) if valid.size else 0.0,
        "closure_inliers": int(filtered_valid.sum()),
        "closure_rejected": int((valid & ~closure_inlier).sum()),
        "closure_residual_median": float(closure.residual_median),
        "closure_residual_mad": float(closure.residual_mad),
        "closure_residual_threshold": float(closure.residual_threshold),
    }
    if count:
        stats["reference_mean_reprojection_error_px"] = float(
            result.reference_reprojection_error.numpy()[valid].mean())
        stats["current_mean_reprojection_error_px"] = float(
            result.current_reprojection_error.numpy()[valid].mean())
        stats["displacement_rms"] = float(
            math.sqrt(float(np.square(displacement[valid]).sum(axis=1).mean())))
    (deformation_dir / "initial_to_current_summary.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    codes, _ = compute_pair_reason_codes(pair_result, roi_mask,
                                         max_reprojection_error_px=max_reprojection_error_px,
                                         image_size=image_size)
    np.save(quality_dir / "reason_codes.npy", codes)
    quality = pair_quality_summary(pair_result, roi_mask,
                                   max_reprojection_error_px=max_reprojection_error_px,
                                   image_size=image_size)
    (quality_dir / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    stats["quality"] = quality
    return stats


def _update_manifest(manifest_path: Path, solve: dict[str, Any]) -> None:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["solve"] = solve
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _pin_2d_config(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load the planar PIN configuration used for every pairwise field.

    Multi-camera configuration owns only pair selection, pair-local ROI,
    calibration, reconstruction, and output paths.  Model, seed, and training
    options belong to the standard 2-D PIN route so both workflows use exactly
    the same planar solver settings.
    """
    source = values.get("pin_2d_config", "config/pin_2d.yaml")
    if not isinstance(source, (str, Path, Mapping)):
        raise ValueError("pin_2d_config must be a YAML path or mapping")
    return _mapping(source)


def pin_multi_slover_dic(
    config: str | Path | Mapping[str, Any] = "config/pin_multi.yaml",
    *,
    write_case_artifacts: bool = True,
    max_pairs: int | None = None,
):
    """Run pairwise multi-camera PIN-DIC on every selected camera pair.

    Stage 1 generates reference-time SIFT pair ROIs; stage 2 assembles three
    PINProblem instances per pair; stage 3 solves them in C++ and reconstructs
    X0, Xk and dX per pair.  Products are written under
    ``result/pin_multi_slover/pairs/<pair_id>/``.
    """
    backend = _require_backend()
    values = _mapping(config)
    configure_runtime(values)
    pin_values = _pin_2d_config(values)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    image_root = root / str(case.get("images", "images"))
    frame = int(case.get("frame", -1))

    roi_result = pin_multi_pair_roi(values)
    ready = [(str(item["left"]), str(item["right"]), str(item["pair_id"]))
             for item in roi_result.results if item.get("status") == "ok"]
    if max_pairs is not None:
        ready = ready[:max(0, int(max_pairs))]
    if not ready:
        raise ValueError("pin_multi_slover: no camera pair produced a valid ROI; check pair_roi diagnostics")

    calibration_path = Path(case.get("calibration", "result/calibration/calibration_result_scaled.json"))
    calibration_path = calibration_path if calibration_path.is_absolute() else root / calibration_path
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    cameras = _cameras_by_name(calibration)
    names = sorted(cameras)
    references, deformed_frames = named_multiview_image_pairs(image_root, names)
    try:
        current_paths = deformed_frames[frame]
    except IndexError as error:
        raise ValueError(f"case.frame {frame} is outside the {len(deformed_frames)} multi-view deformed frames") from error
    reference_paths = dict(zip(names, references))
    current_paths_by_name = dict(zip(names, current_paths))

    reconstruction = values.get("reconstruction", {})
    problem = backend.PINMultiProblem()
    problem.world_scale = float(reconstruction.get("world_scale", 1.0))
    problem.require_image_bounds = bool(reconstruction.get("require_image_bounds", True))
    problem.set_reconstruction_options(
        float(reconstruction.get("max_reprojection_error_px", 5.0)),
        bool(reconstruction.get("require_positive_depth", True)),
        int(reconstruction.get("undistort_iterations", 12)))

    pair_frames: list[dict[str, Any]] = []
    pair_context: dict[str, dict[str, Any]] = {}
    for left, right, pair_id in ready:
        if left not in cameras or right not in cameras:
            raise ValueError(f"Calibration missing camera label for pair {pair_id}")
        l0_path, lk_path = reference_paths[left], current_paths_by_name[left]
        r0_path, rk_path = reference_paths[right], current_paths_by_name[right]
        l0, r0, lk, rk = (_read_gray(path) for path in (l0_path, r0_path, lk_path, rk_path))
        mask_path = roi_result.output_root / pair_id / "left_mask.npy"
        mask = np.load(mask_path) != 0
        shapes = {image.shape for image in (l0, r0, lk, rk)}
        if len(shapes) != 1 or l0.shape != mask.shape:
            raise ValueError(f"Pair {pair_id}: all four images and the left ROI mask must share one shape")
        problems = [_build_problem(l0, target, mask, pin_values)
                    for target in (r0, lk, rk)]
        for planar_problem in problems:
            planar_problem.compute_neural_strain_2d = False
        problem.add_pair(pair_id, problems[0], problems[1], problems[2],
                         _camera(backend, cameras[left]), _camera(backend, cameras[right]))
        pair_frames.append({"pair_id": pair_id, "left": left, "right": right,
                            "reference": str(l0_path.name), "current": str(lk_path.name),
                            "left_mask": str(mask_path)})
        pair_context[pair_id] = {
            "roi_mask": mask,
            "image_size": (int(l0.shape[1]), int(l0.shape[0])),
            "max_reprojection_error_px": float(reconstruction.get("max_reprojection_error_px", 5.0)),
        }

    result = backend.PINMultiSolver().solve(problem)
    if write_case_artifacts:
        output = Path(values.get("output", {}).get("result", "result"))
        visualization = Path(values.get("output", {}).get("visualization", "visualization"))
        result_root = output if output.is_absolute() else root / output
        visualization_root = visualization if visualization.is_absolute() else root / visualization
        pair_stats = [
            _save_pair_result(pair, result_root, visualization_root,
                              roi_mask=pair_context[pair.pair_id]["roi_mask"],
                              image_size=pair_context[pair.pair_id]["image_size"],
                              max_reprojection_error_px=pair_context[pair.pair_id]["max_reprojection_error_px"])
            for pair in result.pairs]
        solve = {
            "stage": "pairwise_solve",
            "world_scale": problem.world_scale,
            "reference_selection": "first image in every view directory",
            "current_frame": frame,
            "pairs": pair_stats,
        }
        _update_manifest(roi_result.manifest_path, solve)
        fusion_config = values.get("fusion", {})
        if bool(fusion_config.get("enabled", False)):
            from ..pin_multi_fusion import fuse_pin_multi_surfaces

            fusion_summary = fuse_pin_multi_surfaces(values, result_root=result_root)
            manifest = json.loads(roi_result.manifest_path.read_text(encoding="utf-8"))
            manifest["fusion"] = fusion_summary
            roi_result.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def run_pin_multi_case(config: str | Path | Mapping[str, Any] = "config/pin_multi.yaml") -> list[Any]:
    """Solve, fuse, and compute traditional 3D strain for every multiview time step."""
    values = _mapping(config)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    _, _, frames = multiview_image_pairs(root / str(case.get("images", "images")))
    base_output = values.get("output", {})
    base_pair_roi = values.get("pair_roi", {}).get("output")
    results = []
    for index, frame_paths in enumerate(frames):
        current = copy.deepcopy(dict(values))
        current.setdefault("case", {})["frame"] = index
        namespace = frame_paths[0].stem
        output = current.setdefault("output", {})
        for key in ("result", "visualization"):
            if key in base_output:
                output[key] = str(Path(base_output[key]) / namespace)
        if base_pair_roi is not None:
            current.setdefault("pair_roi", {})["output"] = str(Path(base_pair_roi) / namespace)
        results.append(pin_multi_slover_dic(current))
    return results
