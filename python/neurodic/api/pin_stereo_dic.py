"""Thin file/YAML assembly for the compiled stereo PIN workflow."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..models import _require_backend
from ..runtime import configure_runtime
from ..case_io import stereo_image_pairs
from .pin_dic import _build_problem, _mapping
from ..pin_multi_quality import compute_pair_reason_codes, REASON_CODES


def _read_gray(path: Path) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image.astype(np.float32)


def _camera(backend, values: Mapping[str, Any]):
    import torch
    camera = backend.CameraModel()
    camera.intrinsics = torch.as_tensor(values["K"], dtype=torch.float64)
    camera.rotation = torch.as_tensor(values["R"], dtype=torch.float64)
    camera.translation = torch.as_tensor(values["t"], dtype=torch.float64)
    camera.distortion = torch.as_tensor(values.get("distortion", []), dtype=torch.float64)
    camera.image_width = int(values.get("image_width", 0))
    camera.image_height = int(values.get("image_height", 0))
    camera.rms_error = float(values.get("rms_error", 0.0))
    camera.label = str(values.get("label", ""))
    return camera


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _field_image(xy: np.ndarray, values: np.ndarray, valid: np.ndarray, index: int) -> np.ndarray:
    height = int(xy[:, 1].max()) + 1 if len(xy) else 1
    width = int(xy[:, 0].max()) + 1 if len(xy) else 1
    image = np.full((height, width), np.nan, dtype=np.float64)
    image[xy[valid, 1], xy[valid, 0]] = values[valid, index]
    return image


def _save_pair_visualization(path: Path, result, title: str) -> None:
    import matplotlib.pyplot as plt
    xy = result.displacement.coordinates.numpy().astype(np.int64)
    uv = result.displacement.values.numpy()
    valid = np.isfinite(uv).all(axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, index, label in zip(axes, range(2), ("u", "v")):
        rendered = axis.imshow(_field_image(xy, uv, valid, index), cmap="turbo")
        axis.set_title(f"{title}: {label}")
        figure.colorbar(rendered, ax=axis)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save(result, result_root: Path, visualization_root: Path, *, roi_mask: np.ndarray | None = None,
          image_size: tuple[int, int] | None = None, max_reprojection_error_px: float = 5.0) -> None:
    disp_dir, reconstruct_dir, deformation_dir = (result_root / name for name in ("disp", "reconstruct", "deformation"))
    vis_disp_dir, vis_reconstruct_dir, vis_deformation_dir = (
        visualization_root / name for name in ("disp", "reconstruct", "deformation"))
    for directory in (disp_dir, reconstruct_dir, deformation_dir, vis_disp_dir, vis_reconstruct_dir, vis_deformation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, field, title in (
        ("reference_disparity", result.reference_disparity, "L0 to R0 reference disparity"),
        ("left_temporal", result.left_temporal, "L0 to Llast temporal displacement"),
        ("deformed_disparity", result.deformed_disparity, "L0 to Rlast deformed disparity"),
    ):
        np.savez(disp_dir / f"{name}.npz", coordinates=field.displacement.coordinates.numpy(),
                 displacement=field.displacement.values.numpy(), iterations=field.diagnostics.iterations,
                 final_loss=field.diagnostics.final_loss,
                 training_history=field.training_history.numpy(),
                 training_history_columns=np.asarray(["phase", "phase_step", "loss"]),
                 training_history_schema_version=np.asarray("neurodic.pin.training/v1"))
        if field.evaluation_requested_count:
            residuals = field.evaluation_residuals.numpy()
            finite = residuals[np.isfinite(residuals)]
            evaluation = {"schema_version": "neurodic.fixed_evaluation/v1", "solver": "pin_stereo",
                          "scope": {"field": name},
                          "evaluation_set": {"identity": f"pin-v1:{field.evaluation_seed}:{field.evaluation_eligible_count}:{field.evaluation_patch_radius}:{field.evaluation_loss_type}",
                                             "seed": field.evaluation_seed, "sampling": "stable_hash_ranked_roi_indices",
                                             "eligible_count": field.evaluation_eligible_count, "requested_count": field.evaluation_requested_count},
                          "loss": {"type": field.evaluation_loss_type, "patch_radius": field.evaluation_patch_radius,
                                   "aggregation": "mean_per_valid_window", "unit": "photometric_objective"},
                          "valid_count": field.evaluation_valid_count,
                          "valid_ratio": field.evaluation_valid_count / field.evaluation_requested_count,
                          "summary": {"mean": float(finite.mean()) if len(finite) else None,
                                      "median": float(np.median(finite)) if len(finite) else None,
                                      "p95": float(np.percentile(finite, 95)) if len(finite) else None}}
            np.savez_compressed(disp_dir / f"{name}_evaluation.npz", schema_version=np.asarray(evaluation["schema_version"]),
                                indices=field.evaluation_indices.numpy(), residual=residuals)
            (disp_dir / f"{name}_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        _save_pair_visualization(vis_disp_dir / f"{name}.png", field, title)
    valid = result.valid.numpy().astype(bool)
    reference = result.reference_points.numpy()
    current = result.current_points.numpy()
    np.savez(reconstruct_dir / "initial.npz", left_coordinates=result.left_reference_coordinates.numpy(),
             right_coordinates=result.right_reference_coordinates.numpy(), points=reference, valid=valid,
             reprojection_error=result.reference_reprojection_error.numpy())
    np.savez(reconstruct_dir / "last.npz", left_coordinates=result.left_current_coordinates.numpy(),
             right_coordinates=result.right_current_coordinates.numpy(), points=current, valid=valid,
             reprojection_error=result.current_reprojection_error.numpy())
    displacement = result.displacement_3d.numpy()
    strain = result.strain_3d.values.numpy()
    np.savez(deformation_dir / "initial_to_last.npz", coordinates=result.left_reference_coordinates.numpy(),
             reference_points=reference, current_points=current, displacement=displacement, strain=strain,
             strain_components=np.asarray(["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_xz"]), valid=valid)
    reason_codes, reprojection = compute_pair_reason_codes(
        result, roi_mask, image_size=image_size, max_reprojection_error_px=max_reprojection_error_px)
    reference_error = result.reference_reprojection_error.numpy()
    current_error = result.current_reprojection_error.numpy()
    geometry_dir = result_root / "diagnostics"; geometry_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(geometry_dir / "stereo_geometry.npz", schema_version=np.asarray("neurodic.stereo_geometry/v1"),
                        reason_code=reason_codes, reason_names=np.asarray([REASON_CODES[i] for i in sorted(REASON_CODES)]),
                        valid=valid, reference_reprojection_error=reference_error,
                        current_reprojection_error=current_error,
                        reference_positive_depth=reference[:, 2] > 0.0, current_positive_depth=current[:, 2] > 0.0)
    counts = {name: int((reason_codes == code).sum()) for code, name in REASON_CODES.items()}
    (geometry_dir / "stereo_geometry.json").write_text(json.dumps({
        "schema_version": "neurodic.stereo_geometry/v1", "reason_priority": [REASON_CODES[i] for i in sorted(REASON_CODES)],
        "scope": {"states": ["reference", "current"], "fields": ["reference_disparity", "left_temporal", "deformed_disparity"]},
        "counts": counts, "max_reprojection_error_px": max_reprojection_error_px}, indent=2), encoding="utf-8")
    summary = {"total_points": int(result.valid.numel()),
               "valid_points": int(result.valid.sum().item()),
               "coordinate_frame": "calibration world frame",
               "displacement": "X_current - X_reference",
               "strain": "Green-Lagrange; weighted local least-squares gradient on triangulated 3D points"}
    (deformation_dir / "initial_to_last_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    xy = result.left_reference_coordinates.numpy().astype(np.int64)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, index, label in zip(axes, range(3), ("U", "V", "W")):
        rendered = axis.imshow(_field_image(xy, displacement, valid, index), cmap="turbo")
        axis.set_title(f"3D displacement {label}")
        figure.colorbar(rendered, ax=axis)
    figure.savefig(vis_deformation_dir / "initial_to_last.png", dpi=160)
    plt.close(figure)
    for name, points in (("initial", reference), ("last", current)):
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        rendered = axis.scatter(points[valid, 0], points[valid, 1], c=points[valid, 2], s=1, cmap="turbo")
        axis.set_aspect("equal")
        axis.set_title(f"Reconstructed {name} shape (colored by Z)")
        figure.colorbar(rendered, ax=axis, label="Z")
        figure.savefig(vis_reconstruct_dir / f"{name}.png", dpi=160)
        plt.close(figure)


def pin_stereo_dic(config: str | Path | Mapping[str, Any] = "config/pin_stereo.yaml",
                   *, write_case_artifacts: bool = True):
    """Solve L0->R0, L0->L1 and L0->R1 in C++, then triangulate both states."""
    backend = _require_backend()
    values = _mapping(config)
    configure_runtime(values)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    (left_reference, right_reference), deformed_pairs = stereo_image_pairs(
        _resolve(root, case["left_images"]), _resolve(root, case["right_images"]))
    frame = int(case.get("frame", -1))
    try:
        left_current, right_current = deformed_pairs[frame]
    except IndexError as error:
        raise ValueError(f"case.frame {frame} is outside the {len(deformed_pairs)} stereo deformed pairs") from error
    l0, r0 = _read_gray(left_reference), _read_gray(right_reference)
    l1, r1 = _read_gray(left_current), _read_gray(right_current)
    import cv2
    roi = cv2.imread(str(_resolve(root, case["roi"])), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise ValueError("Unable to read stereo ROI")
    mask = roi != 0
    shapes = {image.shape for image in (l0, r0, l1, r1)}
    if len(shapes) != 1 or l0.shape != mask.shape:
        raise ValueError("All stereo images and the L0 ROI must have matching shapes")
    problems = [_build_problem(l0, target, mask, values) for target in (r0, l1, r1)]
    for planar_problem in problems:
        planar_problem.compute_neural_strain_2d = False
    camera_data = json.loads(_resolve(root, case["camera_pair"]).read_text(encoding="utf-8"))
    problem = backend.PINStereoProblem(*problems, _camera(backend, camera_data["left"]),
                                       _camera(backend, camera_data["right"]))
    reconstruction = values.get("reconstruction", {})
    problem.world_scale = float(reconstruction.get("world_scale", camera_data.get("world_scale", 1.0)))
    problem.require_image_bounds = bool(reconstruction.get("require_image_bounds", True))
    strain = values.get("traditional_strain", {})
    problem.compute_traditional_strain = bool(strain.get("enabled", True))
    problem.traditional_strain_neighbors = int(strain.get("neighbors", 12))
    problem.set_reconstruction_options(float(reconstruction.get("max_reprojection_error_px", 5.0)),
                                       bool(reconstruction.get("require_positive_depth", True)),
                                       int(reconstruction.get("undistort_iterations", 12)))
    result = backend.PINStereoSolver().solve(problem)
    if write_case_artifacts:
        output = Path(values.get("output", {}).get("result", "result"))
        visualization = Path(values.get("output", {}).get("visualization", "visualization"))
        _save(result, output if output.is_absolute() else root / output,
              visualization if visualization.is_absolute() else root / visualization, roi_mask=mask,
              image_size=(l0.shape[1], l0.shape[0]),
              max_reprojection_error_px=float(reconstruction.get("max_reprojection_error_px", 5.0)))
    return result


def run_stereo_case(config: str | Path | Mapping[str, Any] = "config/pin_stereo.yaml") -> list[Any]:
    """Solve every matched stereo deformation pair and save one 3D strain field per time step."""
    values = _mapping(config)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    _, pairs = stereo_image_pairs(_resolve(root, case["left_images"]), _resolve(root, case["right_images"]))
    base_output = values.get("output", {})
    results = []
    for index, (left, _) in enumerate(pairs):
        current = copy.deepcopy(dict(values))
        current.setdefault("case", {})["frame"] = index
        output = current.setdefault("output", {})
        for key in ("result", "visualization"):
            if key in base_output:
                output[key] = str(Path(base_output[key]) / left.stem)
        results.append(pin_stereo_dic(current))
    return results
