"""Thin YAML/file assembly for the compiled multi-view NDeF-DIC solver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..config import load_config
from ..models import _require_backend
from ..runtime import configure_runtime


def _mapping(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_gray(path: Path) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    # Original deformation_dataset.py converts grayscale observations to [0,1].
    return image.astype(np.float32) / 255.0


def _camera(backend, values: Mapping[str, Any]):
    camera = backend.CameraModel()
    camera.intrinsics = torch.as_tensor(values["K"], dtype=torch.float64)
    camera.rotation = torch.as_tensor(values["R"], dtype=torch.float64)
    camera.translation = torch.as_tensor(values["t"], dtype=torch.float64)
    camera.distortion = torch.as_tensor(values.get("distortion", []), dtype=torch.float64)
    camera.image_width = int(values["image_width"])
    camera.image_height = int(values["image_height"])
    camera.rms_error = float(values.get("rms_error", 0.0))
    camera.label = str(values.get("label", ""))
    return camera


def _load_masks(root: Path, view_names: list[str], shape: tuple[int, int], values: Mapping[str, Any]) -> np.ndarray:
    masks = values.get("case", {}).get("masks")
    if masks is None:
        return np.ones((len(view_names), *shape), dtype=bool)
    folder = _resolve(root, masks)
    loaded = []
    for name in view_names:
        npy = folder / f"{name}_mask.npy"
        if npy.exists():
            mask = np.load(npy).astype(bool)
        else:
            import cv2
            candidates = [folder / f"{name}_mask.png", folder / f"{name}.bmp", folder / f"{name}.png"]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path is not None else None
            if image is None:
                raise ValueError(f"Mask for {name} was not found under {folder}")
            mask = image != 0
        if mask.shape != shape:
            raise ValueError(f"Mask for {name} must match image shape {shape}, got {mask.shape}")
        loaded.append(mask)
    return np.stack(loaded)


def _set_axes_equal_3d(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float((maximum - minimum).max()) * 0.5, 1.0e-12)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _draw_scalar_field_3d(axis, points: np.ndarray, values: np.ndarray, title: str,
                          label: str, cmap: str, *, symmetric: bool = False):
    limits = {}
    if symmetric:
        bound = max(float(np.abs(values).max()), 1.0e-12)
        limits = {"vmin": -bound, "vmax": bound}
    rendered = axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=values,
                            s=1.4, cmap=cmap, linewidths=0.0, depthshade=False, **limits)
    axis.set_title(title)
    axis.set_xlabel("World X")
    axis.set_ylabel("World Y")
    axis.set_zlabel("World Z")
    _set_axes_equal_3d(axis, points)
    return rendered, label


def _plot_scalar_field_3d(points: np.ndarray, values: np.ndarray, path: Path,
                          title: str, label: str, cmap: str, *, symmetric: bool = False) -> None:
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(9, 7), dpi=180, constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    rendered, color_label = _draw_scalar_field_3d(
        axis, points, values, title, label, cmap, symmetric=symmetric)
    colorbar = figure.colorbar(rendered, ax=axis, shrink=0.68, pad=0.08)
    colorbar.set_label(color_label)
    axis.view_init(elev=24, azim=-58)
    figure.savefig(path)
    plt.close(figure)


def _plot_displacement_components_3d(points: np.ndarray, displacement: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    magnitude = np.linalg.norm(displacement, axis=1)
    fields = (
        (magnitude, "Displacement magnitude", "|u|", "viridis", False),
        (displacement[:, 0], "X displacement (Ux)", "Ux", "coolwarm", True),
        (displacement[:, 1], "Y displacement (Uy)", "Uy", "coolwarm", True),
        (displacement[:, 2], "Z displacement (Uz)", "Uz", "coolwarm", True),
    )
    figure = plt.figure(figsize=(15, 12), dpi=180, constrained_layout=True)
    for index, (values, title, label, cmap, symmetric) in enumerate(fields, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        rendered, color_label = _draw_scalar_field_3d(
            axis, points, values, title, label, cmap, symmetric=symmetric)
        colorbar = figure.colorbar(rendered, ax=axis, shrink=0.62, pad=0.08)
        colorbar.set_label(color_label)
        axis.view_init(elev=24, azim=-58)
    figure.savefig(path)
    plt.close(figure)


def _save(result, result_root: Path, visualization_root: Path, surface_payload: Mapping[str, Any],
          camera_names: list[str]) -> None:
    reconstruct = result_root / "ndef" / "reconstruct"
    deformation = result_root / "ndef" / "deformation"
    diagnostics = result_root / "ndef" / "diagnostics"
    vis_surface = visualization_root / "ndef" / "reconstruct"
    vis_deformation = visualization_root / "ndef" / "deformation"
    vis_diagnostics = visualization_root / "ndef" / "diagnostics"
    for directory in (reconstruct, deformation, diagnostics, vis_surface, vis_deformation, vis_diagnostics):
        directory.mkdir(parents=True, exist_ok=True)
    reference, current = result.surface.coordinates.numpy(), result.surface.values.numpy()
    displacement = result.deformation.values.numpy()
    valid = result.valid.numpy().astype(bool)
    magnitude = np.linalg.norm(displacement, axis=1).astype(np.float32)
    source_fields = {key: np.asarray(surface_payload[key]) for key in
                     ("normals", "source_camera", "visibility_mask", "projected_uv", "projected_depth",
                      "depth_abs_error", "visible_counts") if key in surface_payload}
    np.savez_compressed(reconstruct / "reference_surface.npz", points=reference,
             points_sfm=result.reference_surface_sfm.numpy(), sfm_to_world_scale=result.sfm_to_world_scale,
             cam_names=np.asarray(camera_names), **source_fields)
    np.savez_compressed(reconstruct / "current_surface.npz", points=current,
             reference_points=reference, displacement=displacement, displacement_magnitude=magnitude,
             points_sfm=result.current_surface_sfm.numpy(), sfm_to_world_scale=result.sfm_to_world_scale,
             cam_names=np.asarray(camera_names))
    np.savez_compressed(deformation / "reference_to_current.npz", reference_points=reference,
             current_points=current, displacement=displacement, displacement_magnitude=magnitude,
             reference_points_sfm=result.reference_surface_sfm.numpy(),
             current_points_sfm=result.current_surface_sfm.numpy(),
             displacement_sfm=result.deformation_sfm.numpy(),
             displacement_magnitude_sfm=np.linalg.norm(result.deformation_sfm.numpy(), axis=1),
             sfm_to_world_scale=result.sfm_to_world_scale, cam_names=np.asarray(camera_names))
    np.savez_compressed(diagnostics / "projection.npz", reference_uv=result.reference_uv.numpy(),
             current_uv=result.current_uv.numpy(), reference_depth=result.reference_depth.numpy(),
             current_depth=result.current_depth.numpy(), valid=valid)
    history_columns = ["epoch", "step", "loss", "photometric_loss", "smoothness",
                       "valid_pairs", "supervised_pairs", "displacement_rms"]
    history = result.training_history.numpy()
    np.savez_compressed(diagnostics / "training.npz", history=history,
                        history_columns=np.asarray(history_columns),
                        sample_counts=result.training_sample_counts.numpy(),
                        coordinate_center=result.coordinate_center.numpy(),
                        coordinate_scale=result.coordinate_scale.numpy(),
                        batch_size=np.asarray(result.training_batch_size),
                        steps_per_epoch=np.asarray(result.steps_per_epoch),
                        completed_epochs=np.asarray(result.completed_epochs),
                        random_seed=np.asarray(result.random_seed), output_scale=np.asarray(result.output_scale))
    history_json = [dict(zip(history_columns, map(float, row))) for row in history]
    (diagnostics / "training_history.json").write_text(json.dumps(history_json, indent=2), encoding="utf-8")
    checkpoint_meta = {
        "coordinate_center": result.coordinate_center,
        "coordinate_scale": result.coordinate_scale,
        "output_scale": result.output_scale,
        "sfm_to_world_scale": result.sfm_to_world_scale,
        "batch_size": result.training_batch_size,
        "steps_per_epoch": result.steps_per_epoch,
        "completed_epochs": result.completed_epochs,
        "random_seed": result.random_seed,
        "camera_names": camera_names,
    }
    torch.save({"model_state_dict": dict(zip(result.model_parameter_names, result.last_model_state)),
                **checkpoint_meta}, deformation / "deformation_field.pt")
    torch.save({"model_state_dict": dict(zip(result.model_parameter_names, result.model_state)),
                "best_loss": result.diagnostics.final_loss, **checkpoint_meta},
               deformation / "deformation_field_best.pt")
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    displacement_stats = {
        "mean": float(magnitude.mean()), "std": float(magnitude.std()), "max": float(magnitude.max()),
        "percentiles": {str(q): float(np.percentile(magnitude, q)) for q in percentiles},
        "components_mean": displacement.mean(axis=0).tolist(),
        "components_std": displacement.std(axis=0).tolist(),
    }
    (diagnostics / "summary.json").write_text(json.dumps({
        "coordinate_frame": "calibration world frame",
        "deformation": "X_current - X_reference",
        "sfm_to_world_scale": result.sfm_to_world_scale,
        "visibility": "fixed surface-dataset reference visibility; current positive depth and image bounds",
        "metrics": dict(result.diagnostics.metrics),
        "iterations": result.diagnostics.iterations,
        "final_loss": result.diagnostics.final_loss,
        "training": {"batch_size": result.training_batch_size, "steps_per_epoch": result.steps_per_epoch,
                     "completed_epochs": result.completed_epochs, "random_seed": result.random_seed,
                     "output_scale": result.output_scale, "history_columns": history_columns},
        "sampling": {"method": "torch.randint_with_replacement", "population": int(len(reference)),
                     "draws": int(result.training_sample_counts.sum().item()),
                     "unique_points": int((result.training_sample_counts.numpy() > 0).sum())},
        "normalization": {"coordinate_center": result.coordinate_center.numpy().tolist(),
                          "coordinate_scale": result.coordinate_scale.numpy().tolist()},
        "displacement_statistics": displacement_stats,
        "surface_input_fields": sorted(source_fields),
        "output_schema": {"final_checkpoint": "deformation/deformation_field.pt",
                          "best_checkpoint": "deformation/deformation_field_best.pt",
                          "history": "diagnostics/training_history.json",
                          "training_arrays": "diagnostics/training.npz"},
    }, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    for name, points in (("reference", reference), ("current", current)):
        _plot_scalar_field_3d(points, points[:, 2], vis_surface / f"{name}_surface.png",
                              f"NDeF {name} surface", "World Z", "turbo")
    _plot_scalar_field_3d(reference, magnitude, vis_deformation / "magnitude.png",
                          "NDeF displacement magnitude", "|u|", "viridis")
    component_names = ("x", "y", "z")
    for component, component_name in enumerate(component_names):
        _plot_scalar_field_3d(reference, displacement[:, component],
                              vis_deformation / f"displacement_{component_name}.png",
                              f"NDeF {component_name.upper()} displacement",
                              f"U{component_name}", "coolwarm", symmetric=True)
    _plot_displacement_components_3d(reference, displacement,
                                     vis_deformation / "displacement_components_3d.png")
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.bar(np.arange(valid.shape[1]), valid.sum(axis=0)); axis.set_title("Valid surface observations by view")
    axis.set_xlabel("view index"); axis.set_ylabel("samples")
    figure.savefig(vis_diagnostics / "valid_observations.png", dpi=160); plt.close(figure)
    if len(history):
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.plot(history[:, 2], label="total"); axis.plot(history[:, 3], label="photometric", alpha=0.8)
        if np.any(history[:, 4] > 0): axis.plot(history[:, 4], label="smoothness", alpha=0.8)
        axis.set_yscale("log"); axis.set_xlabel("training step"); axis.set_ylabel("loss"); axis.legend()
        figure.savefig(vis_diagnostics / "training_loss.png", dpi=160); plt.close(figure)


def ndef_sparse_precalculation(config: str | Path | Mapping[str, Any] = "config/ndef_multiview.yaml", *, write_case_artifacts: bool = True):
    """Compile-time C++ sparse patch-DIC matching and two-time multi-view DLT.

    The supplied reference surface is used solely for cross-camera search
    centres, matching NDeF-DIC's `surface_dataset` contract; its points are not
    used in either triangulation.
    """
    backend = _require_backend(); values = _mapping(config); configure_runtime(values)
    case = values.get("case", {}); root = Path(case.get("root", "."))
    image_root = _resolve(root, case["images"])
    calibration_payload = json.loads(_resolve(root, case["calibration"]).read_text(encoding="utf-8"))
    camera_values = calibration_payload.get("cameras", calibration_payload.get("scaled_cameras"))
    if camera_values is None: raise ValueError("calibration must contain cameras or scaled_cameras")
    names = [str(item.get("label", f"cam_{index}")) for index, item in enumerate(camera_values)]
    frame = int(case.get("frame", -1)); reference, current = [], []
    for name in names:
        paths = sorted(path for path in (image_root / name).iterdir() if path.is_file())
        if len(paths) < 2: raise ValueError(f"{name} requires reference and deformed images")
        reference.append(_read_gray(paths[0])); current.append(_read_gray(paths[frame]))
    if len({image.shape for image in reference + current}) != 1: raise ValueError("all sparse-precalculation images must share one shape")
    masks = _load_masks(root, names, reference[0].shape, values)
    payload = np.load(_resolve(root, case["reference_surface"]), allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        surface = payload["points"]
        visibility, uv = payload.get("visibility_mask"), payload.get("projected_uv")
    else: surface, visibility, uv = payload, None, None
    max_surface_points = int(values.get("surface", {}).get("max_points", len(surface)))
    if max_surface_points < 1: raise ValueError("surface.max_points must be positive")
    if len(surface) > max_surface_points:
        indices = np.linspace(0, len(surface) - 1, max_surface_points, dtype=np.int64)
        surface = surface[indices]
        if visibility is not None: visibility, uv = visibility[indices], uv[indices]
    if visibility is None or uv is None:
        intrinsics = torch.as_tensor(np.stack([item["K"] for item in camera_values]), dtype=torch.float64)
        rotations = torch.as_tensor(np.stack([item["R"] for item in camera_values]), dtype=torch.float64)
        translations = torch.as_tensor(np.stack([item["t"] for item in camera_values]), dtype=torch.float64)
        distortions = torch.as_tensor(np.stack([item.get("distortion", [0.0] * 5) for item in camera_values]), dtype=torch.float64)
        uv_t, depth = backend.project_points_multi_view(torch.as_tensor(surface, dtype=torch.float64), intrinsics, rotations, translations, distortions)
        uv = uv_t.numpy(); visibility = (depth.numpy() > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < reference[0].shape[1]) & (uv[..., 1] >= 0) & (uv[..., 1] < reference[0].shape[0])
    options = backend.NDeFSparsePrecalculationOptions()
    for key, value in values.get("precalculation", {}).get("sparse", {}).items():
        if hasattr(options, key): setattr(options, key, value)
    options.random_seed = int(values.get("precalculation", {}).get("sparse", {}).get(
        "random_seed", values.get("runtime", {}).get("random_seed", 23)))
    requested_device = str(values.get("precalculation", {}).get("sparse", {}).get("device", "auto"))
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else
                          "cpu" if requested_device == "auto" else requested_device)
    result = backend.NDeFSparsePrecalculator(options).solve(torch.from_numpy(np.stack(reference)).to(device),
        torch.from_numpy(np.stack(current)).to(device), torch.from_numpy(masks).to(device),
        torch.from_numpy(np.asarray(visibility, dtype=bool)), torch.from_numpy(np.asarray(uv, dtype=np.float64)),
        [_camera(backend, item) for item in camera_values])
    if write_case_artifacts:
        output = Path(values.get("output", {}).get("result", "result")); output = output if output.is_absolute() else root / output
        folder = output / "ndef" / "precalculation"; folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "sparse_tracks.npz", source_camera=result.source_camera.numpy(), source_uv=result.source_uv.numpy(),
                 reference_points=result.reference_points.numpy(), current_points=result.current_points.numpy(),
                 displacement=result.displacement.numpy(), displacement_magnitude=result.displacement_magnitude.numpy(),
                 camera_count=result.camera_count.numpy(), reference_reprojection_error=result.reference_reprojection_error.numpy(),
                 current_reprojection_error=result.current_reprojection_error.numpy(), mean_match_score=result.mean_match_score.numpy(),
                 inlier_mask=result.inlier_mask.numpy())
        source_camera = result.source_camera.numpy()
        (folder / "sparse_scale.json").write_text(json.dumps({
            "scale_stats": {"median": result.scale.median, "mean": result.scale.mean,
                            "p75": result.scale.p75, "p90": result.scale.p90, "max": result.scale.maximum},
            "n_tracks": int(len(source_camera)), "n_inliers": int(result.inlier_mask.sum().item()),
            "per_camera": [{"camera": name, "requested_seeds": options.points_per_camera,
                            "triangulated_tracks": int(np.count_nonzero(source_camera == index))}
                           for index, name in enumerate(names)],
            "sampling": {"method": "seeded_random_within_spatial_grid_then_random_fill",
                         "without_replacement": True, "random_seed": options.random_seed,
                         "min_texture_std": options.min_texture_std},
            "coordinate_unit": "input camera/surface unit",
        }, indent=2), encoding="utf-8")
    return result


def ndef_dic(config: str | Path | Mapping[str, Any] = "config/ndef_multiview.yaml", *, write_case_artifacts: bool = True):
    """Run C++/LibTorch NDeF optimization for one synchronized multi-view frame."""
    backend = _require_backend()
    values = _mapping(config)
    configure_runtime(values)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    image_root = _resolve(root, case["images"])
    calibration = json.loads(_resolve(root, case["calibration"]).read_text(encoding="utf-8"))
    camera_values = calibration.get("cameras", calibration.get("scaled_cameras"))
    if camera_values is None: raise ValueError("calibration must contain cameras or scaled_cameras")
    names = [str(camera.get("label", f"cam_{index}")) for index, camera in enumerate(camera_values)]
    frame = int(case.get("frame", -1))
    reference, deformed = [], []
    for name in names:
        paths = sorted(path for path in (image_root / name).iterdir() if path.is_file())
        if len(paths) < 2:
            raise ValueError(f"{name} requires reference and deformed images")
        reference.append(_read_gray(paths[0])); deformed.append(_read_gray(paths[frame]))
    if len({image.shape for image in reference + deformed}) != 1:
        raise ValueError("NDeF currently requires all camera images to share one shape")
    masks = _load_masks(root, names, reference[0].shape, values)
    surface_data = np.load(_resolve(root, case["reference_surface"]), allow_pickle=True)
    if isinstance(surface_data, np.lib.npyio.NpzFile):
        if "points" not in surface_data:
            raise ValueError("NDeF surface dataset NPZ requires points[N,3]")
        surface_payload = {key: surface_data[key] for key in surface_data.files}
        surface = surface_payload["points"].astype(np.float32)
        surface_visibility = surface_data.get("visibility_mask")
        surface_uv = surface_data.get("projected_uv")
        surface_counts = surface_data.get("visible_counts")
    else:
        surface = surface_data.astype(np.float32)
        surface_payload = {"points": surface}
        surface_visibility = surface_uv = surface_counts = None
    max_points = int(values.get("surface", {}).get("max_points", len(surface)))
    if max_points < 1: raise ValueError("surface.max_points must be positive")
    if len(surface) > max_points:
        population = len(surface)
        indices = np.linspace(0, len(surface) - 1, max_points, dtype=np.int64)
        surface = surface[indices]
        for key, array in list(surface_payload.items()):
            if np.ndim(array) > 0 and len(array) == population:
                surface_payload[key] = array[indices]
        if surface_visibility is not None:
            surface_visibility, surface_uv, surface_counts = surface_visibility[indices], surface_uv[indices], surface_counts[indices]
    problem = backend.NDeFProblem(torch.from_numpy(surface), torch.from_numpy(np.stack(reference)),
                                  torch.from_numpy(np.stack(deformed)), torch.from_numpy(masks),
                                  torch.from_numpy(masks.copy()), [_camera(backend, item) for item in camera_values])
    if surface_visibility is not None:
        problem.set_surface_observations(torch.from_numpy(surface_visibility), torch.from_numpy(surface_uv),
                                         torch.from_numpy(surface_counts.astype(np.float32)))
    deformation_model = values.get("deformation_model", {}); options = backend.NDeFModelOptions()
    options.hidden_dim = int(deformation_model.get("hidden_dim", options.hidden_dim))
    options.hidden_layers = int(deformation_model.get("hidden_layers", options.hidden_layers))
    options.output_scale = float(deformation_model.get("output_scale", options.output_scale))
    precalculation = values.get("precalculation", {})
    if precalculation.get("displacement"):
        payload = np.load(_resolve(root, precalculation["displacement"]), allow_pickle=True)
        displacement = payload[precalculation.get("key", "displacement")] if isinstance(payload, np.lib.npyio.NpzFile) else payload
        scale = backend.estimate_ndef_displacement_scale(torch.as_tensor(displacement, dtype=torch.float32),
                                                          float(precalculation.get("mad_threshold", 5.0)))
        statistic = str(precalculation.get("statistic", "mean"))
        values_by_name = {"median": scale.median, "mean": scale.mean, "p75": scale.p75, "p90": scale.p90, "max": scale.maximum}
        if statistic not in values_by_name:
            raise ValueError("precalculation.statistic must be median, mean, p75, p90, or max")
        options.output_scale = float(values_by_name[statistic])
    for key, value in deformation_model.get("fourier_encoding", {}).items():
        if key in {"enabled", "num_frequencies", "include_input", "angular_scale"}: setattr(options.fourier_encoding, key, value)
    problem.model_options = options
    scale_config = values.get("scale", {})
    if scale_config.get("calibration_scale"):
        scale_payload = json.loads(_resolve(root, scale_config["calibration_scale"]).read_text(encoding="utf-8"))
        if "sfm_to_world_scale" not in scale_payload:
            raise ValueError("scale.calibration_scale must contain sfm_to_world_scale")
        problem.sfm_to_world_scale = float(scale_payload["sfm_to_world_scale"])
    else:
        problem.sfm_to_world_scale = float(scale_config.get("sfm_to_world_scale", 1.0))
    training = values.get("deformation_training", {})
    for key in ("training_epochs", "batch_size", "auto_batch_start", "auto_batch_max", "memory_fraction",
                "max_steps_per_epoch", "prediction_batch_size", "random_seed", "photometric_iterations",
                "photometric_sample_count", "photometric_learning_rate", "weight_decay", "smoothness_weight",
                "patch_radius", "min_valid_patch_ratio", "invalid_patch_penalty"):
        if key in training: setattr(problem, key, training[key])
    if "seed" in training: problem.random_seed = int(training["seed"])
    problem.bspline_degree = int(values.get("interpolation", {}).get("degree", problem.bspline_degree))
    name = str(training.get("photometric_loss", "znssd")).lower()
    problem.photometric_loss = backend.PhotometricLossType.SSD if name in {"ssd", "mse"} else backend.PhotometricLossType.ZNSSD
    if name not in {"ssd", "mse", "znssd"}: raise ValueError("training.photometric_loss must be 'mse', 'ssd', or 'znssd'")
    problem.set_device(str(training.get("device", "cpu")))
    result = backend.NDeFSolver().solve(problem)
    if write_case_artifacts:
        output = Path(values.get("output", {}).get("result", "result")); visual = Path(values.get("output", {}).get("visualization", "visualization"))
        _save(result, output if output.is_absolute() else root / output,
              visual if visual.is_absolute() else root / visual, surface_payload, names)
    return result
