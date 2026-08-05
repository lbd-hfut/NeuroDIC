"""Calibration result visualizations mirrored under ``case/visualization``."""

from __future__ import annotations

import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def visualization_dir_for_result(case_root: str | Path, result_dir: str | Path) -> Path:
    case_root = Path(case_root).resolve()
    result_dir = Path(result_dir).resolve()
    try:
        relative = result_dir.relative_to(case_root / "result")
    except ValueError:
        relative = result_dir.name
    return case_root / "visualization" / relative


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurodic-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def visualize_stereo_calibration(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write reprojection, baseline, and detected-corner diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    outputs: dict[str, str] = {}
    errors = np.asarray(result.get("per_pair_errors", []), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(errors) + 1), errors, "o-", label="final")
    initial = np.asarray(result.get("initial_per_pair_errors", []), dtype=float)
    if len(initial):
        ax.plot(np.arange(1, len(initial) + 1), initial, "x--", label="initial")
    for index in result.get("rejected_pair_indices", []):
        ax.axvline(int(index) + 1, color="tab:red", alpha=0.25)
    ax.set(xlabel="Calibration pair", ylabel="Reprojection error (px)", title="Stereo calibration reprojection errors")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "reprojection_errors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["reprojection_errors"] = str(path)

    centers = np.asarray([result["left"]["camera_center"], result["right"]["camera_center"]], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(centers[:, 0], centers[:, 2], c=("tab:blue", "tab:orange"), s=70)
    ax.plot(centers[:, 0], centers[:, 2], "k--", alpha=0.65)
    for name, center in zip(("left", "right"), centers):
        ax.annotate(name, (center[0], center[2]), xytext=(5, 5), textcoords="offset points")
    ax.set(aspect="equal", xlabel="World X", ylabel="World Z", title="Stereo camera centers and baseline")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "stereo_geometry.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["stereo_geometry"] = str(path)

    for side in ("left", "right"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for index, detection in enumerate(result.get(f"{side}_detections", [])):
            points = np.asarray(detection.get("image_points", []), dtype=float).reshape((-1, 2))
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=4, label=f"{index + 1:02d}")
        ax.invert_yaxis()
        ax.set(aspect="equal", xlabel="u (px)", ylabel="v (px)", title=f"{side.title()} chessboard corner detections")
        fig.tight_layout()
        path = output_dir / f"{side}_detections.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[f"{side}_detections"] = str(path)
    (output_dir / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


def visualize_multiview_calibration(
    calibration: Mapping[str, Any], image_paths: Sequence[str | Path], output_dir: str | Path
) -> dict[str, str]:
    """Export Traditional-DIC sparse-scene and per-camera observation views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = list(calibration.get("cameras", []))
    points_data = list(calibration.get("points3d", []))
    names = [str(camera.get("label", f"cam_{index}")) for index, camera in enumerate(cameras)]
    centers = np.asarray([camera["camera_center"] for camera in cameras], dtype=float).reshape((-1, 3))
    points = np.asarray([point["xyz"] for point in points_data], dtype=float).reshape((-1, 3))
    cam_indices, point_indices, uv = [], [], []
    for point_index, point in enumerate(points_data):
        for observation in point.get("observations", []):
            xy = np.asarray(observation.get("uv", []), dtype=float).reshape(-1)
            if xy.size >= 2 and np.all(np.isfinite(xy[:2])):
                cam_indices.append(int(observation["camera_index"]))
                point_indices.append(point_index)
                uv.append(xy[:2])
    observed_cameras = np.asarray(cam_indices, dtype=np.int32)
    observed_points = np.asarray(point_indices, dtype=np.int64)
    observed_uv = np.asarray(uv, dtype=float).reshape((-1, 2))
    np.savez_compressed(
        output_dir / "colmap_like_sparse_model.npz", cam_names=np.asarray(names), camera_centers_world=centers,
        points3D=points, observation_cam_indices=observed_cameras, observation_point_indices=observed_points,
        observation_uv=observed_uv,
    )
    (output_dir / "summary.json").write_text(json.dumps({
        "coordinate_system": calibration.get("coordinate_system", "sfm"), "num_cameras": len(names),
        "num_sparse_points": int(len(points)), "num_observations": int(len(observed_cameras)),
    }, indent=2), encoding="utf-8")
    plt = _plt()
    outputs: dict[str, str] = {}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        selected = np.random.RandomState(0).choice(len(points), min(len(points), 30000), replace=False)
        ax.scatter(*points[selected].T, s=1, c="0.25", alpha=0.55)
    ax.scatter(*centers.T, c="tab:red", s=45, marker="^")
    for name, center in zip(names, centers):
        ax.text(*center, name, fontsize=7)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Sparse points and camera poses")
    fig.tight_layout()
    path = output_dir / "sparse_scene.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["sparse_scene"] = str(path)

    columns = min(4, max(1, len(names)))
    rows = ceil(len(names) / columns)
    fig = plt.figure(figsize=(4 * columns, 3.5 * rows))
    for index, name in enumerate(names):
        ax = fig.add_subplot(rows, columns, index + 1, projection="3d")
        selected = observed_points[observed_cameras == index]
        selected = selected[(selected >= 0) & (selected < len(points))]
        if len(selected):
            ax.scatter(*points[selected].T, s=2, c="tab:red", alpha=0.75)
        ax.set_title(f"{name}: {len(selected)} obs", fontsize=9)
    fig.tight_layout()
    path = output_dir / "camera_observations_3d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["camera_observations_3d"] = str(path)

    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.5 * rows), squeeze=False)
    for index, name in enumerate(names):
        ax = axes.flat[index]
        image = plt.imread(image_paths[index])
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        image_uv = observed_uv[observed_cameras == index]
        if len(image_uv):
            selected = np.random.RandomState(index).choice(len(image_uv), min(len(image_uv), 1200), replace=False)
            ax.scatter(image_uv[selected, 0], image_uv[selected, 1], s=8, c="red", linewidths=0, alpha=0.7)
        ax.set(title=f"{name}: {len(image_uv)} obs")
        ax.axis("off")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    path = output_dir / "camera_observations_2d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["camera_observations_2d"] = str(path)
    (output_dir / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs
