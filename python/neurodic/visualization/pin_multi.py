"""Fused pairwise-surface visualization for the pin_multi_slover route.

Writes 3D surface, displacement, strain, and (when a ground_truth directory is
present) reconstruction-error views under ``visualization/pin_multi_slover/fused/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_SCATTER_CAP = 30_000


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurodic-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sample(points: np.ndarray, maximum: int = _SCATTER_CAP, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= maximum:
        return np.arange(points.shape[0])
    return np.random.RandomState(seed).choice(points.shape[0], maximum, replace=False)


def _scatter3d(points: np.ndarray, values: np.ndarray, title: str, path: Path,
               *, label: str, symmetric: bool = False) -> None:
    plt = _plt()
    selected = _sample(points)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    values = np.asarray(values)
    finite = np.isfinite(values[selected])
    if symmetric:
        limit = float(np.nanmax(np.abs(values[selected]))) or 1.0
        rendered = ax.scatter(*points[selected, :3].T, c=values[selected], s=1, cmap="turbo",
                              vmin=-limit, vmax=limit, alpha=0.9)
    else:
        rendered = ax.scatter(*points[selected, :3].T, c=values[selected], s=1, cmap="turbo", alpha=0.9)
    fig.colorbar(rendered, ax=ax, label=label, shrink=0.6)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize_fused(fused_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Surface and displacement views of the fused point cloud."""
    fused_root = Path(fused_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deformation = np.load(fused_root / "deformation.npz")
    reference = deformation["reference_points"]
    displacement = deformation["displacement"]
    valid = deformation["valid"].astype(bool)
    source = deformation["source_pair"].astype(np.int64)

    reference_valid = reference[valid]
    displacement_valid = displacement[valid]
    magnitude = np.linalg.norm(displacement_valid, axis=1)

    outputs: dict[str, str] = {}
    _scatter3d(reference_valid, reference_valid[:, 2], "Fused reference surface (colored by Z)",
               output_dir / "reference_surface.png", label="Z")
    _scatter3d(reference_valid, magnitude, "Displacement magnitude (mm)",
               output_dir / "displacement_magnitude.png", label="|dX| (mm)")
    for index, name in enumerate(("U", "V", "W")):
        _scatter3d(reference_valid, displacement_valid[:, index],
                   f"Displacement component {name} (mm)",
                   output_dir / f"displacement_{name.lower()}.png", label=name, symmetric=True)
    counts = [int((source[valid] == index).sum()) for index in range(int(source.max()) + 1)]
    summary = {"valid_points": int(valid.sum()), "points_by_source": counts,
               "displacement_magnitude_mean": float(magnitude.mean())}
    (output_dir / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def visualize_fused_ground_truth_error(fused_root: str | Path, ground_truth_root: str | Path,
                                       output_dir: str | Path) -> dict[str, Any]:
    """Compare fused displacement with ground-truth displacement at nearest points."""
    from scipy.spatial import cKDTree

    fused_root = Path(fused_root)
    ground_truth_root = Path(ground_truth_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deformation = np.load(fused_root / "deformation.npz")
    reference = deformation["reference_points"]
    displacement = deformation["displacement"]
    valid = deformation["valid"].astype(bool)
    reference_truth = np.load(ground_truth_root / "points_ref.npy").astype(np.float64)
    displacement_truth = np.load(ground_truth_root / "displacement_step1.npy").astype(np.float64)

    tree = cKDTree(reference_truth)
    distances, indices = tree.query(reference[valid])
    truth = displacement_truth[indices]
    error = displacement[valid] - truth
    error_magnitude = np.linalg.norm(error, axis=1)
    surface_distance = distances

    np.savez(output_dir / "displacement_error.npz", coordinates=reference[valid],
             error=error, error_magnitude=error_magnitude,
             surface_distance=surface_distance, valid=np.ones(len(error), bool))

    report: dict[str, Any] = {
        "points": int(valid.sum()),
        "surface_distance_mm": {"mean": float(surface_distance.mean()),
                                "p95": float(np.percentile(surface_distance, 95))},
        "displacement_error_mm": {
            "mean": float(error_magnitude.mean()),
            "rms": float(np.sqrt(np.square(error_magnitude).mean())),
            "p95": float(np.percentile(error_magnitude, 95)),
            "max": float(error_magnitude.max()),
        },
    }
    _scatter3d(reference[valid], error_magnitude, "3D displacement error vs ground truth (mm)",
               output_dir / "displacement_error.png", label="|error| (mm)")
    for index, name in enumerate(("U", "V", "W")):
        _scatter3d(reference[valid], error[:, index], f"Displacement error {name} (mm)",
                   output_dir / f"displacement_error_{name.lower()}.png", label=name, symmetric=True)
    (output_dir / "ground_truth_error.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
