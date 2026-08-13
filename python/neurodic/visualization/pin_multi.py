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
# The 0.1 mm Cylinder mesh contains millions of faces.  One million uniformly
# distributed faces keeps its global coverage visually continuous while leaving
# the full mesh untouched on disk.
_MESH_FACE_CAP = 1_000_000


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


def _mesh3d(vertices: np.ndarray, faces: np.ndarray, face_values: np.ndarray, title: str, path: Path,
            *, label: str, symmetric: bool = False) -> int:
    """Render a deterministic subset of triangles; the saved mesh stays full resolution."""
    plt = _plt()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    count = min(int(faces.shape[0]), _MESH_FACE_CAP)
    ids = np.arange(faces.shape[0]) if count == faces.shape[0] else np.linspace(
        0, faces.shape[0] - 1, count, dtype=np.int64)
    selected_faces = faces[ids]
    triangles = vertices[selected_faces]
    face_values = np.asarray(face_values, dtype=np.float64)[ids]
    finite = np.isfinite(face_values)
    if not finite.any():
        face_values = np.zeros_like(face_values)
    elif not finite.all():
        face_values = np.where(finite, face_values, np.nanmedian(face_values[finite]))
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(triangles, linewidths=0.0, edgecolors="none", alpha=1.0)
    collection.set_array(face_values)
    collection.set_cmap("turbo")
    if symmetric:
        limit = float(np.nanmax(np.abs(face_values))) or 1.0
        collection.set_clim(-limit, limit)
    axis.add_collection3d(collection)
    rendered_vertices = triangles.reshape(-1, 3)
    for setter, values_ in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), rendered_vertices.T):
        lower, upper = float(values_.min()), float(values_.max())
        margin = max((upper - lower) * 0.03, 1e-6)
        setter(lower - margin, upper + margin)
    fig.colorbar(collection, ax=axis, label=label, shrink=0.6)
    axis.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return count


def _prepare_surface_face_fields(vertices: np.ndarray, faces: np.ndarray,
                                 point_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the native postprocess backend once for all rendered scalar fields."""
    try:
        import torch
        from ..models import _require_backend

        prepared = _require_backend().prepare_surface_face_field(
            torch.as_tensor(vertices, dtype=torch.float64),
            torch.as_tensor(faces, dtype=torch.int64),
            torch.as_tensor(point_values, dtype=torch.float64))
        return (prepared.face_centers.numpy(), prepared.face_values.numpy(),
                prepared.valid_faces.numpy().astype(bool))
    except (ImportError, AttributeError):
        valid = ((faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1) &
                 np.isfinite(vertices[faces]).all(axis=(1, 2)) &
                 np.isfinite(point_values[faces]).all(axis=(1, 2)))
        centers = np.full((len(faces), 3), np.nan, dtype=np.float64)
        values = np.full((len(faces), point_values.shape[1]), np.nan, dtype=np.float64)
        centers[valid] = vertices[faces[valid]].mean(axis=1)
        values[valid] = point_values[faces[valid]].mean(axis=1)
        return centers, values, valid


def visualize_fused(fused_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Surface and displacement views of the fused point cloud."""
    fused_root = Path(fused_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deformation = np.load(fused_root / "deformation.npz")
    reference = deformation["reference_points"]
    current = deformation["current_points"]
    displacement = deformation["displacement"]
    valid = deformation["valid"].astype(bool)
    source = deformation["source_pair"].astype(np.int64)

    reference_valid = reference[valid]
    current_valid = current[valid]
    displacement_valid = displacement[valid]
    magnitude = np.linalg.norm(displacement_valid, axis=1)

    mesh_path = fused_root / "surface_mesh.npz"
    mesh = np.load(mesh_path) if mesh_path.exists() else None
    vertices = mesh["vertices"] if mesh is not None else None
    faces = mesh["faces"].astype(np.int64) if mesh is not None else None
    use_mesh = vertices is not None and faces is not None and faces.size > 0 and \
        vertices.shape == reference.shape
    outputs: dict[str, str] = {}
    reference_path = output_dir / "reference_surface.png"
    current_path = output_dir / "current_surface.png"
    magnitude_path = output_dir / "displacement_magnitude.png"
    if use_mesh:
        _, prepared_values, valid_faces = _prepare_surface_face_fields(
            vertices, faces, np.column_stack((reference[:, 2], current[:, 2], np.linalg.norm(displacement, axis=1),
                                               displacement)))
        faces = faces[valid_faces]
        prepared_values = prepared_values[valid_faces]
        rendered_faces = _mesh3d(vertices, faces, prepared_values[:, 0], "Fused reference mesh (colored by Z)",
                                 reference_path, label="Z")
        _mesh3d(current, faces, prepared_values[:, 1], "Fused deformed mesh (colored by Z)", current_path, label="Z")
        _mesh3d(vertices, faces, prepared_values[:, 2], "Displacement magnitude mesh (mm)",
                magnitude_path, label="|dX| (mm)")
    else:
        rendered_faces = 0
        _scatter3d(reference_valid, reference_valid[:, 2], "Fused reference surface (colored by Z)",
                   reference_path, label="Z")
        _scatter3d(current_valid, current_valid[:, 2], "Fused deformed surface (colored by Z)",
                   current_path, label="Z")
        _scatter3d(reference_valid, magnitude, "Displacement magnitude (mm)",
                   magnitude_path, label="|dX| (mm)")
    outputs.update(reference_surface=str(reference_path), current_surface=str(current_path),
                   displacement_magnitude=str(magnitude_path))
    for index, name in enumerate(("U", "V", "W")):
        path = output_dir / f"displacement_{name.lower()}.png"
        if use_mesh:
            _mesh3d(vertices, faces, prepared_values[:, index + 3], f"Displacement component {name} mesh (mm)",
                    path, label=name, symmetric=True)
        else:
            _scatter3d(reference_valid, displacement_valid[:, index],
                       f"Displacement component {name} (mm)", path, label=name, symmetric=True)
        outputs[f"displacement_{name.lower()}"] = str(path)
    counts = [int((source[valid] == index).sum()) for index in range(int(source.max()) + 1)]
    summary = {"valid_points": int(valid.sum()), "points_by_source": counts,
               "displacement_magnitude_mean": float(magnitude.mean()),
               "rendering": "triangle_mesh" if use_mesh else "point_scatter",
               "rendered_faces": rendered_faces}
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
