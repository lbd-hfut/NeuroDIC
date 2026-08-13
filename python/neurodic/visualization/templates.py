"""Reusable rendering templates for NeuroDIC result products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _plt():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurodic-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _limits(values: np.ndarray, *, symmetric: bool = False,
            percentile: tuple[float, float] = (1.0, 99.0)) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    lower, upper = np.percentile(finite, percentile)
    if symmetric:
        bound = max(abs(float(lower)), abs(float(upper)), 1e-12)
        return -bound, bound
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    return (float(lower), float(upper) if upper > lower else float(lower) + 1e-12)


def _equal_3d(axis, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points): points = np.zeros((1, 3))
    low, high = points.min(0), points.max(0); center = (low + high) / 2.0
    radius = max(float((high - low).max()) / 2.0, 1e-9)
    axis.set_xlim(center[0] - radius, center[0] + radius); axis.set_ylim(center[1] - radius, center[1] + radius); axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


@dataclass(frozen=True)
class Field2DPanel:
    """One subplot in a reference-speckle / scalar-field layout."""
    reference_image: np.ndarray
    coordinates: np.ndarray
    values: np.ndarray
    title: str
    label: str
    cmap: str = "turbo"
    symmetric: bool = False
    valid: np.ndarray | None = None


def render_2d_field_overlay(panel: Field2DPanel, axis, *, alpha: float = 0.9, point_size: float = 1.0):
    """Draw one 2-D DIC field over the reference speckle image."""
    image = np.asarray(panel.reference_image); xy = np.asarray(panel.coordinates, dtype=np.float64).reshape((-1, 2)); values = np.asarray(panel.values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(xy).all(1) & np.isfinite(values)
    if panel.valid is not None: valid &= np.asarray(panel.valid, dtype=bool).reshape(-1)
    lower, upper = _limits(values[valid], symmetric=panel.symmetric)
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    plot = axis.scatter(xy[valid, 0], xy[valid, 1], c=values[valid], s=point_size, cmap=panel.cmap,
                        vmin=lower, vmax=upper, linewidths=0, alpha=alpha)
    axis.set(title=panel.title, xlim=(0, image.shape[1]), ylim=(image.shape[0], 0), aspect="equal"); axis.axis("off")
    return plot


def render_2d_field_grid(panels: Sequence[Field2DPanel], output: str | Path, *, rows: int, columns: int,
                         alpha: float = 0.9, point_size: float = 1.0, dpi: int = 170) -> Path:
    """Render caller-defined multi-panel 2-D overlays with alpha=0.9 by default."""
    if rows < 1 or columns < 1 or len(panels) > rows * columns: raise ValueError("grid dimensions must contain all 2-D panels")
    plt = _plt(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4.5 * rows), squeeze=False, constrained_layout=True)
    for axis, panel in zip(axes.flat, panels):
        plot = render_2d_field_overlay(panel, axis, alpha=alpha, point_size=point_size)
        figure.colorbar(plot, ax=axis, fraction=.04, pad=.02, label=panel.label)
    for axis in axes.flat[len(panels):]: axis.axis("off")
    figure.savefig(output, dpi=dpi); plt.close(figure); return output


def render_3d_scatter_field(points: np.ndarray, values: np.ndarray, output: str | Path, *, title: str,
                            label: str, cmap: str = "turbo", symmetric: bool = False,
                            max_points: int = 120_000, dpi: int = 170) -> Path:
    plt = _plt(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3)); values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(points).all(1) & np.isfinite(values); points, values = points[valid], values[valid]
    if len(points) > max_points:
        ids = np.linspace(0, len(points) - 1, max_points, dtype=np.int64); points, values = points[ids], values[ids]
    lower, upper = _limits(values, symmetric=symmetric)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True); axis = figure.add_subplot(projection="3d")
    plot = axis.scatter(*points.T, c=values, s=1, cmap=cmap, vmin=lower, vmax=upper, linewidths=0)
    _equal_3d(axis, points); axis.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    figure.colorbar(plot, ax=axis, shrink=.72, label=label); figure.savefig(output, dpi=dpi); plt.close(figure); return output


def render_3d_mesh_field(vertices: np.ndarray, faces: np.ndarray, values: np.ndarray, output: str | Path, *,
                         title: str, label: str, cmap: str = "turbo", symmetric: bool = False,
                         max_faces: int = 150_000, dpi: int = 170) -> Path:
    """Render a triangular 3-D scalar field; face reduction uses the native backend when available."""
    import torch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from ..models import _require_backend
    plt = _plt(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 3)); faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3)); values = np.asarray(values, dtype=np.float64).reshape(-1)
    prepared = _require_backend().prepare_surface_face_field(torch.as_tensor(vertices), torch.as_tensor(faces), torch.as_tensor(values[:, None]))
    valid = prepared.valid_faces.numpy().astype(bool); faces = faces[valid]; face_values = prepared.face_values.numpy()[valid, 0]
    if len(faces) > max_faces:
        ids = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64); faces, face_values = faces[ids], face_values[ids]
    lower, upper = _limits(face_values, symmetric=symmetric)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True); axis = figure.add_subplot(projection="3d")
    collection = Poly3DCollection(vertices[faces], linewidths=0, edgecolors="none"); collection.set_array(face_values); collection.set_cmap(cmap); collection.set_clim(lower, upper)
    axis.add_collection3d(collection); _equal_3d(axis, vertices[np.unique(faces)] if len(faces) else vertices)
    axis.set(xlabel="X", ylabel="Y", zlabel="Z", title=title); figure.colorbar(collection, ax=axis, shrink=.72, label=label)
    figure.savefig(output, dpi=dpi); plt.close(figure); return output


def render_calibration_scene(camera_centers: np.ndarray, sparse_points: np.ndarray, output: str | Path, *,
                             camera_labels: Sequence[str] = (), title: str = "Sparse points and camera poses") -> Path:
    """Calibration template: sparse structure with camera centres and labels."""
    plt = _plt(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(camera_centers, dtype=np.float64).reshape((-1, 3)); points = np.asarray(sparse_points, dtype=np.float64).reshape((-1, 3))
    figure = plt.figure(figsize=(9, 8), constrained_layout=True); axis = figure.add_subplot(projection="3d")
    if len(points):
        ids = np.linspace(0, len(points) - 1, min(len(points), 30_000), dtype=np.int64); axis.scatter(*points[ids].T, s=1, c="0.25", alpha=.55)
    if len(centers): axis.scatter(*centers.T, c="tab:red", s=45, marker="^")
    for label, center in zip(camera_labels, centers): axis.text(*center, str(label), fontsize=7)
    _equal_3d(axis, np.vstack((points, centers)) if len(points) else centers); axis.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    figure.savefig(output, dpi=170); plt.close(figure); return output
