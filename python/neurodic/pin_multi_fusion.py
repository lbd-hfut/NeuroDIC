"""Pairwise surface and displacement fusion for the pin_multi_slover route.

The fusion stage is disabled by default and only runs when explicitly enabled
through ``fusion.enabled``.  It consumes the independently validated pair
products under ``result/pin_multi_slover/pairs/`` and merges them in world
coordinates: voxel-cell deduplication keeping the highest-confidence point
(lowest reprojection error), with provenance preserved per selected point.
Rigid-body removal is an explicit opt-in configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PINMultiFusionOptions:
    voxel_size: float = 1.0
    max_reprojection_error_px: float = 5.0
    remove_rigid_body_motion: bool = False
    displacement_mad_factor: float = 5.0
    surface_outlier_k_neighbors: int = 16
    surface_outlier_mad_factor: float = 5.0
    traditional_strain_neighbors: int = 12


def _options_from_config(values: Mapping[str, Any]) -> PINMultiFusionOptions:
    fusion = values.get("fusion", {})
    return PINMultiFusionOptions(
        voxel_size=float(fusion.get("voxel_size", 1.0)),
        max_reprojection_error_px=float(fusion.get("max_reprojection_error_px", 5.0)),
        remove_rigid_body_motion=bool(fusion.get("remove_rigid_body_motion", False)),
        displacement_mad_factor=float(fusion.get("displacement_mad_factor", 5.0)),
        surface_outlier_k_neighbors=int(fusion.get("surface_outlier_k_neighbors", 16)),
        surface_outlier_mad_factor=float(fusion.get("surface_outlier_mad_factor", 5.0)),
        traditional_strain_neighbors=int(values.get("traditional_strain", {}).get("neighbors", 12)),
    )


def _load_pair_products(pairs_root: Path, max_reprojection_error_px: float) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    if not pairs_root.exists():
        raise FileNotFoundError(f"No pair products found under {pairs_root}; run the pairwise solve first")
    for pair_dir in sorted(pairs_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        reference_path = pair_dir / "reconstruct" / "reference.npz"
        current_path = pair_dir / "reconstruct" / "current.npz"
        if not reference_path.exists() or not current_path.exists():
            continue
        reference = np.load(reference_path)
        current = np.load(current_path)
        valid = np.asarray(reference["valid"]).astype(bool) & np.asarray(current["valid"]).astype(bool)
        reprojection = np.maximum(np.asarray(reference["reprojection_error"], dtype=np.float64),
                                  np.asarray(current["reprojection_error"], dtype=np.float64))
        products[pair_dir.name] = {
            "reference_points": np.asarray(reference["points"], dtype=np.float64)[valid],
            "current_points": np.asarray(current["points"], dtype=np.float64)[valid],
            "reprojection_error": reprojection[valid],
        }
    return products


def fuse_pin_multi_surfaces(config: str | Path | Mapping[str, Any],
                            *, result_root: str | Path | None = None,
                            visualization_root: str | Path | None = None) -> dict[str, Any]:
    """Fuse all pair products under ``pairs/<pair_id>/`` into ``fused/``.

    ``config`` is the pin_multi_slover YAML mapping (or path); the fusion
    section controls voxel size, reprojection filtering, and explicit rigid
    motion removal.  Returns the fusion summary dict.
    """
    from .config import load_config

    values = load_config(config) if isinstance(config, (str, Path)) else config
    options = _options_from_config(values)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    if result_root is None:
        output = Path(values.get("output", {}).get("result", "result"))
        result_root = output if output.is_absolute() else root / output
    result_root = Path(result_root)
    if visualization_root is None:
        visualization = Path(values.get("output", {}).get("visualization", "visualization"))
        visualization_root = visualization if visualization.is_absolute() else root / visualization
    visualization_root = Path(visualization_root)
    pairs_root = result_root / "pairs"
    fused_root = result_root / "fused"
    products = _load_pair_products(pairs_root, options.max_reprojection_error_px)
    if not products:
        raise ValueError("fusion found no valid pair products; check pair ROIs and solves")
    summary = _fuse(products, fused_root, options)
    try:
        from .visualization.pin_multi import visualize_fused, visualize_fused_ground_truth_error

        visualize_fused(fused_root, visualization_root / "fused")
        ground_truth = root / "ground_truth"
        if (ground_truth / "points_ref.npy").exists() and \
           (ground_truth / "displacement_step1.npy").exists():
            error = visualize_fused_ground_truth_error(fused_root, ground_truth,
                                                       visualization_root / "fused")
            summary["ground_truth"] = error
            (fused_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:  # Visualization must not break the numerical pipeline.
        summary["visualization_error"] = str(exc)
    return summary


def _fuse(products: dict[str, dict[str, Any]], output_dir: Path,
          options: PINMultiFusionOptions) -> dict[str, Any]:
    pair_names = list(products)
    candidates_reference: list[np.ndarray] = []
    candidates_current: list[np.ndarray] = []
    candidates_source: list[np.ndarray] = []
    candidates_reprojection: list[np.ndarray] = []
    removed_by_reprojection: dict[str, int] = {}
    removed_by_displacement: dict[str, int] = {}
    rigid_translations: dict[str, list[float]] = {}
    total_input = 0

    raw_magnitudes = [np.linalg.norm(product["current_points"] - product["reference_points"], axis=1)
                      for product in products.values()]
    raw = np.concatenate(raw_magnitudes) if raw_magnitudes else np.zeros(0)
    if options.displacement_mad_factor > 0.0 and raw.size:
        median = float(np.median(raw))
        mad = float(np.median(np.abs(raw - median)))
        threshold = float(median + options.displacement_mad_factor * mad)
        if not np.isfinite(threshold) or threshold <= 0.0:
            threshold = np.inf
    else:
        median, mad, threshold = 0.0, 0.0, np.inf

    for name in pair_names:
        product = products[name]
        total_input += int(product["reference_points"].shape[0])
        keep = product["reprojection_error"] <= options.max_reprojection_error_px
        removed_by_reprojection[name] = int((~keep).sum())
        reference = product["reference_points"][keep]
        current = product["current_points"][keep]
        reprojection = product["reprojection_error"][keep]

        displacement = current - reference
        displacement_keep = np.linalg.norm(displacement, axis=1) <= threshold
        removed_by_displacement[name] = int((~displacement_keep).sum())
        reference = reference[displacement_keep]
        current = current[displacement_keep]
        reprojection = reprojection[displacement_keep]
        displacement = displacement[displacement_keep]
        if options.remove_rigid_body_motion and displacement.shape[0]:
            rigid = np.median(displacement, axis=0)
            displacement = displacement - rigid
            current = reference + displacement
            rigid_translations[name] = rigid.tolist()

        if reference.shape[0] == 0:
            continue
        candidates_reference.append(reference)
        candidates_current.append(current)
        candidates_source.append(np.full(reference.shape[0], pair_names.index(name), dtype=np.int64))
        candidates_reprojection.append(reprojection)

    if not candidates_reference:
        fused_reference = np.zeros((0, 3), dtype=np.float64)
        fused_current = np.zeros((0, 3), dtype=np.float64)
        fused_source = np.zeros(0, dtype=np.int64)
        fused_reprojection = np.zeros(0, dtype=np.float64)
    else:
        reference = np.concatenate(candidates_reference)
        current = np.concatenate(candidates_current)
        source = np.concatenate(candidates_source)
        reprojection = np.concatenate(candidates_reprojection)
        cells = np.floor(reference / options.voxel_size).astype(np.int64)
        # Sort all pairs together, placing the lowest reprojection-error point
        # first within each exact 3-D voxel coordinate.  Do not hash voxel
        # coordinates into one integer: that can collide for large/negative
        # calibration-world coordinates.
        order = np.lexsort((reprojection, cells[:, 2], cells[:, 1], cells[:, 0]))
        ordered_cells = cells[order]
        first = np.empty(order.shape[0], dtype=bool)
        first[0] = True
        first[1:] = np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)
        selected = order[first]
        fused_reference = reference[selected]
        fused_current = current[selected]
        fused_source = source[selected]
        fused_reprojection = reprojection[selected]

    voxel_selected_points = int(fused_reference.shape[0])
    import torch
    from .models import _require_backend
    surface_cleanup = _require_backend().clean_pin_multi_surface(
        torch.as_tensor(fused_reference, dtype=torch.float64),
        int(options.surface_outlier_k_neighbors), float(options.surface_outlier_mad_factor))
    surface_keep = surface_cleanup.inlier_mask.numpy().astype(bool)
    surface_metrics: dict[str, float | int | None] = {
        "surface_outlier_k_neighbors": int(options.surface_outlier_k_neighbors),
        "surface_outlier_mad_factor": float(options.surface_outlier_mad_factor),
        "surface_neighbor_distance_median_mm": float(surface_cleanup.neighbor_distance_median),
        "surface_neighbor_distance_mad_mm": float(surface_cleanup.neighbor_distance_mad),
        "surface_neighbor_distance_threshold_mm": float(surface_cleanup.neighbor_distance_threshold),
        "surface_plane_residual_median_mm": float(surface_cleanup.plane_residual_median),
        "surface_plane_residual_mad_mm": float(surface_cleanup.plane_residual_mad),
        "surface_plane_residual_threshold_mm": float(surface_cleanup.plane_residual_threshold),
    }
    removed_by_surface = int((~surface_keep).sum())
    fused_reference = fused_reference[surface_keep]
    fused_current = fused_current[surface_keep]
    fused_source = fused_source[surface_keep]
    fused_reprojection = fused_reprojection[surface_keep]

    # Strain is estimated only after all fusion cleanup has completed.  This
    # prevents isolated/off-surface points from destabilising local gradients.
    if fused_reference.shape[0]:
        strain = _require_backend().compute_traditional_strain_3d(
            torch.as_tensor(fused_reference, dtype=torch.float64),
            torch.as_tensor(fused_current - fused_reference, dtype=torch.float64),
            torch.ones(fused_reference.shape[0], dtype=torch.bool), options.traditional_strain_neighbors).numpy()
    else:
        strain = np.empty((0, 6), dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "reference_surface.npz", points=fused_reference, valid=np.ones(fused_reference.shape[0], bool),
             reprojection_error=fused_reprojection, source_pair=fused_source, pair_names=pair_names,
             voxel_size=options.voxel_size)
    np.savez(output_dir / "current_surface.npz", points=fused_current, valid=np.ones(fused_current.shape[0], bool),
             reprojection_error=fused_reprojection, source_pair=fused_source, pair_names=pair_names,
             voxel_size=options.voxel_size)
    np.savez(output_dir / "deformation.npz", coordinates=fused_reference, reference_points=fused_reference,
             current_points=fused_current, displacement=fused_current - fused_reference,
             valid=np.ones(fused_reference.shape[0], bool), source_pair=fused_source, pair_names=pair_names,
             voxel_size=options.voxel_size)
    np.savez(output_dir / "strain.npz", coordinates=fused_reference, strain=strain,
             valid=np.isfinite(strain).all(axis=1),
             strain_components=np.asarray(["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_xz"]),
             source_pair=fused_source, pair_names=pair_names, voxel_size=options.voxel_size)
    if fused_reference.shape[0]:
        mesh_options = _require_backend().SurfaceMeshOptions()
        mesh_options.k_neighbors = int(options.surface_outlier_k_neighbors)
        mesh = _require_backend().triangulate_pin_multi_surface(
            torch.as_tensor(fused_reference, dtype=torch.float64), mesh_options)
        mesh_cleanup = _require_backend().clean_pin_multi_mesh(mesh.vertices, mesh.faces, mesh.quality)
        face_mask = mesh_cleanup.face_mask.numpy().astype(bool)
        np.savez(output_dir / "surface_mesh.npz", vertices=mesh.vertices.numpy(), faces=mesh.faces.numpy()[face_mask],
                 normals=mesh.normals.numpy(), quality=mesh.quality.numpy()[face_mask], face_mask=face_mask,
                 median_spacing=mesh.median_spacing, max_edge_length=mesh.max_edge_length,
                 mean_edge_length=mesh_cleanup.mean_edge_length, overlap_distance=mesh_cleanup.overlap_distance)

    summary: dict[str, Any] = {
        "voxel_size": options.voxel_size,
        "max_reprojection_error_px": options.max_reprojection_error_px,
        "displacement_mad_factor": options.displacement_mad_factor,
        "displacement_median_mm": None if raw.size == 0 else float(median),
        "displacement_mad_mm": None if raw.size == 0 else float(mad),
        "displacement_threshold_mm": None if raw.size == 0 else float(threshold),
        "remove_rigid_body_motion": options.remove_rigid_body_motion,
        "input_points": int(total_input),
        "post_filter_points": int(sum(item.shape[0] for item in candidates_reference)),
        "voxel_selected_points": voxel_selected_points,
        "deduplicated_points": int(sum(item.shape[0] for item in candidates_reference) - voxel_selected_points),
        "removed_by_surface": removed_by_surface,
        "removed_by_reprojection": removed_by_reprojection,
        "removed_by_displacement": removed_by_displacement,
        "selected_points": int(fused_reference.shape[0]),
        "points_by_source": {name: int((fused_source == index).sum())
                             for index, name in enumerate(pair_names)},
        "coordinate_frame": "calibration world frame",
        "traditional_strain_neighbors": options.traditional_strain_neighbors,
    }
    summary.update(surface_metrics)
    if options.remove_rigid_body_motion:
        summary["rigid_translations_per_pair"] = rigid_translations
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
