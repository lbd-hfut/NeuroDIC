"""Pair-supported automatic ROI preprocessing for multi-view NDeF-DIC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass
class NDeFROIOptions:
    outlier_k: int = 6
    outlier_knn_scale: float = 4.0
    component_radius_scale: float = 8.0
    edge_scale: float = 8.0
    radius_scale: float = 6.0
    min_hole_area: int = 500
    tiny_hole_fill_area: int = 3000
    speckle_std_ratio: float = 0.35
    speckle_lap_ratio: float = 0.35
    speckle_grad_ratio: float = 0.35
    min_speckle_std: float = 6.0
    min_speckle_lap: float = 3.0
    overlay_alpha: float = 0.45


def _read_gray(path: Path) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _shared_observations(observations: Mapping[str, np.ndarray], source: int,
                         neighbors: list[int]) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    cameras = np.asarray(observations["cam_indices"], dtype=np.int64)
    points = np.asarray(observations["point_indices"], dtype=np.int64)
    uv = np.asarray(observations["uv"], dtype=np.float64)
    source_rows = np.flatnonzero(cameras == source)
    source_points = points[source_rows]
    neighbor_sets = {neighbor: set(points[cameras == neighbor].tolist()) for neighbor in neighbors}
    support = np.zeros(len(source_rows), dtype=np.uint64)
    counts: dict[int, int] = {}
    for bit, neighbor in enumerate(neighbors):
        present = np.fromiter((point in neighbor_sets[neighbor] for point in source_points), bool, len(source_points))
        support[present] |= np.uint64(1 << bit)
        counts[neighbor] = int(present.sum())
    keep = support != 0
    kept_rows = source_rows[keep]
    kept_support = support[keep]
    if len(kept_rows):
        _, unique_positions = np.unique(points[kept_rows], return_index=True)
        unique_positions.sort()
        kept_rows, kept_support = kept_rows[unique_positions], kept_support[unique_positions]
    return uv[kept_rows], kept_support, counts


def _largest_component(points: np.ndarray, radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    pairs = list(cKDTree(points).query_pairs(radius))
    if not pairs:
        return points
    parent = np.arange(len(points))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for first, second in pairs:
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a
    roots = np.asarray([find(index) for index in range(len(points))])
    labels, counts = np.unique(roots, return_counts=True)
    selected = points[roots == labels[np.argmax(counts)]]
    return selected if len(selected) >= 3 else points


def _remove_outliers(points: np.ndarray, options: NDeFROIOptions) -> np.ndarray:
    from scipy.spatial import cKDTree
    if len(points) <= max(3, options.outlier_k + 1):
        return points
    distances, _ = cKDTree(points).query(points, k=options.outlier_k + 1)
    nearest, kth = distances[:, 1], distances[:, -1]
    positive = nearest[nearest > 0]
    scale = float(np.median(positive)) if len(positive) else float(np.median(kth))
    if not np.isfinite(scale) or scale <= 0:
        return points
    dense = points[kth <= options.outlier_knn_scale * scale]
    if len(dense) < 3:
        dense = points
    return _largest_component(dense, options.component_radius_scale * scale)


def _support_mask(points: np.ndarray, shape: tuple[int, int], options: NDeFROIOptions):
    import cv2
    from scipy.spatial import Delaunay, cKDTree
    height, width = shape
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    positive = nearest[nearest > 0]
    if not len(positive):
        return np.zeros(shape, bool), 0, 0
    scale = float(np.median(positive))
    try:
        triangles = Delaunay(points).simplices
    except Exception:
        return np.zeros(shape, bool), 0, 0
    vertices = points[triangles]
    lengths = np.stack((np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
                        np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
                        np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1)), axis=1)
    first_edge = vertices[:, 1] - vertices[:, 0]
    second_edge = vertices[:, 2] - vertices[:, 0]
    cross = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    area = np.maximum(0.5 * np.abs(cross), 1e-12)
    radius = lengths.prod(axis=1) / (4.0 * area)
    valid = (lengths.max(axis=1) < options.edge_scale * scale) & (radius < options.radius_scale * scale)
    mask = np.zeros((height, width), np.uint8)
    if valid.any():
        cv2.fillPoly(mask, vertices[valid].astype(np.int32)[:, :, None, :], 1)
    return mask.astype(bool), int(len(triangles)), int(valid.sum())


def _texture(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    import cv2
    if not mask.any():
        return {"std": 0.0, "lap_std": 0.0, "grad_mean": 0.0}
    gray = image.astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {"std": float(np.std(gray[mask])), "lap_std": float(np.std(laplacian[mask])),
            "grad_mean": float(np.mean(np.sqrt(gx * gx + gy * gy)[mask]))}


def _build_mask(image: np.ndarray, shared_uv: np.ndarray, options: NDeFROIOptions) -> dict[str, Any]:
    import cv2
    height, width = image.shape
    inside = ((shared_uv[:, 0] >= 0) & (shared_uv[:, 0] < width) &
              (shared_uv[:, 1] >= 0) & (shared_uv[:, 1] < height))
    clean = _remove_outliers(shared_uv[inside], options)
    if len(clean) < 3:
        raise ValueError(f"Only {len(clean)} pair-supported observations remain; at least three are required")
    hull = cv2.convexHull(clean.astype(np.float32)).reshape((-1, 2))
    hull_mask = np.zeros(image.shape, np.uint8)
    cv2.fillPoly(hull_mask, [np.rint(hull).astype(np.int32)], 1)
    hull_mask = hull_mask.astype(bool)
    supported, raw_triangles, valid_triangles = _support_mask(clean, image.shape, options)
    supported &= hull_mask
    holes = hull_mask & ~supported
    final, rejected = supported.copy(), np.zeros_like(supported)
    reference_texture = _texture(image, supported)
    detected = filled = rejected_count = 0
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < options.min_hole_area:
            continue
        detected += 1
        region = labels == label
        values = _texture(image, region)
        speckle = (values["std"] >= max(options.min_speckle_std, options.speckle_std_ratio * reference_texture["std"]) and
                   values["lap_std"] >= max(options.min_speckle_lap, options.speckle_lap_ratio * reference_texture["lap_std"]) and
                   values["grad_mean"] >= options.speckle_grad_ratio * reference_texture["grad_mean"])
        if area <= options.tiny_hole_fill_area or speckle:
            final[region] = True
            filled += 1
        else:
            rejected[region] = True
            rejected_count += 1
    return {"mask": final & hull_mask, "hull_mask": hull_mask, "supported_mask": supported,
            "rejected_hole_mask": rejected, "hull": hull, "clean_uv": clean,
            "raw_triangles": raw_triangles, "valid_triangles": valid_triangles,
            "holes_detected": detected, "holes_filled": filled, "holes_rejected": rejected_count,
            "reference_texture": reference_texture}


def _save_visualizations(root: Path, name: str, image: np.ndarray, built: Mapping[str, Any],
                         shared_uv: np.ndarray, support: np.ndarray, neighbor_names: list[str], alpha: float) -> None:
    import cv2
    for folder in ("mask", "overlay", "common_observations"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    mask_image = built["mask"].astype(np.uint8) * 255
    cv2.imwrite(str(root / "mask" / f"{name}_mask.png"), mask_image)
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = np.zeros_like(base)
    colors[built["mask"]] = (0, 180, 0)
    colors[built["rejected_hole_mask"]] = (0, 0, 220)
    overlay = cv2.addWeighted(base, 1.0 - alpha, colors, alpha, 0.0)
    cv2.polylines(overlay, [np.rint(built["hull"]).astype(np.int32)], True, (255, 255, 255), 2)
    cv2.imwrite(str(root / "overlay" / f"{name}_overlay.png"), overlay)
    points_image = base.copy()
    palette = [(255, 80, 0), (0, 200, 255)]
    both_color = (220, 0, 220)
    for xy, bits in zip(shared_uv, support):
        active = [bit for bit in range(len(neighbor_names)) if int(bits) & (1 << bit)]
        color = both_color if len(active) > 1 else palette[active[0] % len(palette)]
        cv2.circle(points_image, tuple(np.rint(xy).astype(int)), 2, color, -1, cv2.LINE_AA)
    y = 25
    for bit, neighbor in enumerate(neighbor_names):
        cv2.putText(points_image, f"shared with {neighbor}", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, palette[bit % len(palette)], 2, cv2.LINE_AA)
        y += 24
    if len(neighbor_names) > 1:
        cv2.putText(points_image, "shared with both", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, both_color, 2, cv2.LINE_AA)
    cv2.imwrite(str(root / "common_observations" / f"{name}_common_observations.png"), points_image)


def generate_ndef_roi(case_root: str | Path, options: NDeFROIOptions | None = None) -> dict[str, Any]:
    """Generate pair-supported masks and diagnostics for one calibrated case."""
    import cv2
    options = options or NDeFROIOptions()
    root = Path(case_root)
    calibration_dir = root / "result" / "calibration"
    pair_data = json.loads((calibration_dir / "camera_pairs.json").read_text(encoding="utf-8"))
    summary = json.loads((calibration_dir / "summary.json").read_text(encoding="utf-8"))
    loaded = np.load(calibration_dir / "observations.npz")
    observations = {key: loaded[key] for key in loaded.files}
    names = list(pair_data["camera_names"])
    image_by_name = {Path(path).parent.name: Path(path) for path in summary["image_paths"]}
    index_by_name = {name: index for index, name in enumerate(names)}
    result_root, visualization_root = root / "result" / "mask", root / "visualization" / "mask"
    per_camera = result_root / "per_camera"
    per_camera.mkdir(parents=True, exist_ok=True)
    records, masks = [], []
    for source, name in enumerate(names):
        neighbor_names = list(pair_data["neighbors"][name])
        neighbors = [index_by_name[item] for item in neighbor_names]
        shared_uv, support, shared_counts = _shared_observations(observations, source, neighbors)
        image = _read_gray(image_by_name[name])
        built = _build_mask(image, shared_uv, options)
        mask = np.asarray(built["mask"], dtype=bool)
        masks.append(mask)
        np.save(per_camera / f"{name}_mask.npy", mask)
        cv2.imwrite(str(per_camera / f"{name}_mask.png"), mask.astype(np.uint8) * 255)
        _save_visualizations(visualization_root, name, image, built, shared_uv, support, neighbor_names,
                             options.overlay_alpha)
        record = {"camera_index": source, "camera_name": name, "neighbors": neighbor_names,
                  "shared_observations_by_neighbor": {names[index]: count for index, count in shared_counts.items()},
                  "shared_observation_union": int(len(shared_uv)), "points_after_outlier_filter": int(len(built["clean_uv"])),
                  "mask_pixels": int(mask.sum()), "image_pixels": int(mask.size),
                  "mask_fraction": float(mask.mean()), "triangles_raw": built["raw_triangles"],
                  "triangles_valid": built["valid_triangles"], "holes_detected": built["holes_detected"],
                  "holes_filled": built["holes_filled"], "holes_rejected": built["holes_rejected"],
                  "reference_texture": built["reference_texture"]}
        records.append(record)
    shapes = {mask.shape for mask in masks}
    if len(shapes) == 1:
        np.savez_compressed(result_root / "masks.npz", cam_names=np.asarray(names), masks=np.stack(masks))
    metadata = {"schema_version": 1, "strategy": "union of source observations shared with inferred adjacent cameras",
                "camera_pairs": str(calibration_dir / "camera_pairs.json"), "options": asdict(options), "cameras": records}
    (result_root / "mask_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
