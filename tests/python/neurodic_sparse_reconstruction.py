#!/usr/bin/env python3
"""Quantitative radial-distance verification of the NeuroDIC self-calibrated sparse
reconstruction for the CylinderDIC case.

Loads the NeuroDIC sparse reconstruction (``calibration_result.json``) and aligns
it into the case world frame with the same Umeyama similarity used by the
PyCOLMAP reference (SfM camera centres -> theoretical camera centres, see
``pycolmap_sparse_reconstruction.py``), then reports the radial statistics of
``r = sqrt(World X**2 + World Z**2)`` (truth radius 80) and writes two
visualizations:

* a 3-D scatter plot coloured by radial distance, using the same viridis colour
  map, view and axis ranges as ``dense_world_surface.png``, with the colour bar
  fixed to the dense-surface radial range [78.27994, 83.11913] (out-of-range
  points clipped to the colour-bar ends);
* a radial-distance distribution plot (histogram + KDE) with a vertical line at
  the true radius r = 80.

Example:
  MPLCONFIGDIR=/tmp/neurodic-matplotlib \\
  /home/a306/miniconda3/envs/neurodic/bin/python \\
  tests/python/neurodic_sparse_reconstruction.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRUE_RADIUS = 80.0
COLOUR_MIN = 78.27994  # dense surface radial minimum
COLOUR_MAX = 83.11913  # dense surface radial maximum


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ``target = scale * source @ rotation.T + translation`` (Umeyama)."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _percentiles(values: np.ndarray) -> dict[str, float]:
    out = {}
    for name, percentile in (
        ("min", 0.0),
        ("p01", 1.0),
        ("p05", 5.0),
        ("p50", 50.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ):
        out[name] = float(np.percentile(values, percentile))
    out["mean"] = float(np.mean(values))
    out["std"] = float(np.std(values, ddof=1) if values.size > 1 else 0.0)
    return out


def _write_radial_scatter(points: np.ndarray, radial: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=COLOUR_MIN, vmax=COLOUR_MAX, clip=True)
    figure = plt.figure(figsize=(9, 8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=radial, cmap="viridis",
                 norm=norm, s=3.0, linewidths=0, depthshade=False)
    axis.set(xlabel="World X", ylabel="World Y", zlabel="World Z",
             title="NeuroDIC self-calibrated sparse reconstruction (radial distance)")
    axis.set(xlim=(-90, 90), ylim=(-70, 70), zlim=(-90, 90), box_aspect=(1, 1, 1))
    axis.view_init(elev=30, azim=-60)
    colourbar = figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axis, shrink=0.78, pad=0.05)
    colourbar.set_label("radial distance sqrt(World X² + World Z²) — same dense-surface scale")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_radial_distribution(radial: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.hist(radial, bins=80, density=True, alpha=0.55, color="tab:blue", label="sparse points")
    try:
        from scipy import stats

        x = np.linspace(float(np.percentile(radial, 0.5)), float(np.percentile(radial, 99.5)), 400)
        axis.plot(x, stats.gaussian_kde(radial)(x), color="tab:red", lw=1.8, label="KDE")
    except ImportError:  # pragma: no cover - scipy is optional for this diagnostic
        pass
    axis.axvline(TRUE_RADIUS, color="tab:green", ls="--", lw=1.8, label="truth radius r = 80")
    axis.set(xlabel="radial distance r = sqrt(World X² + World Z²)", ylabel="density",
             title="NeuroDIC self-calibrated sparse point radial distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=Path("case/Multi/CylinderDIC"))
    parser.add_argument("--result-subdir", type=str, default="calibration_exhaustive_staged")
    args = parser.parse_args()

    root = args.case_root.resolve()
    result_dir = root / "result" / args.result_subdir
    calibration = json.loads((result_dir / "calibration_result.json").read_text(encoding="utf-8"))
    points = np.asarray([point["xyz"] for point in calibration["points3d"]], dtype=np.float64)
    centres = np.asarray([camera["camera_center"] for camera in calibration["cameras"]], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or centres.shape != (12, 3):
        raise RuntimeError("calibration_result.json does not contain usable cameras/points")

    # Same Umeyama similarity as the PyCOLMAP reference: SfM camera centres ->
    # theoretical camera centres (radius-480 ring about the world origin).
    theoretical = np.load(root / "ground_truth" / "camera_centers.npy")
    scale, rotation, translation = _umeyama(centres, theoretical)
    points_world = scale * (points @ rotation.T) + translation
    centres_world = scale * (centres @ rotation.T) + translation
    alignment_error = np.linalg.norm(centres_world - theoretical, axis=1)

    radial = np.linalg.norm(points_world[:, [0, 2]], axis=1)
    residual = radial - TRUE_RADIUS
    stats = {
        "point_count": int(len(points)),
        "similarity_alignment": "NeuroDIC camera centers to theoretical camera centers (Umeyama)",
        "camera_center_alignment_rmse": float(np.sqrt(np.mean(np.square(alignment_error)))),
        "radial_distance": _percentiles(radial),
        "residual_r_minus_80": _percentiles(residual),
    }
    print(json.dumps(stats, indent=2))

    # ---- pipeline provenance: init pair, registration order, failures ----
    pipeline = "\n".join(calibration.get("pipeline_log", []))
    init_pairs = [line for line in calibration.get("pipeline_log", [])
                  if line.startswith("Init") and "accepted" in line]
    init_rejected = [line for line in calibration.get("pipeline_log", [])
                     if line.startswith("Init") and "rejected" in line]
    order = [attempt["image_index"] for attempt in calibration.get("registration_attempts", [])
             if attempt["success"]]
    failures = [attempt for attempt in calibration.get("registration_attempts", [])
                if not attempt["success"]]
    print("init_pair:", init_pairs[-1][:120] if init_pairs else "?")
    print("init_rejected_candidates:", len(init_rejected))
    for line in init_rejected:
        print("  ", line[:110])
    print("registration_order:", order)
    print("registration_failures:", len(failures))
    for attempt in failures[:8]:
        print(f"  img {attempt['image_index']:2d} vis={attempt['num_visible_points']:4d} "
              f"corr={attempt['num_pnp_correspondences']:4d} inl={attempt['num_pnp_inliers']:4d} "
              f"| {attempt['reason'][:60]}")

    # ---- per-stage statistics ----
    print("stage_stats:")
    for stat in calibration.get("stage_stats", []):
        print(f"  {stat['stage']:30s} reg={stat['num_registered_cameras']:2d} "
              f"pts={stat['num_points3d']:5d} obs={stat['num_observations']:6d} "
              f"rms={stat['mean_reprojection_error']:7.4f} f={stat['focal_length']:9.3f}")

    # ---- track-length statistics ----
    track_lengths = [len(point.get("observations", point.get("track", []))) for point in calibration["points3d"]]
    from collections import Counter

    length_counter = Counter(track_lengths)
    print("track_lengths:", dict(sorted(length_counter.items())))

    visualisation = root / "visualization" / args.result_subdir
    visualisation.mkdir(parents=True, exist_ok=True)
    scatter_path = visualisation / "neurodic_sparse_world_radial_distance.png"
    distribution_path = visualisation / "neurodic_radial_distance_distribution.png"
    track_path = visualisation / "neurodic_track_length_distribution.png"
    inlier_path = visualisation / "neurodic_pairwise_geometry_inliers.png"
    _write_radial_scatter(points_world, radial, scatter_path)
    _write_radial_distribution(radial, distribution_path)
    _write_track_lengths(track_lengths, calibration.get("point_diagnostics", []), track_path)
    _write_pairwise_inliers(calibration.get("inlier_match_counts", []), inlier_path)

    print(f"scatter: {scatter_path}")
    print(f"distribution: {distribution_path}")
    print(f"track_lengths: {track_path}")
    print(f"pairwise_inliers: {inlier_path}")


def _write_track_lengths(track_lengths: list[int], diagnostics: list[dict], path: Path) -> None:
    """Histogram of track lengths, coloured by creation source (if available)."""
    import matplotlib.pyplot as plt

    counts = np.bincount(np.asarray(track_lengths, dtype=int))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(np.arange(len(counts)), counts, color="tab:blue", alpha=0.75)
    axis.set(xlabel="track length (number of observations)", ylabel="point count",
             title="NeuroDIC sparse point track-length distribution")
    axis.grid(alpha=0.25, axis="y")
    for x, count in enumerate(counts):
        if count > 0:
            axis.text(x, count, str(int(count)), ha="center", va="bottom", fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_pairwise_inliers(inlier_match_counts: list[list[int]], path: Path) -> None:
    """Heat map of geometric-verification inlier counts per image pair."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(inlier_match_counts, dtype=float)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = axis.imshow(np.where(mask, matrix, np.nan), cmap="viridis")
    axis.set(xlabel="camera index (j)", ylabel="camera index (i)",
             title="Pairwise geometric-verification inliers")
    colourbar = figure.colorbar(im, ax=axis, shrink=0.85)
    colourbar.set_label("inlier matches")
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
