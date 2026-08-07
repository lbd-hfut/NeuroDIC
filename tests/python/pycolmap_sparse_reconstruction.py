#!/usr/bin/env python3
"""Run a PyCOLMAP self-calibration reconstruction and compare it to NDeF dense surface.

The reconstruction deliberately receives only the reference images.  Ground-truth
camera centres are used *after* reconstruction solely to map COLMAP's arbitrary
similarity frame into the case world frame, enabling the same radial-distance
colourbar as ``dense_world_surface.png``.

Example:
  MPLCONFIGDIR=/tmp/neurodic-matplotlib \
  /home/a306/miniconda3/envs/neurodic/bin/python \
  tests/python/pycolmap_sparse_reconstruction.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _similarity_from_centres(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``target = scale * source @ rotation.T + translation`` (Umeyama)."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    if source_variance <= np.finfo(float).eps:
        raise RuntimeError("COLMAP camera centres are degenerate; cannot determine evaluation similarity")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _camera_centres(reconstruction, theoretical_centres: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source, target, names = [], [], []
    for image in reconstruction.images.values():
        name = Path(image.name).parent.name
        if not name.startswith("cam_"):
            continue
        camera_id = int(name.removeprefix("cam_"))
        if camera_id >= len(theoretical_centres):
            continue
        source.append(np.asarray(image.projection_center(), dtype=np.float64))
        target.append(theoretical_centres[camera_id])
        names.append(name)
    if len(source) < 3:
        raise RuntimeError(f"only {len(source)} registered cameras can be aligned to the case frame")
    return np.asarray(source), np.asarray(target), names


def _write_comparison(path: Path, pycolmap_sparse: np.ndarray, neurodic_sparse: np.ndarray,
                      colour_reference: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    pycolmap_radius = np.linalg.norm(pycolmap_sparse[:, [0, 2]], axis=1)
    neurodic_radius = np.linalg.norm(neurodic_sparse[:, [0, 2]], axis=1)
    reference_radius = np.linalg.norm(colour_reference[:, [0, 2]], axis=1)
    norm = Normalize(vmin=float(reference_radius.min()), vmax=float(reference_radius.max()), clip=True)
    figure = plt.figure(figsize=(16, 7), dpi=180, constrained_layout=True)
    axes = [figure.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
    for axis, points, radius, title, size in (
        (axes[0], pycolmap_sparse, pycolmap_radius, "PyCOLMAP staged self-calibrated sparse reconstruction", 4.0),
        (axes[1], neurodic_sparse, neurodic_radius, "NeuroDIC self-calibrated sparse reconstruction", 4.0),
    ):
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=radius, cmap="viridis", norm=norm,
                     s=size, linewidths=0, depthshade=False)
        axis.set(xlabel="World X", ylabel="World Y", zlabel="World Z", title=title)
        axis.set(xlim=(-90, 90), ylim=(-70, 70), zlim=(-90, 90), box_aspect=(1, 1, 1))
        axis.view_init(elev=30, azim=-60)
    colourbar = figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes, shrink=0.78, pad=0.03)
    colourbar.set_label("radial distance sqrt(World X² + World Z²) — same dense-surface scale")
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=Path("case/Multi/CylinderDIC"))
    parser.add_argument("--overwrite", action="store_true", help="replace the PyCOLMAP work directory")
    parser.add_argument("--focal-length-factor", type=float, default=1.6,
                        help="self-calibration focal-length initialization as a fraction of max image dimension")
    parser.add_argument("--staged-intrinsics", action="store_true",
                        help="fix SIMPLE_PINHOLE intrinsics while registering, then refine only focal length in global BA")
    args = parser.parse_args()

    try:
        import pycolmap
    except ModuleNotFoundError as error:
        raise SystemExit("PyCOLMAP is required: install it in the neurodic environment with `pip install pycolmap`.") from error

    root = args.case_root.resolve()
    image_root = root / "images"
    names = [str(path.relative_to(image_root)) for path in sorted(image_root.glob("cam_*/001.bmp"))]
    if len(names) < 3:
        raise RuntimeError(f"expected at least three reference images under {image_root}")
    output_name = "pycolmap_sparse_staged" if args.staged_intrinsics else "pycolmap_sparse"
    work = root / "result" / output_name
    if work.exists():
        if not args.overwrite:
            raise RuntimeError(f"{work} already exists; pass --overwrite to replace this test output")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = False
    extraction.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions()
    if args.staged_intrinsics:
        # The synthetic images are generated without distortion and use fx=fy;
        # keep this minimally-parameterized model stable during registration.
        reader.camera_model = "SIMPLE_PINHOLE"
    reader.default_focal_length_factor = args.focal_length_factor
    pycolmap.extract_features(database, image_root, names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = False
    pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.multiple_models = False
    mapping.min_model_size = 3
    mapping.random_seed = 23
    if args.staged_intrinsics:
        mapping.ba_refine_focal_length = False
        mapping.ba_refine_principal_point = False
        mapping.ba_refine_extra_params = False
        mapping.mapper.abs_pose_refine_focal_length = False
        mapping.mapper.abs_pose_refine_extra_params = False
    models = pycolmap.incremental_mapping(database, image_root, work / "models", options=mapping)
    if not models:
        raise RuntimeError("PyCOLMAP incremental mapping produced no reconstruction")
    reconstruction = max(models.values(), key=lambda item: (item.num_reg_images(), item.num_points3D()))
    if args.staged_intrinsics:
        bundle = pycolmap.BundleAdjustmentOptions()
        bundle.refine_focal_length = True
        bundle.refine_principal_point = False
        bundle.refine_extra_params = False
        pycolmap.bundle_adjustment(reconstruction, bundle)

    theoretical = np.load(root / "ground_truth" / "camera_centers.npy")
    source_centres, target_centres, registered = _camera_centres(reconstruction, theoretical)
    scale, rotation, translation = _similarity_from_centres(source_centres, target_centres)
    sparse_colmap = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
    sparse_world = scale * (sparse_colmap @ rotation.T) + translation
    dense = np.load(root / "result" / "surface" / "deformation_surface_dataset.npz")["points"]
    neurodic_calibration = json.loads((root / "result" / "calibration" / "calibration_result_scaled.json").read_text())
    neurodic_sparse = np.asarray([point["xyz"] for point in neurodic_calibration["scaled_points3d"]], dtype=np.float64)
    visualisation = root / "visualization" / "surface" / f"{output_name}_vs_neurodic_selfcal_radial_distance.png"
    _write_comparison(visualisation, sparse_world, neurodic_sparse, dense)
    np.savez_compressed(work / "sparse_world_aligned.npz", points=sparse_world,
                        points_colmap=sparse_colmap, scale=scale, rotation=rotation, translation=translation)
    radial = np.linalg.norm(sparse_world[:, [0, 2]], axis=1)
    estimated_camera = next(iter(reconstruction.cameras.values()))
    summary = {"registered_cameras": registered, "registered_camera_count": len(registered),
               "point_count": len(sparse_world), "strategy": "staged_fixed_pinhole_then_focal_ba" if args.staged_intrinsics else "unconstrained_self_calibration",
               "similarity_alignment": "COLMAP camera centers to theoretical camera centers",
               "estimated_camera": {"model": estimated_camera.model_name,
                                    "params": np.asarray(estimated_camera.params, dtype=float).tolist()},
               "radial_distance": {"min": float(radial.min()), "median": float(np.median(radial)), "max": float(radial.max())},
               "neurodic_self_calibration": {"point_count": int(len(neurodic_sparse)),
                                                "radial_distance_median": float(np.median(np.linalg.norm(neurodic_sparse[:, [0, 2]], axis=1)))},
               "visualization": str(visualisation)}
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
