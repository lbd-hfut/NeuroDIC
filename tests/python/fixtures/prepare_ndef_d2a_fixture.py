"""Prepare the bounded, two-camera D2-A NDeF fixture.

This script writes only the explicitly requested synthetic fixture root (by
default below ``/tmp``).  It does not import a solver, Torch, CUDA, or the ROI
algorithm; the ROI product is produced later through the guarded action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ["cam_0", "cam_1"]
WIDTH = HEIGHT = 32


def _pgm(path: Path, offset: int) -> None:
    values = [str((x * 7 + y * 11 + offset) % 256) for y in range(HEIGHT) for x in range(WIDTH)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P2\n# bounded D2-A reference fixture\n32 32\n255\n" + " ".join(values) + "\n", encoding="ascii")


def prepare(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for camera_index, name in enumerate(NAMES):
        reference = root / "images" / name / "000.pgm"
        _pgm(reference, camera_index * 13)
        _pgm(root / "images" / name / "001.pgm", camera_index * 13 + 3)
        image_paths.append(str(reference))

    cameras = []
    points = []
    observations_camera: list[int] = []
    observations_point: list[int] = []
    observations_uv: list[list[float]] = []
    point_index = 0
    for y_index in range(5):
        for x_index in range(5):
            x = -2.0 + x_index
            y = -2.0 + y_index
            z = 10.0
            point_observations = []
            for camera_index, tx in enumerate((0.0, 0.5)):
                u = 16.0 + 20.0 * ((x + tx) / z)
                v = 16.0 + 20.0 * (y / z)
                observations_camera.append(camera_index)
                observations_point.append(point_index)
                observations_uv.append([u, v])
                point_observations.append({"camera_index": camera_index, "uv": [u, v]})
            points.append({"xyz": [x, y, z], "observations": point_observations, "reprojection_error": 0.0})
            point_index += 1
    for label, tx in zip(NAMES, (0.0, 0.5)):
        cameras.append({
            "label": label,
            "K": [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [tx, 0.0, 0.0],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_width": WIDTH,
            "image_height": HEIGHT,
        })
    calibration_dir = root / "result" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration = {"cameras": cameras, "points3d": points, "sfm_to_world_scale": 1.0}
    (calibration_dir / "calibration_result_scaled.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    np.savez_compressed(calibration_dir / "observations.npz",
                        point_indices=np.asarray(observations_point, dtype=np.int64),
                        cam_indices=np.asarray(observations_camera, dtype=np.int64),
                        uv=np.asarray(observations_uv, dtype=np.float64))
    (calibration_dir / "camera_pairs.json").write_text(json.dumps({
        "camera_names": NAMES,
        "neighbors": {"cam_0": ["cam_1"], "cam_1": ["cam_0"]},
    }, indent=2), encoding="utf-8")
    (calibration_dir / "summary.json").write_text(json.dumps({
        "image_paths": image_paths,
        "camera_names": NAMES,
        "camera_count": len(NAMES),
    }, indent=2), encoding="utf-8")
    (calibration_dir / "calibration_scale.json").write_text(json.dumps({"sfm_to_world_scale": 1.0}, indent=2), encoding="utf-8")

    config = {
        "solver": "ndef",
        "mode": "multiview",
        "runtime": {"random_seed": 20260814, "deterministic": True},
        "case": {
            "root": str(root), "images": "images",
            "calibration": "result/calibration/calibration_result_scaled.json",
            "masks": "result/ndef_d2a/roi/per_camera",
            "reference_surface": "result/ndef_d2a/surface/deformation_surface_dataset.npz",
            "frame": -1,
        },
        "output": {"result": "result", "visualization": "visualization", "ndef_subdir": "ndef_d2a"},
        "surface": {
            "max_points": 100,
            "max_reprojection_p95_px": 5.0,
            "sparse_filter": {"min_track_length": 2, "max_reprojection_error": 3.0,
                              "radius_mad_thresh": 8.0, "knn_k": 8, "knn_mad_thresh": 8.0},
            "fusion_relative_sample_spacing": 0.006,
            "fusion_depth_tolerance_factor": 1.0,
            "fusion_min_visible_cameras": 2,
            "fusion_max_points": 100,
            "fusion_candidate_spacing_factor": 0.5,
            "fusion_max_candidate_points": 500,
            "fusion_seed": 4242,
        },
        "surface_model": {"hidden_dim": 32, "pixel_layers": 3, "camera_layers": 2,
                           "trunk_layers": 3, "camera_embedding_dim": 16,
                           "positional_encoding_enabled": True,
                           "positional_encoding_num_frequencies": 4},
        "surface_training": {"device": "cuda", "pretrain_iterations": 1,
                              "pretrain_learning_rate": 0.001, "weight_decay": 0.000001,
                              "smoothness_weight": 0.0, "smooth_samples_per_camera": 256},
        "surface_dense_training": {"enabled": True, "epochs": 1, "samples_per_camera": 4,
                                    "auto_batch": False, "spacing_px": 4, "patch_radius": 2,
                                    "learning_rate": 0.0001, "anchor_weight": 0.0,
                                    "min_valid_patch_ratio": 0.5, "seed": 4242,
                                    "prediction_batch_size": 8},
        "scale": {"sfm_to_world_scale": 1.0},
        "precalculation": {"displacement": "result/ndef_d2a/precalculation/sparse_tracks.npz"},
    }
    config_path = root / "ndef_d2a.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "case_paths.yaml").write_text(json.dumps({"ndef_d2a": {
        "case": config["case"], "output": config["output"],
        "precalculation": config["precalculation"],
    }}, indent=2), encoding="utf-8")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/neurodic-d2a-ndef-smoke"))
    args = parser.parse_args()
    print(prepare(args.root))
