"""Synthetic DIC image generators for pairwise PIN-DIC route tests.

Lightweight fixtures: random-texture planes with known pixel displacements,
checkerboard repetitive textures, unrelated textures, and occluded views.
All generators are deterministic given a seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_random_texture(shape: tuple[int, int], *, seed: int = 0, smooth_radius: int = 0) -> np.ndarray:
    """Smooth random texture normalized to float32 [0, 1]."""
    import cv2

    rng = np.random.default_rng(seed)
    height, width = shape
    noise = rng.random((height, width), dtype=np.float32)
    if smooth_radius > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), float(smooth_radius))
    return noise.astype(np.float32)


def make_checkerboard(shape: tuple[int, int], cell: int = 8, *, seed: int = 0) -> np.ndarray:
    """Periodic checkerboard used to expose repetitive-texture ambiguity."""
    rng = np.random.default_rng(seed)
    height, width = shape
    block = np.indices((cell, cell)).sum(axis=0) % 2
    pattern = np.tile(block, (height // cell + 1, width // cell + 1))[:height, :width]
    return pattern.astype(np.float32) * 0.5 + rng.random((height, width), dtype=np.float32) * 0.05


def shift_image(image: np.ndarray, dx: float, dy: float, *, border_value: float | None = None) -> np.ndarray:
    """Translate an image with constant border fill; inverse-map safe."""
    import cv2

    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    if border_value is None:
        border_value = float(image.min())
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def occluded_shift_pair(shape: tuple[int, int], dx: float, dy: float, *, seed: int = 0,
                        occlusion_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reference pair where a central square of the right image is occluded.

    Returns (left, right, meta) with ``meta["occlusion"]`` describing the
    occluded pixel rectangle; valid matches must avoid that region.
    """
    texture = make_random_texture(shape, seed=seed)
    right = shift_image(texture, dx, dy)
    height, width = shape
    half = occlusion_ratio / 2.0
    x0, x1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    y0, y1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    replacement = make_random_texture((y1 - y0, x1 - x0), seed=seed + 100)
    right[y0:y1, x0:x1] = replacement
    meta = {"occlusion": [x0, y0, x1, y1]}
    return texture, right, meta


def unrelated_texture_pair(shape: tuple[int, int], *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two statistically similar but unrelated textures (no correspondences)."""
    return make_random_texture(shape, seed=seed), make_random_texture(shape, seed=seed + 1000)


def displaced_checker_pair(shape: tuple[int, int], dx: float, dy: float, *, cell: int = 8,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard reference and a shifted version (repetitive texture)."""
    left = make_checkerboard(shape, cell=cell, seed=seed)
    return left, shift_image(left, dx, dy)


def synthetic_multiview_case(root: str | Path, *, shape: tuple[int, int] = (256, 192),
                             seed: int = 0, baseline: float = 0.2, depth: float = 5.0,
                             pixel_displacement: tuple[float, float] = (8.0, 0.0),
                             camera_count: int = 3) -> dict[str, Any]:
    """Write a synthetic multi-camera plane case and return its config mapping.

    Cameras sit on a line looking at a random-texture plane at world depth
    ``depth``; the reference frame shows the texture, the current frame is a
    rigid translation of ``pixel_displacement`` pixels.  The pipeline is
    exercised with a small, mapping-form ``pin_2d_config``.
    """
    import cv2

    case_root = Path(root)
    focal = 800.0
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    disparity = focal * float(baseline) / float(depth)
    texture = make_random_texture(shape, seed=seed)
    cameras: list[dict[str, Any]] = []
    for index in range(camera_count):
        name = f"cam_{index}"
        reference = shift_image(texture, -index * disparity, 0.0)
        current = shift_image(reference, *pixel_displacement)
        image_dir = case_root / "images" / name
        image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / "001.bmp"), (np.clip(reference, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(image_dir / "002.bmp"), (np.clip(current, 0, 1) * 255).astype(np.uint8))
        rotation = np.eye(3)
        translation = np.array([-index * float(baseline), 0.0, 0.0], dtype=np.float64)
        cameras.append({
            "label": f"images/{name}/001.bmp",
            "K": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "image_width": int(shape[1]),
            "image_height": int(shape[0]),
            "rms_error": 0.1,
            "camera_center": (-(rotation.T @ translation)).tolist(),
        })
    calibration_dir = case_root / "result" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "calibration_result_scaled.json").write_text(
        json.dumps({"cameras": cameras, "sfm_to_world_scale": 1.0}), encoding="utf-8")
    return {
        "solver": "pin_multi_slover",
        "mode": "pairwise_multiview",
        "runtime": {"random_seed": int(seed), "deterministic": True},
        "case": {"root": str(case_root), "images": "images",
                 "calibration": "result/calibration/calibration_result_scaled.json",
                 "reference_frame": "001.bmp", "frame": -1},
        "camera_pairs": {"selection": "auto_spatial_neighbors", "wrap": True, "manual": []},
        "pair_roi": {"generator": "reference_pair_sift", "output": "result/pin_multi_slover/pair_roi",
                     "feature_method": "sift", "max_features": 4000, "match_ratio": 0.75,
                     "mutual_check": True, "ransac_reprojection_threshold_px": 3.0,
                     "min_matches": 10, "support": "convex_hull", "alpha_radius_scale": 8.0,
                     "erode_pixels": 0},
        "pin_2d_config": {
            "interpolation": {"type": "bspline", "degree": 5},
            "model": {"type": "mlp", "hidden_dim": 32, "hidden_layers": 3},
            "training": {"device": "cpu", "photometric_loss": "znssd",
                         "photometric_sampling_enabled": True,
                         "photometric_iterations": 40, "seed_iterations": 15,
                         "photometric_sample_count": 2048},
        },
        "reconstruction": {"min_views": 2, "max_reprojection_error_px": 5.0,
                           "require_positive_depth": True, "require_image_bounds": True,
                           "world_scale": 1.0},
        "fusion": {"enabled": False, "remove_rigid_body_motion": False},
        "output": {"result": "result/pin_multi_slover", "visualization": "visualization/pin_multi_slover"},
    }
