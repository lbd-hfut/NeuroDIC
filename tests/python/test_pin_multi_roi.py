"""Pair ROI generation tests for the independent pin_multi_slover route."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurodic.pin_multi_roi import (
    PINMultiPairROIOptions,
    generate_pin_multi_pair_roi,
    pair_id_for,
    pin_multi_pair_roi,
)
from synthetic_dic_data import (
    displaced_checker_pair,
    make_random_texture,
    occluded_shift_pair,
    shift_image,
    unrelated_texture_pair,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_image(path: Path, image: np.ndarray) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(image, 0, 1) * 255.0).astype(np.uint8))
    return path


def test_pair_id_format() -> None:
    assert pair_id_for("cam_0", "cam_1") == "cam_0__cam_1"


def test_normal_pair_matches_with_known_disparity(tmp_path: Path) -> None:
    left = make_random_texture((512, 512), seed=1)
    right = shift_image(left, 30.0, 5.0)
    left_path = _write_image(tmp_path / "left.png", left)
    right_path = _write_image(tmp_path / "right.png", right)
    output = tmp_path / "roi"
    result = generate_pin_multi_pair_roi(left_path, right_path, output)
    assert result["status"] == "ok"
    assert result["match_count"] >= 50
    assert result["mask_area_ratio"] > 0.1
    for name in ("left_mask.npy", "right_mask.npy", "left_mask.png", "right_mask.png",
                 "matches.npz", "overlay.png", "meta.json"):
        assert (output / name).exists(), name
    matches = np.load(output / "matches.npz")
    disparity = np.median(matches["right_uv"] - matches["left_uv"], axis=0)
    assert abs(disparity[0] - 30.0) < 2.0
    assert abs(disparity[1] - 5.0) < 2.0


def test_unrelated_textures_are_skipped(tmp_path: Path) -> None:
    left, right = unrelated_texture_pair((512, 512), seed=3)
    left_path = _write_image(tmp_path / "left.png", left)
    right_path = _write_image(tmp_path / "right.png", right)
    result = generate_pin_multi_pair_roi(left_path, right_path, tmp_path / "roi")
    assert result["status"] == "skipped"
    assert result["reason"] == "min_matches"
    assert result["match_count"] < 20


def test_occluded_pair_matches_avoid_occlusion(tmp_path: Path) -> None:
    dx, dy = 25.0, 0.0
    left, right, meta = occluded_shift_pair((512, 512), dx, dy, seed=7)
    left_path = _write_image(tmp_path / "left.png", left)
    right_path = _write_image(tmp_path / "right.png", right)
    result = generate_pin_multi_pair_roi(left_path, right_path, tmp_path / "roi")
    x0, y0, x1, y1 = meta["occlusion"]
    avoid_x0, avoid_x1 = int(x0 - dx), int(x1 - dx)
    if result["status"] == "ok":
        matches = np.load(tmp_path / "roi" / "matches.npz")
        inside = (matches["left_uv"][:, 0] >= avoid_x0) & (matches["left_uv"][:, 0] < avoid_x1) & \
                 (matches["left_uv"][:, 1] >= y0) & (matches["left_uv"][:, 1] < y1)
        assert not inside.any(), "matches must avoid the occluded region"


def test_repetitive_checker_texture_is_handled(tmp_path: Path) -> None:
    left, right = displaced_checker_pair((512, 512), 24.0, 0.0, seed=11)
    left_path = _write_image(tmp_path / "left.png", left)
    right_path = _write_image(tmp_path / "right.png", right)
    first = generate_pin_multi_pair_roi(left_path, right_path, tmp_path / "roi")
    second = generate_pin_multi_pair_roi(left_path, right_path, tmp_path / "roi2")
    assert first["status"] in {"ok", "skipped"}
    assert first["status"] == second["status"]
    if first["status"] == "ok":
        diagnostics = first["diagnostics"]
        assert diagnostics["ransac_matches"] <= diagnostics["mutual_matches"]
        assert diagnostics["ransac_matches"] > 0


def _cylinder_dic_config(tmp_path: Path, output_name: str) -> dict:
    case_root = ROOT / "case" / "Multi" / "CylinderDIC"
    return {
        "case": {
            "root": str(case_root),
            "images": "images",
            "calibration": "result/calibration/calibration_result_scaled.json",
            "reference_frame": "001.bmp",
        },
        "camera_pairs": {"selection": "manual", "manual": [["cam_0", "cam_1"]]},
        "pair_roi": {"output": str(tmp_path / output_name)},
    }


def test_cylinder_dic_real_pair_roi(tmp_path: Path) -> None:
    result = pin_multi_pair_roi(_cylinder_dic_config(tmp_path, "pair_roi"))
    assert result.pair_ids == ["cam_0__cam_1"]
    assert result.results[0]["status"] == "ok"
    pair_dir = result.output_root / "cam_0__cam_1"
    assert (pair_dir / "left_mask.npy").exists()
    mask = np.load(pair_dir / "left_mask.npy")
    assert mask.shape == (1080, 1440)
    assert 0.05 < float(mask.mean()) < 0.95
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["route"] == "pin_multi_slover"
    assert len(manifest["pairs"]) == 1


def test_ndef_mask_directory_is_untouched(tmp_path: Path) -> None:
    mask_dir = ROOT / "case" / "Multi" / "CylinderDIC" / "result" / "mask" / "per_camera"
    if not mask_dir.exists():
        pytest.skip("NDeF per-camera mask directory not present in this checkout")
    before = {str(path.relative_to(mask_dir)): path.read_bytes() for path in sorted(mask_dir.rglob("*")) if path.is_file()}
    pin_multi_pair_roi(_cylinder_dic_config(tmp_path, "pair_roi"))
    after = {str(path.relative_to(mask_dir)): path.read_bytes() for path in sorted(mask_dir.rglob("*")) if path.is_file()}
    assert after == before
