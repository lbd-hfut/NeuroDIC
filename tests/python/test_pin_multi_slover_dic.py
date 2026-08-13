"""End-to-end pairwise multi-camera PIN-DIC workflow tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurodic.api.pin_multi_slover_dic import pin_multi_slover_dic
from synthetic_dic_data import synthetic_multiview_case


def test_synthetic_multiview_case_output_contract(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=5)
    assert "pairwise_pin" not in config
    assert config["pin_2d_config"]["training"]["photometric_iterations"] == 40
    result = pin_multi_slover_dic(config, max_pairs=2)
    assert len(result.pairs) == 2
    assert [pair.pair_id for pair in result.pairs] == ["cam_0__cam_1", "cam_1__cam_2"]
    for pair in result.pairs:
        for field in (pair.result.reference_disparity, pair.result.left_temporal,
                      pair.result.deformed_disparity):
            assert field.diagnostics.iterations == 40

    case_root = Path(config["case"]["root"])
    result_root = case_root / "result" / "pin_multi_slover"
    for pair_id in ("cam_0__cam_1", "cam_1__cam_2"):
        for name in ("reference_disparity", "left_temporal", "deformed_disparity"):
            assert (result_root / "pairs" / pair_id / "disp" / f"{name}.npz").exists(), name
        for name in ("reference", "current"):
            assert (result_root / "pairs" / pair_id / "reconstruct" / f"{name}.npz").exists(), name
        assert (result_root / "pairs" / pair_id / "deformation" / "initial_to_current.npz").exists()
        summary_path = result_root / "pairs" / pair_id / "deformation" / "initial_to_current_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["valid_points"] > 0
        assert 0.0 < summary["valid_ratio"] <= 1.0
        assert summary["reference_mean_reprojection_error_px"] >= 0.0

    deformation = np.load(result_root / "pairs" / "cam_0__cam_1" / "deformation" / "initial_to_current.npz")
    assert "strain" not in deformation.files
    valid = deformation["valid"].astype(bool)
    displacement = deformation["displacement"][valid]
    assert displacement.size > 0
    assert np.all(np.isfinite(displacement))
    expected = 8.0 * 5.0 / 800.0
    assert abs(float(np.mean(displacement[:, 0])) - expected) < 0.04
    assert abs(float(np.mean(displacement[:, 1]))) < 0.03
    assert abs(float(np.mean(displacement[:, 2]))) < 0.03

    reference = np.load(result_root / "pairs" / "cam_0__cam_1" / "reconstruct" / "reference.npz")
    assert np.all(np.isfinite(reference["points"][valid]))
    assert abs(float(np.median(reference["points"][valid, 2])) - 5.0) < 0.5

    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["route"] == "pin_multi_slover"
    assert "solve" in manifest
    assert len(manifest["solve"]["pairs"]) == 2
    assert manifest["solve"]["world_scale"] == 1.0

    visualization = case_root / "visualization" / "pin_multi_slover"
    assert (visualization / "pairs" / "cam_0__cam_1" / "disp" / "reference_disparity.png").exists()
    assert (visualization / "pairs" / "cam_0__cam_1" / "reconstruct" / "reference.png").exists()
    assert (visualization / "pairs" / "cam_0__cam_1" / "deformation" / "initial_to_current.png").exists()


def test_synthetic_multiview_case_does_not_touch_ndef_masks(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=6)
    pin_multi_slover_dic(config, max_pairs=1)
    case_root = Path(config["case"]["root"])
    assert not (case_root / "result" / "mask" / "per_camera").exists()
    pair_roi = case_root / "result" / "pin_multi_slover" / "pair_roi"
    assert not (case_root / "result" / "pin_multi_slover" / "fused").exists()
    assert (pair_roi / "cam_0__cam_1" / "left_mask.npy").exists()
