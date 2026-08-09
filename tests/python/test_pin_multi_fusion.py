"""Pairwise surface fusion tests for the pin_multi_slover route."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurodic.api.pin_multi_slover_dic import pin_multi_slover_dic
from neurodic.pin_multi_fusion import _surface_inlier_mask, fuse_pin_multi_surfaces
from synthetic_dic_data import synthetic_multiview_case


def _enable_fusion(config: dict, **kwargs: object) -> dict:
    config = dict(config)
    config["fusion"] = {"enabled": True, "voxel_size": 0.05, "max_reprojection_error_px": 5.0,
                        "remove_rigid_body_motion": False, **kwargs}
    return config


def test_surface_cleaning_rejects_an_isolated_point() -> None:
    x, y = np.meshgrid(np.arange(5, dtype=np.float64), np.arange(5, dtype=np.float64))
    surface = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    points = np.vstack((surface, np.array([[100.0, 100.0, 100.0]])))
    keep, metrics = _surface_inlier_mask(points, k_neighbors=4, mad_factor=5.0)
    assert keep[:-1].all()
    assert not keep[-1]
    assert metrics["surface_neighbor_distance_threshold_mm"] is not None


def test_fusion_disabled_by_default_does_not_write_fused(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=21)
    pin_multi_slover_dic(config, max_pairs=1)
    case_root = Path(config["case"]["root"])
    assert not (case_root / "result" / "pin_multi_slover" / "fused").exists()
    manifest = json.loads((case_root / "result" / "pin_multi_slover" / "manifest.json").read_text(encoding="utf-8"))
    assert "fusion" not in manifest


def test_fusion_deduplicates_and_keeps_provenance(tmp_path: Path) -> None:
    config = _enable_fusion(synthetic_multiview_case(tmp_path / "case", seed=22))
    result = pin_multi_slover_dic(config, max_pairs=2)
    case_root = Path(config["case"]["root"])
    result_root = case_root / "result" / "pin_multi_slover"
    fused_root = result_root / "fused"

    input_points = 0
    for pair in result.pairs:
        reference = np.load(result_root / "pairs" / pair.pair_id / "reconstruct" / "reference.npz")
        current = np.load(result_root / "pairs" / pair.pair_id / "reconstruct" / "current.npz")
        valid = np.asarray(reference["valid"]).astype(bool) & np.asarray(current["valid"]).astype(bool)
        input_points += int(valid.sum())

    for name in ("reference_surface.npz", "current_surface.npz", "deformation.npz"):
        assert (fused_root / name).exists(), name
    summary = json.loads((fused_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_points"] <= input_points
    assert summary["selected_points"] > 0
    assert summary["input_points"] == input_points
    assert set(summary["points_by_source"]) == {"cam_0__cam_1", "cam_1__cam_2"}

    deformation = np.load(fused_root / "deformation.npz")
    valid = deformation["valid"].astype(bool)
    assert int(valid.sum()) == summary["selected_points"]
    assert np.all(np.isfinite(deformation["displacement"][valid]))
    sources = np.unique(deformation["source_pair"][valid])
    assert set(int(item) for item in sources) == {0, 1}
    cells = np.floor(deformation["reference_points"][valid] / summary["voxel_size"]).astype(np.int64)
    assert len(np.unique(cells, axis=0)) == summary["selected_points"]
    assert summary["deduplicated_points"] == summary["post_filter_points"] - summary["voxel_selected_points"]
    assert summary["selected_points"] == summary["voxel_selected_points"] - summary["removed_by_surface"]

    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fusion"]["selected_points"] == summary["selected_points"]


def test_fusion_rigid_body_removal_is_explicit(tmp_path: Path) -> None:
    config = _enable_fusion(synthetic_multiview_case(tmp_path / "case", seed=23),
                            remove_rigid_body_motion=True)
    pin_multi_slover_dic(config, max_pairs=1)
    case_root = Path(config["case"]["root"])
    fused_root = case_root / "result" / "pin_multi_slover" / "fused"
    summary = json.loads((fused_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["remove_rigid_body_motion"] is True
    assert "rigid_translations_per_pair" in summary
    deformation = np.load(fused_root / "deformation.npz")
    valid = deformation["valid"].astype(bool)
    median = np.median(deformation["displacement"][valid], axis=0)
    assert np.all(np.abs(median) < 0.01), f"rigid motion must be removed, median={median}"


def test_fusion_reject_reprojection_outliers(tmp_path: Path) -> None:
    config = _enable_fusion(synthetic_multiview_case(tmp_path / "case", seed=24),
                            max_reprojection_error_px=-1.0)
    pin_multi_slover_dic(config, max_pairs=1)
    case_root = Path(config["case"]["root"])
    fused_root = case_root / "result" / "pin_multi_slover" / "fused"
    summary = json.loads((fused_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_points"] == 0
    assert summary["removed_by_reprojection"]["cam_0__cam_1"] == summary["input_points"]
