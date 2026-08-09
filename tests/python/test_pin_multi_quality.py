"""Per-point quality reason-code tests for pairwise PIN products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurodic.api.pin_multi_slover_dic import pin_multi_slover_dic
from neurodic.pin_multi_quality import compute_pair_reason_codes, pair_quality_summary, REASON_CODES
from synthetic_dic_data import synthetic_multiview_case


def test_quality_outputs_consistent_with_cpp_valid(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=9)
    result = pin_multi_slover_dic(config, max_pairs=1)
    pair = result.pairs[0]
    case_root = Path(config["case"]["root"])
    pair_dir = case_root / "result" / "pin_multi_slover" / "pairs" / pair.pair_id

    codes = np.load(pair_dir / "quality" / "reason_codes.npy")
    quality = json.loads((pair_dir / "quality" / "quality.json").read_text(encoding="utf-8"))
    assert codes.shape[0] == int(pair.result.valid.numel())
    assert codes.dtype == np.int8
    assert int((codes == 0).sum()) == int(pair.result.valid.sum().item())
    assert sum(int(value) for value in quality["reason_codes"].values()) == int(codes.size)
    assert quality["valid_points"] == int((codes == 0).sum())
    assert "pin_diagnostics" in quality
    assert set(quality["pin_diagnostics"]) == {"reference_disparity", "left_temporal", "deformed_disparity"}

    for code, name in REASON_CODES.items():
        if name != "valid" and int((codes == code).sum()) == codes.size:
            raise AssertionError(f"all points share reason {name}; suspicious for a matched pair")


def test_shrunk_roi_produces_outside_roi_codes(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=10)
    result = pin_multi_slover_dic(config, max_pairs=1)
    pair = result.pairs[0]
    case_root = Path(config["case"]["root"])
    mask = np.load(case_root / "result" / "pin_multi_slover" / "pair_roi" / pair.pair_id / "left_mask.npy") != 0
    shrunken = mask.copy()
    shrunken[100:, :] = False

    codes, _ = compute_pair_reason_codes(pair, shrunken)
    coordinates = pair.result.left_reference_coordinates.numpy()
    below = coordinates[:, 1] >= 100
    valid_in_shrunken = (codes == 0) & below
    assert int(valid_in_shrunken.sum()) == 0
    outside = (codes == 2) & below
    assert int(outside.sum()) > 0


def test_reason_code_priority_matches_cpp_boundary(tmp_path: Path) -> None:
    config = synthetic_multiview_case(tmp_path / "case", seed=11)
    result = pin_multi_slover_dic(config, max_pairs=1)
    pair = result.pairs[0]
    summary = pair_quality_summary(pair, max_reprojection_error_px=1e-9)
    counts = summary["reason_codes"]
    assert counts["valid"] == 0
    assert counts["reprojection_error"] > 0
