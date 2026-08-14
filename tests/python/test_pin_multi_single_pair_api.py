"""Native-free contract tests for the explicit one-pair public API."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import neurodic.api.pin_multi_slover_dic as api


def _config(root: Path) -> dict:
    calibration = root / "calibration.json"
    calibration.write_text(json.dumps({"cameras": [{"label": "cam_0"}, {"label": "cam_1"}]}))
    return {"case": {"root": str(root), "images": "images"}, "reconstruction": {}, "pin_2d_config": {}}, calibration


def test_single_pair_uses_shared_assembly_and_explicit_scope(tmp_path, monkeypatch):
    values, calibration = _config(tmp_path); roi = tmp_path / "roi"; roi.mkdir(); (roi / "left_mask.npy").write_bytes(b"mask")
    calls = []
    class Problem:
        world_scale = 1.0
        require_image_bounds = True
        def set_reconstruction_options(self, *args): pass
    class Solver:
        def solve(self, _problem): return SimpleNamespace(pairs=[SimpleNamespace(pair_id="cam_0__cam_1")])
    monkeypatch.setattr(api, "_require_backend", lambda: SimpleNamespace(PINMultiProblem=Problem, PINMultiSolver=Solver))
    monkeypatch.setattr(api, "configure_runtime", lambda _v: None)
    monkeypatch.setattr(api, "_pin_2d_config", lambda _v: {})
    monkeypatch.setattr(api, "named_multiview_image_pairs", lambda *_: ([Path("l0"), Path("r0")], [[Path("lk"), Path("rk")]]))
    monkeypatch.setattr(api, "_add_resolved_pair", lambda *args, **kwargs: calls.append(kwargs) or {"roi_mask": None, "image_size": (1, 1), "max_reprojection_error_px": 5.0})
    monkeypatch.setattr(api, "_save_pair_result", lambda *_args, **_kwargs: {"valid_points": 0})
    api.solve_pin_multi_pair(values, pair_id="cam_0__cam_1", reference_camera="cam_0", secondary_camera="cam_1",
                             selected_frame=0, pair_roi_dir=roi, calibration_path=calibration,
                             result_root=tmp_path / "out", visualization_root=tmp_path / "vis")
    assert calls and calls[0]["reference_camera"] == "cam_0" and calls[0]["secondary_camera"] == "cam_1"
    with pytest.raises(ValueError):
        api.solve_pin_multi_pair(values, pair_id="cam_0__cam_1", reference_camera="cam_0", secondary_camera="cam_1",
                                 selected_frame=-1, pair_roi_dir=roi, calibration_path=calibration,
                                 result_root=tmp_path / "out2", visualization_root=tmp_path / "vis2")
