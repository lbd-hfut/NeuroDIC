"""Bounded synthetic contracts for Loop 4B observability artifacts."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from neurodic.pin_multi_fusion import PINMultiFusionOptions, _fuse
from neurodic.pin_multi_quality import REASON_INDEX, compute_pair_reason_codes

class _Array:
    def __init__(self, value): self.value = np.asarray(value)
    def numpy(self): return self.value


def _stereo(*, nonfinite=False, depth=1.0, reprojection=0.0, xy=(5.0, 5.0)):
    arrays = np.array([[*xy]], dtype=float)
    disp = np.array([[np.nan, 0.0]]) if nonfinite else np.zeros((1, 2))
    field = SimpleNamespace(displacement=SimpleNamespace(values=_Array(disp)))
    return SimpleNamespace(valid=SimpleNamespace(numel=lambda: 1), reference_disparity=field,
        left_temporal=field, deformed_disparity=field, reference_points=_Array([[0., 0., depth]]),
        current_points=_Array([[0., 0., depth]]), displacement_3d=_Array(np.zeros((1, 3))),
        left_reference_coordinates=_Array(arrays), left_current_coordinates=_Array(arrays),
        right_reference_coordinates=_Array(arrays), right_current_coordinates=_Array(arrays),
        reference_reprojection_error=_Array([reprojection]), current_reprojection_error=_Array([reprojection]))


def test_stereo_reason_priority_synthetic():
    assert compute_pair_reason_codes(_stereo())[0][0] == REASON_INDEX["valid"]
    assert compute_pair_reason_codes(_stereo(nonfinite=True, depth=-1, reprojection=99))[0][0] == REASON_INDEX["invalid_field"]
    assert compute_pair_reason_codes(_stereo(depth=-1))[0][0] == REASON_INDEX["negative_depth"]
    assert compute_pair_reason_codes(_stereo(xy=(-1, 5)), image_size=(10, 10))[0][0] == REASON_INDEX["out_of_bounds"]
    assert compute_pair_reason_codes(_stereo(reprojection=9), max_reprojection_error_px=5)[0][0] == REASON_INDEX["reprojection_error"]


def _products(displacement_b=(1., 0., 0.), overlap=True):
    a = np.array([[0., 0., 0.], [2., 0., 0.]])
    b = np.array([[0.1, 0., 0.]]) if overlap else np.array([[9., 0., 0.]])
    return {"a": {"reference_points": a, "current_points": a + np.array([1., 0., 0.]), "reprojection_error": np.zeros(2)},
            "b": {"reference_points": b, "current_points": b + np.array(displacement_b), "reprojection_error": np.zeros(1)}}


def test_prefusion_consistency_and_winner_invariance(tmp_path):
    options = PINMultiFusionOptions(voxel_size=1.0, displacement_mad_factor=0.0, surface_outlier_mad_factor=0.0)
    same = _fuse(_products(), tmp_path / "same", options)
    data = np.load(tmp_path / "same" / "preselection_consistency.npz")
    assert data["disagreement_median"][0] == 0.0 and data["position_spread"][0] > 0.0
    different = _fuse(_products((3., 0., 0.)), tmp_path / "different", options)
    assert np.load(tmp_path / "different" / "preselection_consistency.npz")["disagreement_median"][0] > 0.5
    empty = _fuse(_products(overlap=False), tmp_path / "empty", options)
    assert empty["voxel_selected_points"] == 3
    assert np.load(tmp_path / "empty" / "preselection_consistency.npz")["voxel"].shape[0] == 0
    assert same["voxel_selected_points"] == different["voxel_selected_points"]
