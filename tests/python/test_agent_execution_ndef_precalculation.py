"""Native-free lifecycle and contract tests for Adapter E1."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.adapters.execution_ndef_precalculation import (
    ACTION_ID,
    IMPLEMENTATION_ID,
    INPUTS_KEY,
    guarded_ndef_precalculation_action,
    validate_ndef_precalculation_outputs,
)
from neurodic.agent.artifacts import content_identity
from neurodic.agent.execution import execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial


NAMES = ["cam_B", "cam_A"]  # catches accidental lexical camera ordering


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    case = tmp_path / "case"
    calibration_dir = case / "result" / "calibration"
    calibration_dir.mkdir(parents=True)
    cameras = [{"label": name, "K": [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [float(index), 0.0, 0.0], "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "image_width": 4, "image_height": 3} for index, name in enumerate(NAMES)]
    (calibration_dir / "calibration_result_scaled.json").write_text(json.dumps({
        "cameras": cameras, "points3d": [{"xyz": [0.0, 0.0, 1.0]}], "sfm_to_world_scale": 1.0}), encoding="utf-8")
    _write(calibration_dir / "observations.npz", b"observations")
    (calibration_dir / "camera_pairs.json").write_text(json.dumps({
        "camera_names": NAMES, "neighbors": {NAMES[0]: [NAMES[1]], NAMES[1]: [NAMES[0]]}}), encoding="utf-8")
    for name in NAMES:
        _write(case / "images" / name / "000.bmp", f"{name}-reference".encode())
        _write(case / "images" / name / "001.bmp", f"{name}-current".encode())

    config = {
        "solver": "ndef", "mode": "multiview",
        "runtime": {"random_seed": 23, "deterministic": True},
        "case": {"root": str(case), "images": "images", "calibration": "result/calibration/calibration_result_scaled.json",
                 "masks": "configured_masks", "reference_surface": "configured_surface.npz", "frame": -1},
        "output": {"result": "result", "visualization": "visualization", "ndef_subdir": "ndef"},
        "surface": {"max_points": 4},
        "precalculation": {"displacement": "configured_tracks.npz", "sparse": {
            "points_per_camera": 2, "neighbors_per_camera": 1, "patch_radius": 1,
            "cross_search_radius": 2, "temporal_search_radius": 2,
            "cross_ncc_threshold": 0.45, "temporal_ncc_threshold": 0.55,
            "min_texture_std": 0.02, "max_reprojection_error": 3.0,
            "displacement_mad_threshold": 5.0, "match_batch_size": 2, "device": "cpu"}},
    }
    config_path = tmp_path / "ndef.yaml"
    paths_path = tmp_path / "case_paths.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    paths_path.write_text(json.dumps({"ndef_test": {"case": config["case"], "output": config["output"],
                                                       "precalculation": config["precalculation"]}}), encoding="utf-8")

    managed = tmp_path / "managed"
    dependencies: list[dict] = []
    dependencies.append(_producer(managed, "roi_source", "roi_attempt", "ndef.roi.generate_call",
                                  "neurodic.ndef.roi/v1", "ndef_roi", _roi_artifacts, NAMES))
    dependencies.append(_producer(managed, "surface_source", "surface_attempt", "ndef.combined_surface_call",
                                  "neurodic.ndef.surface_combined/v1", "ndef_surface", _surface_artifacts, NAMES))
    return config_path, paths_path, managed, dependencies


def _producer(managed: Path, trial_id: str, attempt_id: str, action: str, implementation: str,
              dependency_id: str, builder, names: list[str]) -> dict:
    trial = managed / "trials" / trial_id
    root = trial / "artifacts" / action / attempt_id
    root.mkdir(parents=True, exist_ok=True)
    builder(root, names)
    signature = {"stage_id": action, "implementation": {"adapter": implementation}, "scope": {}}
    records = []
    required = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        record = {"location": str(path.relative_to(trial)), "identity": content_identity(path).to_dict(),
                  "producer_action_id": action, "producer_signature": signature, "stage_attempt_id": attempt_id}
        records.append(record)
        if dependency_id == "ndef_roi" and relative.startswith("roi/per_camera/") and relative.endswith(".npy"):
            required.append({"relative_path": relative, "identity": record["identity"]})
        if dependency_id == "ndef_surface" and relative == "surface/deformation_surface_dataset.npz":
            required.append({"relative_path": relative, "identity": record["identity"]})
    (trial / "manifest.json").write_text(json.dumps({"trial_id": trial_id,
        "stage_attempts": [{"stage_attempt_id": attempt_id, "status": "completed", "action_id": action,
                            "producer_signature": signature}], "produced_artifacts": records}), encoding="utf-8")
    if dependency_id == "ndef_roi":
        by_path = {item["relative_path"]: item for item in required}
        required = [by_path[f"roi/per_camera/{name}_mask.npy"] for name in names]
    return {"dependency_id": dependency_id, "source_trial_id": trial_id, "source_attempt_id": attempt_id,
            "producer_action_id": action, "producer_signature": signature, "scope": {},
            "required_artifacts": required}


def _roi_artifacts(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / "roi" / "per_camera" / f"{name}_mask.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ones((3, 4), dtype=bool))


def _surface_artifacts(root: Path, names: list[str]) -> None:
    points = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float64)
    visibility = np.ones((2, len(names)), dtype=bool)
    uv = np.ones((2, len(names), 2), dtype=np.float64)
    (root / "surface").mkdir(parents=True, exist_ok=True)
    np.savez(root / "surface" / "deformation_surface_dataset.npz", points=points,
             visibility_mask=visibility, projected_uv=uv,
             normals=np.ones_like(points), source_camera=np.zeros(2, dtype=np.int64),
             projected_depth=np.ones((2, len(names))), depth_abs_error=np.zeros((2, len(names))),
             visible_counts=np.full(2, len(names), dtype=np.int64), cam_names=np.asarray(names))


def _plan(fixture: tuple[Path, Path, Path, list[dict]], trial_id: str = "e1_trial") -> dict:
    config, paths, _managed, dependencies = fixture
    return plan_trial(config, case_key="ndef_test", case_paths=paths, trial_id=trial_id,
                      scope={"ndef_precalculation_only": True}, restore_missing=True,
                      upstream_dependencies=dependencies).data["trial_plan"]


def _fake_precalculation(monkeypatch: pytest.MonkeyPatch, *, mode: str | None = None) -> list[dict]:
    calls: list[dict] = []
    module = types.ModuleType("neurodic.api.ndef_dic")

    def fake(config, write_case_artifacts=True):
        calls.append(copy.deepcopy(config))
        if mode == "runtime":
            raise RuntimeError("fake scientific failure")
        if mode == "interrupt":
            raise KeyboardInterrupt()
        output = Path(config["output"]["result"]) / "precalculation"
        output.mkdir(parents=True, exist_ok=True)
        reference = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        current = reference + np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        displacement = current - reference
        np.savez(output / "sparse_tracks.npz", source_camera=np.asarray([0, 1], dtype=np.int64),
                 source_uv=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64), reference_points=reference,
                 current_points=current, displacement=displacement,
                 displacement_magnitude=np.linalg.norm(displacement, axis=1),
                 camera_count=np.asarray([2, 2], dtype=np.int64),
                 reference_reprojection_error=np.asarray([0.1, 0.2], dtype=np.float64),
                 current_reprojection_error=np.asarray([0.1, 0.2], dtype=np.float64),
                 mean_match_score=np.asarray([0.8, 0.9], dtype=np.float64),
                 inlier_mask=np.asarray([True, False], dtype=bool))
        if mode == "partial":
            raise RuntimeError("partial export")
        (output / "sparse_scale.json").write_text(json.dumps({
            "scale_stats": {"median": 0.1, "mean": 0.15, "p75": 0.2, "p90": 0.2, "max": 0.2},
            "n_tracks": 2, "n_inliers": 1,
            "per_camera": [{"camera": NAMES[0], "requested_seeds": 2, "triangulated_tracks": 1},
                            {"camera": NAMES[1], "requested_seeds": 2, "triangulated_tracks": 1}],
            "sampling": {"method": "fake", "without_replacement": True, "random_seed": 23, "min_texture_std": 0.02},
            "coordinate_unit": "input camera/surface unit"}), encoding="utf-8")

    module.ndef_sparse_precalculation = fake
    monkeypatch.setitem(sys.modules, "neurodic.api.ndef_dic", module)
    return calls


def test_precalculation_plan_and_capability_are_independent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)
    assert plan["plan_status"] == "ready"
    assert [item["action_id"] for item in plan["execution_actions"]] == [ACTION_ID]
    assert plan["scope"][INPUTS_KEY]["camera_ids"] == NAMES
    assert capability_for(ACTION_ID).completion_scope == "requested_action_only"
    assert capability_for(ACTION_ID).execution_supported is True


def test_fake_success_binds_managed_inputs_and_publishes_only_two_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); config, paths, managed, dependencies = fixture
    calls = _fake_precalculation(monkeypatch)
    plan = _plan(fixture)
    result = execute_trial(plan, managed_root=managed)
    assert result.status == "ok"
    execution = result.data["execution"]
    assert execution["execution_status"] == "completed"
    assert len(execution["produced_artifacts"]) == 2
    assert calls and calls[0]["case"]["reference_surface"].endswith("surface/deformation_surface_dataset.npz")
    assert Path(calls[0]["case"]["masks"]).name == "per_camera"
    assert calls[0]["output"]["ndef_subdir"] is None
    assert calls[0]["precalculation"]["sparse"]["device"] == "cpu"
    assert calls[0]["precalculation"]["sparse"]["random_seed"] if "random_seed" in calls[0]["precalculation"]["sparse"] else True
    assert not (Path(execution["stage_attempts"][0]["staging_root"]).exists())


def test_runtime_keyboard_interrupt_and_partial_export_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for mode, expected in (("runtime", "failed"), ("interrupt", "interrupted"), ("partial", "failed")):
        fixture = _fixture(tmp_path / mode); calls = _fake_precalculation(monkeypatch, mode=mode)
        result = execute_trial(_plan(fixture, f"e1_{mode}"), managed_root=fixture[2])
        assert result.status == "warning"
        assert result.data["execution"]["stage_attempts"][0]["status"] == expected
        assert result.data["execution"]["produced_artifacts"] == []
        assert len(calls) == 1


def test_safe_reuse_does_not_invoke_scientific_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); calls = _fake_precalculation(monkeypatch)
    first = execute_trial(_plan(fixture, "e1_first"), managed_root=fixture[2])
    assert first.status == "ok"
    calls.clear()
    second = execute_trial(_plan(fixture, "e1_second"), managed_root=fixture[2])
    assert second.status == "ok"
    assert second.data["execution"]["stage_attempts"][0]["status"] == "reused"
    assert calls == []


def test_toc_tou_surface_roi_and_image_changes_are_rejected_before_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); plan = _plan(fixture); calls = _fake_precalculation(monkeypatch)
    surface = next(path for path in (fixture[2] / "trials" / "surface_source").rglob("deformation_surface_dataset.npz"))
    surface.write_bytes(surface.read_bytes() + b"tamper")
    with pytest.raises(Exception):
        execute_trial(plan, managed_root=fixture[2])
    assert calls == []

    fixture = _fixture(tmp_path / "roi"); plan = _plan(fixture); calls = _fake_precalculation(monkeypatch)
    mask = next(path for path in (fixture[2] / "trials" / "roi_source").rglob("cam_B_mask.npy")); mask.unlink()
    with pytest.raises(Exception):
        execute_trial(plan, managed_root=fixture[2])
    assert calls == []

    fixture = _fixture(tmp_path / "image"); plan = _plan(fixture); calls = _fake_precalculation(monkeypatch)
    image = fixture[0].parent / "case" / "images" / "cam_B" / "001.bmp"
    image.write_bytes(b"changed")
    with pytest.raises(Exception):
        execute_trial(plan, managed_root=fixture[2])
    assert calls == []


@pytest.mark.parametrize("suffix", [".png", ".bmp"])
def test_missing_managed_npy_never_falls_back_to_image_or_all_one_mask(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch,
                                                                        suffix: str) -> None:
    fixture = _fixture(tmp_path); calls = _fake_precalculation(monkeypatch)
    mask = next(path for path in (fixture[2] / "trials" / "roi_source").rglob("cam_B_mask.npy")); mask.unlink()
    _write(mask.with_suffix(suffix), b"fallback-image")
    with pytest.raises(Exception):
        execute_trial(_plan(fixture), managed_root=fixture[2])
    assert calls == []


def test_auto_device_and_missing_sparse_options_are_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    values = json.loads(fixture[0].read_text()); values["precalculation"]["sparse"]["device"] = "auto"
    fixture[0].write_text(json.dumps(values))
    paths = json.loads(fixture[1].read_text()); paths["ndef_test"]["precalculation"] = values["precalculation"]
    fixture[1].write_text(json.dumps(paths))
    blocked = _plan(fixture)
    assert blocked["plan_status"] == "blocked"
    assert any("DEVICE_UNRESOLVED" in item["code"] for item in blocked["policy_violations"])


def test_validator_rejects_uncontrolled_files_and_inconsistent_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path); plan = _plan(fixture); inputs = plan["scope"][INPUTS_KEY]
    root = tmp_path / "staging"; root.mkdir()
    (root / "unexpected.txt").write_text("no")
    with pytest.raises(ValueError, match="UNCONTROLLED_OUTPUT"):
        validate_ndef_precalculation_outputs(root, json.loads(fixture[0].read_text()), inputs)


def test_no_baseline_case_or_config_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); config, paths, managed, _ = fixture
    before = {path: path.read_bytes() for path in (config, paths, config.parent / "case" / "result" / "calibration" / "calibration_result_scaled.json")}
    _fake_precalculation(monkeypatch)
    execute_trial(_plan(fixture), managed_root=managed)
    assert {path: path.read_bytes() for path in before} == before
