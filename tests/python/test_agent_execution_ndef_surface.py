"""Native-free lifecycle tests for the combined NDeF surface adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.adapters.execution_ndef import (ACTION_ID, IMPLEMENTATION_ID, ROI_ACTION_ID,
                                                     ROI_IMPLEMENTATION_ID, guarded_ndef_surface_action,
                                                     managed_surface_inputs, validate_ndef_surface_outputs)
from neurodic.agent.artifacts import content_identity
from neurodic.agent.execution import _stage_signature, execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)


def _roi_dependency(managed: Path, names: list[str]) -> list[dict]:
    trial, attempt = managed / "trials" / "roi_source", "roi_attempt"
    artifact_root = trial / "artifacts" / ROI_ACTION_ID / attempt / "roi"
    _write(artifact_root / "mask_meta.json", b"{}")
    (artifact_root / "per_camera").mkdir(parents=True, exist_ok=True)
    for name in names: np.save(artifact_root / "per_camera" / f"{name}_mask.npy", np.ones((2, 3), bool))
    signature = {"stage_id": ROI_ACTION_ID, "implementation": {"adapter": ROI_IMPLEMENTATION_ID}, "scope": {"camera_ids": names}}
    records = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            records.append({"location": str(path.relative_to(trial)), "identity": content_identity(path).to_dict(),
                            "producer_action_id": ROI_ACTION_ID, "producer_signature": signature, "stage_attempt_id": attempt})
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "manifest.json").write_text(json.dumps({"trial_id": "roi_source", "stage_attempts": [{"stage_attempt_id": attempt, "status": "completed", "action_id": ROI_ACTION_ID, "producer_signature": signature}], "produced_artifacts": records}))
    required = [{"relative_path": "roi/mask_meta.json", "identity": content_identity(artifact_root / "mask_meta.json").to_dict()}]
    required += [{"relative_path": f"roi/per_camera/{name}_mask.npy", "identity": content_identity(artifact_root / "per_camera" / f"{name}_mask.npy").to_dict()} for name in names]
    return [{"dependency_id": "ndef_roi", "source_trial_id": "roi_source", "source_attempt_id": attempt,
             "producer_action_id": ROI_ACTION_ID, "producer_signature": signature, "scope": {"camera_ids": names}, "required_artifacts": required}]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[str], list[dict]]:
    case = tmp_path / "case"; names = ["cam_B", "cam_A"]  # deliberately not lexical order
    calibration = {"cameras": [{"label": name, "K": [[1,0,0],[0,1,0],[0,0,1]], "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0], "distortion": [0,0,0,0,0], "image_width": 3, "image_height": 2} for name in names], "points3d": [{"xyz": [0,0,1], "observations": [], "reprojection_error": 0.1}], "sfm_to_world_scale": 1.0}
    calibration_dir = case / "result" / "calibration"
    calibration_dir.mkdir(parents=True); (calibration_dir / "calibration_result_scaled.json").write_text(json.dumps(calibration))
    _write(calibration_dir / "observations.npz", b"observations")
    (calibration_dir / "camera_pairs.json").write_text(json.dumps({"camera_names": names, "neighbors": {"cam_B": ["cam_A"], "cam_A": ["cam_B"]}}))
    for name in names:
        _write(case / "images" / name / "000.bmp", f"{name}-reference".encode())
        _write(case / "images" / name / "001.bmp", f"{name}-frame".encode())
    (case / "configured_masks").mkdir(parents=True, exist_ok=True)
    for name in names: np.save(case / "configured_masks" / f"{name}_mask.npy", np.ones((2, 3), bool))
    _write(case / "configured_surface.npz"); _write(case / "configured_tracks.npz")
    config = {"solver": "ndef", "mode": "multiview", "case": {"root": str(case), "images": "images", "calibration": "result/calibration/calibration_result_scaled.json", "masks": "configured_masks", "reference_surface": "configured_surface.npz", "frame": -1}, "output": {"result": "result", "visualization": "visualization", "ndef_subdir": "ndef"}, "surface": {"sparse_filter": {}, "fusion_min_visible_cameras": 2, "fusion_max_points": 10}, "surface_model": {"hidden_dim": 2, "pixel_layers": 1, "camera_layers": 1, "trunk_layers": 1, "camera_embedding_dim": 2, "positional_encoding_enabled": False, "positional_encoding_num_frequencies": 1}, "surface_training": {"pretrain_iterations": 1, "pretrain_learning_rate": .1, "weight_decay": 1e-6, "device": "cpu", "smoothness_weight": 99.}, "surface_dense_training": {"enabled": True, "epochs": 1, "samples_per_camera": 2, "auto_batch": False, "spacing_px": 1, "patch_radius": 0, "learning_rate": .1, "anchor_weight": 0., "min_valid_patch_ratio": .5, "seed": 7, "prediction_batch_size": 10}, "precalculation": {"displacement": "configured_tracks.npz"}, "scale": {"sfm_to_world_scale": 99.}, "runtime": {"random_seed": 4, "deterministic": False}}
    config_path, paths_path = tmp_path / "ndef.yaml", tmp_path / "paths.yaml"
    config_path.write_text(json.dumps(config)); paths_path.write_text(json.dumps({"ndef_test": {"case": config["case"], "output": config["output"], "precalculation": config["precalculation"]}}))
    managed = tmp_path / "managed"; dependencies = _roi_dependency(managed, names)
    return config_path, paths_path, managed, names, dependencies


def _plan(fixture, trial_id: str) -> dict:
    config, paths, _managed, _names, dependencies = fixture
    return plan_trial(config, case_key="ndef_test", case_paths=paths, trial_id=trial_id,
                      override={"surface_training": {"pretrain_iterations": 2}}, restore_missing=True,
                      upstream_dependencies=dependencies).to_dict()["data"]["trial_plan"]


def _fake_surface(monkeypatch: pytest.MonkeyPatch, *, mutate: str | None = None, interrupt: bool = False, partial: bool = False):
    calls: list[dict] = []; module = types.ModuleType("neurodic.api.ndef_surface")
    def fake(config):
        calls.append(copy.deepcopy(config)); output = config["output"]; root = Path(output["result"]); pretrain, surface = root / "pretrain/surface", root / "surface"; pretrain.mkdir(parents=True); surface.mkdir(parents=True)
        if interrupt: raise KeyboardInterrupt()
        np.savez(pretrain / "surface_pretrain.npz", sparse_uv=np.ones((2,2),np.float32), sparse_camera=np.asarray([0,1]), sparse_depth=np.ones(2,np.float32), sparse_prediction=np.ones(2,np.float32), query_uv=np.ones((2,2),np.float32), query_camera=np.asarray([0,1]), query_depth=np.ones(2,np.float32), roi_uv_bounds=np.ones((2,4),np.float32), depth_mean=np.asarray(1.), depth_std=np.asarray(1.))
        (pretrain / "surface_pretrain_meta.json").write_text("{}")
        if partial: raise RuntimeError("partial export")
        history = np.ones((1,3),np.float32); np.savez(surface / "surface_dense_samples.npz", uv=np.ones((2,2),np.float32), camera=np.asarray([0,1]), targets=np.asarray([[1,-1],[-1,0]]), depth=np.ones(2,np.float32), world=np.ones((2,3),np.float32), history=history, history_columns=np.asarray(["photo_loss","anchor_loss","total_loss"]), roi_uv_bounds=np.ones((2,4),np.float32), depth_mean=np.asarray(1.), depth_std=np.asarray(1.))
        np.savez(surface / "surface_dense_field.npz", uv=np.ones((2,2),np.float32), camera=np.asarray([0,1]), depth=np.ones(2,np.float32), world=np.ones((2,3),np.float32), grid_stride=np.asarray(1), roi_uv_bounds=np.ones((2,4),np.float32), depth_mean=np.asarray(1.), depth_std=np.asarray(1.))
        visible=np.ones((1,2),bool); np.savez(surface / "deformation_surface_dataset.npz", points=np.ones((1,3),np.float32), normals=np.asarray([[1,0,0]],np.float32), source_camera=np.asarray([0],np.int16), visibility_mask=visible, projected_uv=np.ones((1,2,2),np.float32), projected_depth=np.ones((1,2),np.float32), depth_abs_error=np.ones((1,2),np.float32), visible_counts=np.asarray([2],np.int16), cam_names=np.asarray(["cam_B","cam_A"]))
        (surface / "surface_dense_meta.json").write_text("{}")
        if mutate == "bad_camera":
            with np.load(surface / "deformation_surface_dataset.npz") as payload: arrays = {key: payload[key] for key in payload.files}
            arrays["source_camera"] = np.asarray([9]); np.savez(surface / "deformation_surface_dataset.npz", **arrays)
        if mutate == "bad_history":
            with np.load(surface / "surface_dense_samples.npz") as payload: arrays = {key: payload[key] for key in payload.files}
            arrays["history"] = np.ones((1,2)); np.savez(surface / "surface_dense_samples.npz", **arrays)
    module.pretrain_ndef_surface = fake; monkeypatch.setitem(sys.modules, "neurodic.api.ndef_surface", module)
    return calls


def test_ndef_capability_and_planned_camera_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path); plan = _plan(fixture, "ndef_order")
    assert capability_for(ACTION_ID).completion_scope == "combined_action"
    assert [item["action_id"] for item in plan["execution_actions"] if item["action_id"] == ACTION_ID] == [ACTION_ID]
    assert plan["scope"]["ndef_surface_inputs"]["camera_ids"] == ["cam_B", "cam_A"]
    assert plan["plan_status"] == "ready"


def test_fake_combined_surface_lifecycle_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); calls = _fake_surface(monkeypatch)
    serialized_plan = json.loads(json.dumps(_plan(fixture, "ndef_source")))
    first = execute_trial(serialized_plan, managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert first["execution_status"] == "partial" and len(calls) == 1
    assert calls[0]["surface_training"]["weight_decay"] == 1e-6
    assert isinstance(calls[0]["surface_training"]["weight_decay"], float)
    assert calls[0]["case"]["masks"].endswith("per_camera")
    assert calls[0]["output"]["result"].startswith(str(fixture[2] / "trials/ndef_source/staging"))
    assert not list((fixture[2] / "trials/ndef_source/staging").rglob("deformation_surface_dataset.npz"))
    second = execute_trial(_plan(fixture, "ndef_target"), managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert second["stage_attempts"][0]["status"] == "reused" and len(calls) == 1


def test_string_numeric_is_rejected_before_public_surface_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    config_path = fixture[0]
    config = json.loads(config_path.read_text())
    config["surface_training"]["weight_decay"] = "1e-06"
    config_path.write_text(json.dumps(config))
    calls = _fake_surface(monkeypatch)

    execution = execute_trial(_plan(fixture, "ndef_string_weight_decay"), managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]

    assert execution["execution_status"] == "failed"
    assert calls == []
    assert "NDeF surface configuration has an invalid scalar type or value" in execution["stage_attempts"][0]["error"]


@pytest.mark.parametrize("mutate", ["bad_camera", "bad_history"])
def test_invalid_output_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: str) -> None:
    fixture = _fixture(tmp_path); _fake_surface(monkeypatch, mutate=mutate)
    report = execute_trial(_plan(fixture, f"ndef_invalid_{mutate}"), managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert report["execution_status"] == "failed"
    assert not list((fixture[2] / "trials" / f"ndef_invalid_{mutate}" / "artifacts").rglob("*"))


def test_interrupt_and_partial_export_are_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); _fake_surface(monkeypatch, interrupt=True)
    interrupted = execute_trial(_plan(fixture, "ndef_interrupt"), managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert interrupted["execution_status"] == "interrupted"
    fixture = _fixture(tmp_path / "partial"); _fake_surface(monkeypatch, partial=True)
    failed = execute_trial(_plan(fixture, "ndef_partial"), managed_root=fixture[2], action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert failed["execution_status"] == "failed" and not list((fixture[2] / "trials/ndef_partial/artifacts").rglob("*"))


def test_signature_excludes_unused_and_management_fields_but_binds_camera_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path); plan = _plan(fixture, "ndef_signature"); action = guarded_ndef_surface_action()
    from neurodic.agent.config import merge_sparse_override
    from neurodic.agent.inspect import resolve_config
    values = merge_sparse_override(resolve_config(fixture[0], case_key="ndef_test", case_paths=fixture[1])["effective_config"], plan["override"], solver="ndef")[0]
    first = _stage_signature(plan, values, action, next(item for item in plan["execution_actions"] if item["action_id"] == ACTION_ID)["covers_stages"])
    altered = copy.deepcopy(values); altered["runtime"]["random_seed"] = 99; altered["surface_training"]["smoothness_weight"] = 0.; altered["case"]["frame"] = 0
    assert first.digest == _stage_signature(plan, altered, action, ("ndef.surface",)).digest
    changed = copy.deepcopy(plan); changed["scope"]["ndef_surface_inputs"]["camera_ids"] = list(reversed(changed["scope"]["ndef_surface_inputs"]["camera_ids"]))
    assert first.digest != _stage_signature(changed, values, action, ("ndef.surface",)).digest


def test_post_plan_input_and_roi_tamper_fail_before_fake_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path); calls = _fake_surface(monkeypatch); plan = _plan(fixture, "ndef_tamper")
    calibration = Path(json.loads(fixture[0].read_text())["case"]["root"]) / "result/calibration/calibration_result_scaled.json"
    calibration.write_text(calibration.read_text() + " ")
    with pytest.raises(Exception): execute_trial(plan, managed_root=fixture[2], action_id=ACTION_ID)
    assert not calls
    fixture = _fixture(tmp_path / "roi"); calls = _fake_surface(monkeypatch); plan = _plan(fixture, "ndef_roi_tamper")
    mask = fixture[2] / "trials/roi_source/artifacts" / ROI_ACTION_ID / "roi_attempt/roi/per_camera/cam_B_mask.npy"; mask.write_bytes(b"tampered")
    with pytest.raises(Exception): execute_trial(plan, managed_root=fixture[2], action_id=ACTION_ID)
    assert not calls


def test_auto_batch_and_current_case_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path); config = json.loads(fixture[0].read_text()); config["surface_dense_training"]["auto_batch"] = True; fixture[0].write_text(json.dumps(config))
    blocked = _plan(fixture, "ndef_auto")
    assert blocked["plan_status"] == "blocked" and any(item["code"] == "NDEF.AUTO_BATCH_UNRESOLVED" for item in blocked["policy_violations"])
    from neurodic.agent.trials import plan_trial as real_plan
    current = real_plan(Path(__file__).resolve().parents[2] / "config/ndef_multi.yaml", case_key="ndef_multi", case_paths=Path(__file__).resolve().parents[2] / "config/case_paths.yaml", trial_id="ndef_current", restore_missing=True).to_dict()["data"]["trial_plan"]
    assert current["plan_status"] == "blocked"
    assert {item["code"] for item in current["policy_violations"]} >= {"NDEF.CALIBRATION_NOT_MANAGED", "NDEF.ROI_NOT_MANAGED", "NDEF.AUTO_BATCH_UNRESOLVED", "NDEF.REAL_SMOKE_UNBOUNDED"}
