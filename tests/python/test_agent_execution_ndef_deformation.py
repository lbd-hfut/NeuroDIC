"""Native-free F1 lifecycle tests for the managed deformation action."""

from __future__ import annotations

import copy
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.artifacts import content_identity
from neurodic.agent.adapters.execution_ndef_deformation import (
    _checkpoint,
    _deformation_delta_allclose,
    ndef_deformation_backend_capability,
)
from neurodic.agent.execution import execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial


NAMES = ["cam_B", "cam_A"]


def _producer(managed: Path, trial_id: str, attempt_id: str, action: str, implementation: str,
              dependency_id: str, builder) -> dict:
    trial = managed / "trials" / trial_id; root = trial / "artifacts" / action / attempt_id
    root.mkdir(parents=True, exist_ok=True); builder(root)
    signature = {"stage_id": action, "implementation": {"adapter": implementation}, "scope": {}}
    records, required = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file(): continue
        rel = path.relative_to(root).as_posix(); identity = content_identity(path).to_dict()
        records.append({"location": str(path.relative_to(trial)), "identity": identity,
                        "producer_action_id": action, "producer_signature": signature,
                        "stage_attempt_id": attempt_id})
        required.append({"relative_path": rel, "identity": identity})
    (trial / "manifest.json").write_text(json.dumps({"trial_id": trial_id,
        "stage_attempts": [{"stage_attempt_id": attempt_id, "status": "completed", "action_id": action,
                            "producer_signature": signature}], "produced_artifacts": records}), encoding="utf-8")
    if dependency_id == "ndef_surface":
        required = [item for item in required if Path(item["relative_path"]).name == "deformation_surface_dataset.npz"]
    elif dependency_id == "ndef_precalculation":
        required = [item for item in required if Path(item["relative_path"]).name in {"sparse_tracks.npz", "sparse_scale.json"}]
    else:
        required_all = required
        required = [{"relative_path": "roi/mask_meta.json", "identity": next(item["identity"] for item in required_all if item["relative_path"] == "roi/mask_meta.json")}]
        required += [{"relative_path": f"roi/per_camera/{name}_mask.npy", "identity": next(item["identity"] for item in required_all if item["relative_path"] == f"roi/per_camera/{name}_mask.npy")} for name in NAMES]
    return {"dependency_id": dependency_id, "source_trial_id": trial_id, "source_attempt_id": attempt_id,
            "producer_action_id": action, "producer_signature": signature, "scope": {},
            "required_artifacts": required}


def _fixture(tmp_path: Path):
    case = tmp_path / "case"; cal = case / "result" / "calibration"; cal.mkdir(parents=True)
    cameras = [{"label": name, "K": [[4., 0., 2.], [0., 4., 1.5], [0., 0., 1.]],
                "R": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "t": [float(i), 0., 0.],
                "distortion": [0., 0., 0., 0., 0.], "image_width": 4, "image_height": 3}
               for i, name in enumerate(NAMES)]
    (cal / "calibration_result_scaled.json").write_text(json.dumps({"cameras": cameras, "sfm_to_world_scale": 1.0}), encoding="utf-8")
    (cal / "camera_pairs.json").write_text(json.dumps({"camera_names": NAMES, "neighbors": {NAMES[0]: [NAMES[1]], NAMES[1]: [NAMES[0]]}}), encoding="utf-8")
    for name in NAMES:
        (case / "images" / name).mkdir(parents=True)
        (case / "images" / name / "000.bmp").write_bytes(f"ref-{name}".encode())
        (case / "images" / name / "001.bmp").write_bytes(f"cur-{name}".encode())
    config = {"solver": "ndef", "mode": "multiview", "runtime": {"random_seed": 23, "deterministic": False},
              "case": {"root": str(case), "images": "images", "calibration": "result/calibration/calibration_result_scaled.json", "reference_surface": "legacy.npz", "masks": "legacy", "frame": -1},
              "output": {"result": "result", "visualization": "visualization", "ndef_subdir": "ndef"},
              "deformation_model": {"hidden_dim": 32, "hidden_layers": 5, "fourier_encoding": {"enabled": False, "num_frequencies": 6, "include_input": True, "angular_scale": 3.141592653589793}, "output_scale": 1.0},
              "precalculation": {"key": "displacement", "statistic": "mean", "mad_threshold": 5.0},
              "deformation_training": {"device": "cpu", "training_epochs": 1, "batch_size": 4, "max_steps_per_epoch": 1, "prediction_batch_size": 8, "seed": 23, "photometric_learning_rate": .003, "photometric_loss": "mse", "patch_radius": 2, "min_valid_patch_ratio": 1.0, "invalid_patch_penalty": .05, "smoothness_weight": 0., "weight_decay": 0.},
              "evaluation": {"enabled": False, "sample_count": 0, "seed": 0}}
    config_path = tmp_path / "ndef.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
    paths_path = tmp_path / "paths.json"; paths_path.write_text(json.dumps({"f_test": config}), encoding="utf-8")
    managed = tmp_path / "managed"
    def surface(root):
        (root / "surface").mkdir(parents=True, exist_ok=True)
        points = np.asarray([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]], dtype=np.float64)
        np.savez(root / "surface" / "deformation_surface_dataset.npz", points=points,
                 visibility_mask=np.ones((4, 2), bool), projected_uv=np.ones((4, 2, 2), np.float64),
                 visible_counts=np.full(4, 2, np.int64), cam_names=np.asarray(NAMES))
    def precalc(root):
        root.joinpath("precalculation").mkdir(parents=True, exist_ok=True)
        reference = np.asarray([[0., 0., 1.], [1., 0., 1.]], np.float64); current = reference + .1
        np.savez(root / "precalculation" / "sparse_tracks.npz", displacement=current-reference)
        (root / "precalculation" / "sparse_scale.json").write_text(json.dumps({"scale_stats": {"mean": .1}}), encoding="utf-8")
    def roi(root):
        for name in NAMES:
            path = root / "roi" / "per_camera" / f"{name}_mask.npy"; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, np.ones((3, 4), bool))
        (root / "roi" / "mask_meta.json").write_text(json.dumps({"camera_names": NAMES}), encoding="utf-8")
    deps = [_producer(managed, "surface_source", "a", "ndef.combined_surface_call", "neurodic.ndef.surface_combined/v1", "ndef_surface", surface),
            _producer(managed, "precalc_source", "a", "ndef.precalculation_call", "neurodic.ndef.precalculation/v1", "ndef_precalculation", precalc),
            _producer(managed, "roi_source", "a", "ndef.roi.generate_call", "neurodic.ndef.roi/v1", "ndef_roi", roi)]
    return config_path, paths_path, managed, deps


def _plan(fixture, trial_id="f_trial"):
    config, paths, _managed, deps = fixture
    return plan_trial(config, case_key="f_test", case_paths=paths, trial_id=trial_id,
                      scope={"ndef_deformation_only": True}, restore_missing=True,
                      upstream_dependencies=deps).data["trial_plan"]


def _fake(monkeypatch, *, mode=None, cache_probe=None):
    calls = []
    backend = types.SimpleNamespace(**{
        name: object() for name in (
            "CameraModel", "NDeFProblem", "NDeFModelOptions",
            "estimate_ndef_displacement_scale", "PhotometricLossType", "NDeFSolver",
        )
    })
    monkeypatch.setattr(
        "neurodic.agent.adapters.execution_ndef_deformation._import_native_backend",
        lambda: backend,
    )
    module = types.ModuleType("neurodic.api.ndef_dic")
    def fake(config, write_case_artifacts=True):
        calls.append(copy.deepcopy(config)); root = Path(config["output"]["result"]); n = 4; points = np.asarray([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]], np.float64); current = points + .1; disp = current - points; sfm = points.copy()
        if mode == "runtime": raise RuntimeError("fake deformation failure")
        if mode == "interrupt": raise KeyboardInterrupt()
        if mode == "cache":
            cache_root = Path(os.environ["MPLCONFIGDIR"])
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_file = cache_root / "fontlist-v3.11.0.json"
            cache_file.write_text("fake font cache", encoding="utf-8")
            if cache_probe is not None:
                cache_probe.append((cache_root, cache_file.exists(), root))
        if mode == "unexpected_scientific":
            path = root / "scientific" / "unexpected.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected", encoding="utf-8")
        if mode == "hidden_scientific":
            path = root / "scientific" / ".unexpected" / "file.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected", encoding="utf-8")
        current_sfm = current.copy()
        displacement_sfm = disp.copy()
        magnitude = np.linalg.norm(disp, axis=1)
        magnitude_sfm = np.linalg.norm(displacement_sfm, axis=1)
        if mode == "material_corruption":
            disp[0, 0] += 1e-3
        if mode == "world_corruption":
            current_sfm[0, 0] += 1e-2
            displacement_sfm = current_sfm - sfm
            magnitude_sfm = np.linalg.norm(displacement_sfm, axis=1)
        if mode == "magnitude_corruption":
            magnitude[0] += 1e-3
        (root / "reconstruct").mkdir(parents=True); (root / "deformation").mkdir(); (root / "diagnostics").mkdir()
        for name in ("reference_surface.npz",): np.savez(root / "reconstruct" / name, points=points, points_sfm=sfm, sfm_to_world_scale=np.asarray(1.), cam_names=np.asarray(NAMES))
        np.savez(root / "reconstruct/current_surface.npz", points=current, points_sfm=current_sfm, sfm_to_world_scale=np.asarray(1.), cam_names=np.asarray(NAMES))
        np.savez(root / "deformation/reference_to_current.npz", reference_points=points, current_points=current, displacement=disp, displacement_magnitude=magnitude, strain=np.zeros((n, 6)), strain_components=np.asarray(["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_xz"]), reference_points_sfm=sfm, current_points_sfm=current_sfm, displacement_sfm=displacement_sfm, displacement_magnitude_sfm=magnitude_sfm, sfm_to_world_scale=np.asarray(1.), cam_names=np.asarray(NAMES))
        np.savez(root / "diagnostics/projection.npz", reference_uv=np.ones((n, 2, 2)), current_uv=np.ones((n, 2, 2)), reference_depth=np.ones((n, 2)), current_depth=np.ones((n, 2)), valid=np.ones((n, 2), bool))
        history = np.asarray([[1., 1., 1., 1., 0., 1., 1., .1]])
        np.savez(root / "diagnostics/training.npz", history=history, history_columns=np.asarray(["epoch", "step", "loss", "photometric_loss", "smoothness", "valid_pairs", "supervised_pairs", "displacement_rms"]), batch_size=np.asarray(4), steps_per_epoch=np.asarray(1), completed_epochs=np.asarray(1), random_seed=np.asarray(23), output_scale=np.asarray(float(np.linalg.norm([.1, .1, .1]))))
        (root / "diagnostics/training_history.json").write_text(json.dumps([{k: float(v) for k, v in zip(["epoch", "step", "loss", "photometric_loss", "smoothness", "valid_pairs", "supervised_pairs", "displacement_rms"], history[0])}]), encoding="utf-8")
        (root / "diagnostics/summary.json").write_text(json.dumps({"coordinate_frame": "calibration world frame"}), encoding="utf-8")
        if config.get("evaluation", {}).get("enabled"):
            np.savez(root / "diagnostics/evaluation.npz", indices=np.asarray([0]), residual=np.asarray([0.1]))
            (root / "diagnostics/evaluation.json").write_text(json.dumps({"schema_version": "neurodic.fixed_evaluation/v1"}), encoding="utf-8")
        for name in ("deformation_field.pt", "deformation_field_best.pt"):
            with zipfile.ZipFile(root / "deformation" / name, "w") as archive:
                archive.writestr("archive/data.pkl", b"safe"); archive.writestr("archive/version", b"3"); archive.writestr("archive/data/0", b"0")
        if mode == "visualization":
            path = root / "visualization" / "explicit-product.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("explicit visualization product", encoding="utf-8")
        if mode == "partial": raise RuntimeError("partial deformation export")
    module.ndef_dic = fake; monkeypatch.setitem(sys.modules, "neurodic.api.ndef_dic", module)
    return calls


def test_f2b2_backend_unavailable_fails_before_scientific_callable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path)
    calls = _fake(monkeypatch)

    def unavailable():
        raise ModuleNotFoundError("controlled missing neurodic._neurodic")

    monkeypatch.setattr(
        "neurodic.agent.adapters.execution_ndef_deformation._import_native_backend",
        unavailable,
    )
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "warning"
    assert result.warnings == ({"code": "CAPABILITY.UNSUPPORTED"},)
    assert result.data["execution"]["stage_attempts"][0]["error"] == (
        "NDeF deformation native backend preflight failed"
    )
    assert calls == []


def test_f2b2_backend_available_preflight_does_not_invoke_scientific_callable(
        monkeypatch: pytest.MonkeyPatch):
    calls = _fake(monkeypatch)
    capability = ndef_deformation_backend_capability()
    assert capability["available"] is True
    assert capability["missing_symbols"] == []
    assert all(record["present"] for record in capability["symbols"].values())
    assert calls == []


def test_f1_fake_combined_lifecycle_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); plan = _plan(fixture); assert plan["plan_status"] == "ready"; assert [a["action_id"] for a in plan["execution_actions"]] == ["ndef.deformation_combined_call"]
    calls = _fake(monkeypatch); first = execute_trial(plan, managed_root=fixture[2]); assert first.status == "ok"; assert calls
    second = execute_trial(_plan(fixture, "f_trial_2"), managed_root=fixture[2]); assert second.status == "ok"; assert len(calls) == 1


def test_f1_capability_and_no_split_actions():
    capability = capability_for("ndef.deformation_combined_call"); assert capability.execution_supported and capability.completion_scope == "combined_action"
    assert not capability_for("ndef.deformation_train_call").execution_supported


def test_f1_fake_evaluation_outputs_are_conditional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); config = json.loads(fixture[0].read_text()); config["evaluation"] = {"enabled": True, "sample_count": 1, "seed": 17}; fixture[0].write_text(json.dumps(config)); fixture[1].write_text(json.dumps({"f_test": config}))
    calls = _fake(monkeypatch); plan = _plan(fixture); assert plan["plan_status"] == "ready"; result = execute_trial(plan, managed_root=fixture[2]); assert result.status == "ok"; assert calls


def test_f1_config_rejects_auto_batch_before_plan(tmp_path: Path):
    fixture = _fixture(tmp_path); config = json.loads(fixture[0].read_text()); config["deformation_training"]["batch_size"] = 0; fixture[0].write_text(json.dumps(config)); fixture[1].write_text(json.dumps({"f_test": config}))
    plan = _plan(fixture); assert plan["plan_status"] == "blocked"; assert any("AUTO_BATCH" in item["code"] for item in plan["policy_violations"])


@pytest.mark.parametrize("mode", ["runtime", "interrupt", "partial"])
def test_f1_failures_do_not_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    fixture = _fixture(tmp_path); calls = _fake(monkeypatch, mode=mode); result = execute_trial(_plan(fixture), managed_root=fixture[2]); assert result.status == "warning" and calls
    assert not list((fixture[2] / "trials" / "f_trial" / "artifacts").rglob("reference_to_current.npz"))


def test_f1_dependency_toc_tou_fails_before_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); plan = _plan(fixture); calls = _fake(monkeypatch)
    surface = next((fixture[2] / "trials" / "surface_source").rglob("deformation_surface_dataset.npz")); surface.write_bytes(surface.read_bytes() + b"tamper")
    with pytest.raises(Exception): execute_trial(plan, managed_root=fixture[2])
    assert not calls


def test_f2_matplotlib_cache_is_private_and_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); cache_probe = []; calls = _fake(monkeypatch, mode="cache", cache_probe=cache_probe)
    previous = os.environ.get("MPLCONFIGDIR")
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "ok" and calls and len(cache_probe) == 1
    cache_root, existed_during_call, staging = cache_probe[0]
    assert existed_during_call and cache_root != staging and not cache_root.is_relative_to(staging)
    assert not cache_root.exists()
    assert calls[0]["MPLCONFIGDIR"] == str(cache_root)
    assert os.environ.get("MPLCONFIGDIR") == previous
    published = result.data["execution"]["produced_artifacts"]
    assert all("fontlist-v3.11.0.json" not in str(item) for item in published)


@pytest.mark.parametrize("mode", ["unexpected_scientific", "hidden_scientific"])
def test_f2_uncontrolled_scientific_files_remain_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    fixture = _fixture(tmp_path); _fake(monkeypatch, mode=mode)
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "warning"
    assert result.data["execution"]["stage_attempts"][0]["error"] == "NDEF.DEFORMATION_UNCONTROLLED_OUTPUT"


def test_f2_explicit_visualization_namespace_policy_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); _fake(monkeypatch, mode="visualization")
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "ok"
    locations = {item["location"] for item in result.data["execution"]["produced_artifacts"]}
    assert any(path.endswith("visualization/explicit-product.json") for path in locations)


def test_f2_private_cache_is_not_a_reuse_determinant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); calls = _fake(monkeypatch, mode="cache")
    first = execute_trial(_plan(fixture, "f_cache_1"), managed_root=fixture[2])
    second = execute_trial(_plan(fixture, "f_cache_2"), managed_root=fixture[2])
    assert first.status == "ok" and second.status == "ok"
    assert len(calls) == 1
    assert second.data["execution"]["reused_artifacts"]


def test_f2b1_float32_cancellation_floor_is_accepted():
    reference = np.asarray([[0.0, 0.0, 10.0]], dtype=np.float32)
    current = np.asarray([[0.0, 0.0, 10.003091]], dtype=np.float32)
    displacement = np.asarray([[0.0, 0.0, 0.0030904312]], dtype=np.float32)
    delta = current - reference
    assert not np.allclose(delta, displacement, rtol=1e-5, atol=1e-7)
    assert _deformation_delta_allclose(delta, displacement, operands=np.abs(current) + np.abs(reference))


def test_f2b1_material_corruption_remains_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); _fake(monkeypatch, mode="material_corruption")
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "warning"
    assert result.data["execution"]["stage_attempts"][0]["error"] == "NDEF.DEFORMATION_INCONSISTENT"


def test_f2b1_world_sfm_corruption_remains_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); _fake(monkeypatch, mode="world_corruption")
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "warning"
    assert result.data["execution"]["stage_attempts"][0]["error"] == "NDEF.WORLD_SFM_INCONSISTENT"


def test_f2b1_magnitude_corruption_remains_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path); _fake(monkeypatch, mode="magnitude_corruption")
    result = execute_trial(_plan(fixture), managed_root=fixture[2])
    assert result.status == "warning"
    assert result.data["execution"]["stage_attempts"][0]["error"] == "NDEF.DEFORMATION_INCONSISTENT"


def test_f2b1_native_checkpoint_metadata_members_are_structural(tmp_path: Path):
    checkpoint = tmp_path / "deformation_field.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        for name, payload in {
            "model/data.pkl": b"safe", "model/version": b"3", "model/byteorder": b"little",
            "model/.format_version": b"1", "model/.storage_alignment": b"64", "model/data/0": b"0",
            "model/.data/serialization_id": b"0",
        }.items():
            archive.writestr(name, payload)
    assert _checkpoint(checkpoint, "final")
    archive_path = tmp_path / "unexpected.pt"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in {
            "model/data.pkl": b"safe", "model/version": b"3", "model/data/0": b"0", "model/evil": b"x",
        }.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="NDEF.CHECKPOINT_INVALID"):
        _checkpoint(archive_path, "final")
