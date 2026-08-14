"""Native-free control tests for the managed NDeF ROI action."""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fixtures.prepare_ndef_d2a_fixture import prepare
from neurodic.agent.adapters.execution_ndef_roi import ACTION_ID, INPUTS_KEY, guarded_ndef_roi_action
from neurodic.agent.adapters.execution_ndef import managed_surface_inputs
from neurodic.agent.execution import _stage_signature, execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "fixture"
    config = prepare(root)
    return config, root / "case_paths.yaml"


def _plan(config: Path, paths: Path, trial_id: str) -> dict:
    return plan_trial(config, case_key="ndef_d2a", case_paths=paths, trial_id=trial_id,
                      scope={"ndef_roi_only": True}, restore_missing=True).to_dict()["data"]["trial_plan"]


def _fake_roi(monkeypatch: pytest.MonkeyPatch, *, corrupt: bool = False):
    calls: list[Path] = []
    module = types.ModuleType("neurodic.ndef_roi")

    def fake(case_root, options=None, *, result_root=None, visualization_root=None):
        calls.append(Path(case_root))
        calibration = json.loads((Path(case_root) / "result/calibration/calibration_result_scaled.json").read_text())
        names = [str(item["label"]) for item in calibration["cameras"]]
        roi = Path(result_root); (roi / "per_camera").mkdir(parents=True, exist_ok=True)
        masks = np.stack([np.ones((4, 4), dtype=bool) for _ in names])
        if corrupt:
            masks = masks.astype(np.uint8)
        for index, name in enumerate(names):
            np.save(roi / "per_camera" / f"{name}_mask.npy", masks[index])
        np.savez_compressed(roi / "masks.npz", cam_names=np.asarray(names), masks=masks)
        (roi / "mask_meta.json").write_text(json.dumps({
            "schema_version": 1,
            "cameras": [{"camera_index": index, "camera_name": name, "mask_pixels": 16,
                          "image_pixels": 16, "shared_observation_union": 25,
                          "points_after_outlier_filter": 25, "mask_fraction": 1.0}
                         for index, name in enumerate(names)],
        }))
        return {}

    module.generate_ndef_roi = fake
    monkeypatch.setitem(sys.modules, "neurodic.ndef_roi", module)
    return calls


def test_roi_capability_and_ordered_frozen_inputs(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path); plan = _plan(config, paths, "roi_order")
    assert capability_for(ACTION_ID).completion_scope == "combined_action"
    assert [item["action_id"] for item in plan["execution_actions"]] == [ACTION_ID]
    assert plan["plan_status"] == "ready"
    assert plan["scope"][INPUTS_KEY]["camera_ids"] == ["cam_0", "cam_1"]


def test_roi_atomic_publish_and_baseline_zero_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, paths = _fixture(tmp_path); calls = _fake_roi(monkeypatch)
    root = Path(json.loads(config.read_text())["case"]["root"])
    before = (root / "result/calibration/calibration_result_scaled.json").read_bytes()
    result = execute_trial(_plan(config, paths, "roi_atomic"), managed_root=tmp_path / "managed", action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert result["execution_status"] == "completed" and calls == [root]
    assert (root / "result/calibration/calibration_result_scaled.json").read_bytes() == before
    assert result["stage_attempts"][0]["status"] == "completed"
    assert not list((tmp_path / "managed/trials/roi_atomic/staging").rglob("*.npy"))
    assert len(result["produced_artifacts"]) == 4


def test_roi_signature_binds_content_and_excludes_runtime(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path); plan = _plan(config, paths, "roi_signature")
    from neurodic.agent.inspect import resolve_config
    effective = resolve_config(config, case_key="ndef_d2a", case_paths=paths)["effective_config"]
    action = guarded_ndef_roi_action()
    first = _stage_signature(plan, effective, action, ("ndef.roi",))
    altered = copy.deepcopy(effective); altered["runtime"]["random_seed"] = 7
    assert first.digest == _stage_signature(plan, altered, action, ("ndef.roi",)).digest
    changed = copy.deepcopy(plan); changed["scope"][INPUTS_KEY]["camera_ids"] = ["cam_1", "cam_0"]
    assert first.digest != _stage_signature(changed, effective, action, ("ndef.roi",)).digest


@pytest.mark.parametrize("relative", [
    "result/calibration/calibration_result_scaled.json",
    "result/calibration/observations.npz",
    "result/calibration/camera_pairs.json",
    "images/cam_0/000.pgm",
])
def test_roi_input_tamper_stales_plan_before_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str) -> None:
    config, paths = _fixture(tmp_path); plan = _plan(config, paths, "roi_tamper")
    calls = _fake_roi(monkeypatch)
    target = Path(json.loads(config.read_text())["case"]["root"] ) / relative
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(Exception):
        execute_trial(plan, managed_root=tmp_path / "managed", action_id=ACTION_ID)
    assert not calls


def test_roi_validator_failure_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, paths = _fixture(tmp_path); _fake_roi(monkeypatch, corrupt=True)
    result = execute_trial(_plan(config, paths, "roi_invalid"), managed_root=tmp_path / "managed", action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert result["execution_status"] == "failed"
    assert not list((tmp_path / "managed/trials/roi_invalid/artifacts").rglob("*"))


def test_roi_safe_reuse_skips_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, paths = _fixture(tmp_path); calls = _fake_roi(monkeypatch); managed = tmp_path / "managed"
    first = execute_trial(_plan(config, paths, "roi_source"), managed_root=managed, action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert first["execution_status"] == "completed" and len(calls) == 1
    monkeypatch.setattr(sys.modules["neurodic.ndef_roi"], "generate_ndef_roi",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ROI callable invoked")))
    second = execute_trial(_plan(config, paths, "roi_reuse"), managed_root=managed, action_id=ACTION_ID).to_dict()["data"]["execution"]
    assert second["stage_attempts"][0]["status"] == "reused" and len(calls) == 1


def test_surface_dependency_rejects_legacy_roi_producer(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    from neurodic.agent.inspect import resolve_config
    effective = resolve_config(config, case_key="ndef_d2a", case_paths=paths)["effective_config"]
    legacy = {"dependency_id": "ndef_roi", "producer_action_id": "legacy.roi",
              "producer_signature": {"stage_id": "legacy.roi", "implementation": {"adapter": "legacy"}},
              "scope": {}, "required_artifacts": []}
    with pytest.raises(ValueError):
        managed_surface_inputs({"upstream_dependencies": [legacy]}, effective)
