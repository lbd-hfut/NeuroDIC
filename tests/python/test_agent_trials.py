"""Loop 6 dry-run planning tests; no test invokes a scientific workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from neurodic.agent.trials import plan_trial


ROOT = Path(__file__).resolve().parents[2]


def _snapshot(root: Path) -> list[tuple[str, int, int]]:
    return sorted((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in root.rglob("*") if path.is_file())


def _plan(config: str, case_key: str, override: dict) -> dict:
    return plan_trial(ROOT / config, case_key=case_key, case_paths=ROOT / "config/case_paths.yaml", override=override).to_dict()["data"]["trial_plan"]


def test_ndef_deformation_change_has_only_deformation_config_closure() -> None:
    plan = _plan("config/ndef_multi.yaml", "ndef_multi", {"deformation_training": {"photometric_learning_rate": 0.0005}})
    assert plan["plan_status"] in {"ready", "partial"}
    assert plan["config_invalidated_stages"] == ["ndef.deformation.train", "ndef.deformation.infer", "ndef.postprocess", "ndef.evaluate"]
    assert "ndef.surface" not in plan["config_invalidated_stages"]
    assert plan["dry_run"] and not plan["execution_performed"] and plan["baseline_writes"] == []


def test_evaluation_only_change_does_not_invalidate_pin_training() -> None:
    plan = _plan("config/pin_2d.yaml", "pin_2d", {"evaluation": {"seed": 19}})
    assert plan["config_invalidated_stages"] == ["pin.evaluate"]
    assert all(not stage.startswith("pin.train") for stage in plan["config_invalidated_stages"])


def test_pin_multi_fusion_change_preserves_pair_config_closure() -> None:
    plan = _plan("config/pin_multi.yaml", "pin_multi", {"fusion": {"voxel_size": 0.2}})
    assert plan["config_invalidated_stages"] == ["pin_multi.fusion", "pin_multi.postprocess", "pin_multi.evaluate"]
    assert "pin_multi.pair_solve" not in plan["config_invalidated_stages"]


@pytest.mark.parametrize(("config", "case_key", "override"), [
    ("config/pin_2d.yaml", "pin_2d", {"case": {"root": "elsewhere"}}),
    ("config/pin_2d.yaml", "pin_2d", {"output": {"result": "elsewhere"}}),
    ("config/pin_2d.yaml", "pin_2d", {"solver": "ndef"}),
    ("config/pin_multi.yaml", "pin_multi", {"camera_pairs": {"wrap": False}}),
    ("config/ndef_multi.yaml", "ndef_multi", {"scale": {"sfm_to_world_scale": 2.0}}),
])
def test_protected_paths_block_without_plan(config: str, case_key: str, override: dict) -> None:
    plan = _plan(config, case_key, override)
    assert plan["plan_status"] == "blocked"
    assert plan["policy_violations"][0]["code"] == "TRIAL.PROTECTED_PATH"
    assert plan["execution_actions"] == []


def test_unknown_wrong_type_and_noop_are_strict() -> None:
    with pytest.raises(Exception): _plan("config/pin_2d.yaml", "pin_2d", {"not_a_field": 1})
    with pytest.raises(Exception): _plan("config/pin_2d.yaml", "pin_2d", {"training": {"seed_iterations": "bad"}})
    plan = _plan("config/pin_2d.yaml", "pin_2d", {"training": {"seed_iterations": 5000}})
    assert plan["plan_status"] == "no_effect"
    assert plan["changes"] == []
    assert plan["minimum_rerun_stages"] == []


def test_planning_leaves_config_and_case_tree_unchanged() -> None:
    config = ROOT / "config/ndef_multi.yaml"; case = ROOT / "case/Multi/CylinderDIC"
    before_hash = hashlib.sha256(config.read_bytes()).hexdigest(); before_tree = _snapshot(case)
    _plan("config/ndef_multi.yaml", "ndef_multi", {"deformation_training": {"photometric_learning_rate": 0.0005}})
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before_hash
    assert _snapshot(case) == before_tree


def test_explicit_restore_missing_can_plan_without_config_changes() -> None:
    report = plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                        override={}, restore_missing=True, scope={"selected_frame": 0}).to_dict()["data"]["trial_plan"]
    assert report["changes"] == []
    assert report["plan_status"] == "ready"
    assert report["execution_performed"] is False
    assert report["planning_intent"] == {"restore_missing": True}


def test_planning_intent_is_serialized_and_changes_plan_identity() -> None:
    from neurodic.agent.execution import _revalidate
    restored = plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                          override={}, restore_missing=True, scope={"selected_frame": 0}, trial_id="intent-restored").to_dict()["data"]["trial_plan"]
    normal = plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                        override={}, restore_missing=False, scope={"selected_frame": 0}, trial_id="intent-restored").to_dict()["data"]["trial_plan"]
    loaded = json.loads(json.dumps(restored))
    assert loaded["planning_intent"] == {"restore_missing": True}
    assert restored["plan_identity"] != normal["plan_identity"]
    assert _revalidate(loaded)["plan_identity"] == restored["plan_identity"]


def test_tampered_restore_intent_fails_revalidation() -> None:
    from neurodic.agent.errors import ControlPlaneError
    from neurodic.agent.execution import _revalidate
    plan = plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                      override={}, restore_missing=True, scope={"selected_frame": 0}, trial_id="intent-tampered").to_dict()["data"]["trial_plan"]
    plan["planning_intent"] = {"restore_missing": False}
    with pytest.raises(ControlPlaneError) as error:
        _revalidate(plan)
    assert error.value.record.code == "TRIAL.PLAN_STALE"


def test_normal_pin_multi_plan_persists_non_restore_intent() -> None:
    plan = plan_trial(ROOT / "config/pin_multi.yaml", case_key="pin_multi", case_paths=ROOT / "config/case_paths.yaml",
                      override={"pair_roi": {"max_features": 11999}}, scope={"pair_id": "cam_0__cam_1"}).to_dict()["data"]["trial_plan"]
    assert plan["planning_intent"] == {"restore_missing": False}


def _pin_scope_fixture(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "case"; case.mkdir()
    for index in range(4):
        (case / f"{index:03d}.bmp").write_bytes(f"frame-{index}".encode())
    config = tmp_path / "pin.yaml"; config.write_text((ROOT / "config/pin_2d.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    paths = tmp_path / "paths.yaml"; paths.write_text(
        f"pin_2d:\n  case:\n    root: {case}\n    images_dir: .\n  output:\n    result: result/pin\n    visualization: visualization/pin\n", encoding="utf-8")
    return config, paths


def test_zero_override_without_restore_is_no_effect_when_nothing_requires_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, paths = _pin_scope_fixture(tmp_path)
    monkeypatch.setattr("neurodic.agent.trials._artifact_assessment", lambda *_args: ((), ()))
    report = plan_trial(config, case_key="pin_2d", case_paths=paths, override={}, scope={"selected_frame": 0}).to_dict()["data"]["trial_plan"]
    assert report["plan_status"] == "no_effect" and report["execution_actions"] == []


def test_zero_override_restore_with_nothing_to_rerun_stays_no_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, paths = _pin_scope_fixture(tmp_path)
    monkeypatch.setattr("neurodic.agent.trials._artifact_assessment", lambda *_args: ((), ()))
    report = plan_trial(config, case_key="pin_2d", case_paths=paths, override={}, restore_missing=True,
                        scope={"selected_frame": 0}).to_dict()["data"]["trial_plan"]
    assert report["plan_status"] == "no_effect" and report["execution_actions"] == []


def test_pin_scope_resolves_exact_frame_and_affects_plan_identity(tmp_path: Path) -> None:
    config, paths = _pin_scope_fixture(tmp_path)
    first = plan_trial(config, case_key="pin_2d", case_paths=paths, override={"training": {"seed_iterations": 4999}},
                       scope={"selected_frame": 0}).to_dict()["data"]["trial_plan"]
    second = plan_trial(config, case_key="pin_2d", case_paths=paths, override={"training": {"seed_iterations": 4999}},
                        scope={"selected_frame": 1}).to_dict()["data"]["trial_plan"]
    assert first["plan_status"] == second["plan_status"] == "ready"
    assert first["scope"] == {"selected_frame": 0} and second["scope"] == {"selected_frame": 1}
    assert first["plan_identity"] != second["plan_identity"]


def test_pin_scope_selects_different_deformed_input_and_producer_signature(tmp_path: Path) -> None:
    from neurodic.agent.adapters.execution_pin import guarded_pin_action
    from neurodic.agent.config import merge_sparse_override
    from neurodic.agent.execution import _stage_signature
    from neurodic.agent.inspect import resolve_config
    config, paths = _pin_scope_fixture(tmp_path)
    plans = [plan_trial(config, case_key="pin_2d", case_paths=paths, override={"training": {"seed_iterations": 4999}},
                        scope={"selected_frame": frame}).to_dict()["data"]["trial_plan"] for frame in (0, 1)]
    action = guarded_pin_action()
    signatures = []
    inputs = []
    for plan in plans:
        base = resolve_config(config, case_key="pin_2d", case_paths=paths)["effective_config"]
        values, _ = merge_sparse_override(base, plan["override"], solver="pin")
        inputs.append(action.input_identities(plan, values))
        signatures.append(_stage_signature(plan, values, action, plan["execution_actions"][0]["covers_stages"]))
    assert inputs[0]["deformed_image"]["digest"] != inputs[1]["deformed_image"]["digest"]
    assert signatures[0].digest != signatures[1].digest


@pytest.mark.parametrize("scope", [{"selected_frame": -1}, {"selected_frame": 2}, {"selected_frame": "0"}])
def test_invalid_pin_scope_is_structured_error(tmp_path: Path, scope: dict) -> None:
    config, paths = _pin_scope_fixture(tmp_path)
    with pytest.raises(Exception) as error:
        plan_trial(config, case_key="pin_2d", case_paths=paths, override={"training": {"seed_iterations": 4999}}, scope=scope)
    assert getattr(error.value, "record").code == "SCHEMA.INVALID"


def test_cli_scope_json_is_frozen_into_pin_plan(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    result = subprocess.run([sys.executable, "-m", "neurodic.cli", "trial", "plan", "--config", "config/pin_2d.yaml", "--case-key", "pin_2d",
                             "--restore-missing", "--scope-json", '{"selected_frame":0}', "--trial-id", "pin-cli-scope"],
                            cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)["data"]["trial_plan"]
    assert result.stderr == "" and plan["scope"] == {"selected_frame": 0} and plan["plan_status"] == "ready"


def test_cli_emits_one_json_dry_run_document(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"; override.write_text("deformation_training:\n  photometric_learning_rate: 0.0005\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    result = subprocess.run([sys.executable, "-m", "neurodic.cli", "trial", "plan", "--config", "config/ndef_multi.yaml", "--case-key", "ndef_multi", "--override", str(override)], cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert result.stderr == "" and payload["operation"] == "trial.plan"
    assert payload["data"]["trial_plan"]["execution_performed"] is False


def test_planner_import_and_execution_remain_native_free() -> None:
    code = (
        "import sys; from neurodic.agent.trials import plan_trial; "
        "plan_trial('config/pin_2d.yaml', case_key='pin_2d', override={'evaluation': {'seed': 4}}); "
        "print('neurodic._neurodic' in sys.modules)"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
