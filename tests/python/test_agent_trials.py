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
    report = plan_trial(ROOT / "config/ndef_multi.yaml", case_key="ndef_multi", case_paths=ROOT / "config/case_paths.yaml",
                        override={}, restore_missing=True).to_dict()["data"]["trial_plan"]
    assert report["changes"] == []
    assert report["plan_status"] == "partial"
    assert report["execution_performed"] is False


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
