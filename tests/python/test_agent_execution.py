"""Loop 7 control-plane lifecycle tests using only trusted fake adapters."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from neurodic.agent.execution import ProducerSignature, TrustedAction, execute_trial
from neurodic.agent.errors import ControlPlaneError
from neurodic.agent.trials import plan_trial


ROOT = Path(__file__).resolve().parents[2]


def _plan(trial_id: str) -> dict:
    return plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                      trial_id=trial_id, override={"training": {"seed_iterations": 4999}}).to_dict()["data"]["trial_plan"]


def _adapter(run) -> dict[str, TrustedAction]:
    return {"pin.combined_solver_call": TrustedAction("pin.combined_solver_call", run, "neurodic.test.fake/v1")}


def test_producer_signature_is_deterministic_and_implementation_specific() -> None:
    base = dict(stage_id="pin.solver", stage_config_identity="sha256:config", input_identities={"input": "sha256:in"},
                scope={"frame": 1}, output_contract="neurodic.managed-artifact/v1")
    first = ProducerSignature(implementation={"adapter": "neurodic.test.adapter/v1", "neurodic": {"git_revision": "abc"}}, **base)
    same = ProducerSignature(implementation={"adapter": "neurodic.test.adapter/v1", "neurodic": {"git_revision": "abc"}}, **base)
    changed = ProducerSignature(implementation={"adapter": "neurodic.test.adapter/v2", "neurodic": {"git_revision": "abc"}}, **base)
    assert first.digest == same.digest
    assert first.digest != changed.digest


def test_safe_reuse_verifies_matching_producer_without_running_adapter(tmp_path: Path) -> None:
    calls = 0
    def run(_config, staging: Path, _scope):
        nonlocal calls; calls += 1
        (staging / "result.json").write_text('{"ok":true}', encoding="utf-8")
        return ["result.json"]
    execute_trial(_plan("reuse_source"), managed_root=tmp_path, trusted_actions=_adapter(run))
    report = execute_trial(_plan("reuse_target"), managed_root=tmp_path, trusted_actions=_adapter(run)).to_dict()
    assert calls == 1
    execution = report["data"]["execution"]
    assert execution["stage_attempts"][0]["status"] == "reused"
    assert execution["reused_artifacts"][0]["reuse_source_trial"] == "reuse_source"


def test_fake_execution_stages_then_atomically_publishes_with_provenance(tmp_path: Path) -> None:
    plan = _plan("fake_publish")
    baseline = ROOT / "config/pin_2d.yaml"; before = hashlib.sha256(baseline.read_bytes()).hexdigest()
    seen: dict[str, Path] = {}
    def run(_config, staging: Path, _scope):
        seen["staging"] = staging
        output = staging / "result.json"; output.write_text('{"ok":true}', encoding="utf-8")
        assert not (staging.parents[2] / "artifacts").exists() or not list((staging.parents[2] / "artifacts").rglob("result.json"))
        return ["result.json"]
    report = execute_trial(plan, managed_root=tmp_path, trusted_actions=_adapter(run)).to_dict()
    execution = report["data"]["execution"]; trial = tmp_path / "trials/fake_publish"
    assert execution["execution_status"] == "completed"
    artifact = execution["produced_artifacts"][0]
    assert artifact["identity"]["strength"] == "content"
    assert artifact["producer_signature"]["digest"].startswith("sha256:")
    assert (trial / artifact["location"]).is_file() and not seen["staging"].exists()
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == before
    assert (trial / "manifest.json").is_file() and (trial / "effective_config.json").is_file()


def test_partial_failure_is_not_published(tmp_path: Path) -> None:
    def run(_config, staging: Path, _scope):
        (staging / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("fake failure")
    report = execute_trial(_plan("fake_failed"), managed_root=tmp_path, trusted_actions=_adapter(run)).to_dict()
    trial = tmp_path / "trials/fake_failed"
    assert report["data"]["execution"]["execution_status"] == "failed"
    assert not list((trial / "artifacts").rglob("*"))
    assert (trial / "staging").exists()


def test_keyboard_interrupt_is_interrupted_not_failed(tmp_path: Path) -> None:
    def run(_config, staging: Path, _scope):
        (staging / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt()
    report = execute_trial(_plan("fake_interrupted"), managed_root=tmp_path, trusted_actions=_adapter(run)).to_dict()
    assert report["data"]["execution"]["execution_status"] == "interrupted"
    assert report["data"]["execution"]["stage_attempts"][0]["status"] == "interrupted"


def test_tampered_plan_and_untrusted_actions_are_blocked(tmp_path: Path) -> None:
    stale = copy.deepcopy(_plan("fake_stale")); stale["plan_identity"] = "sha256:tampered"
    with pytest.raises(ControlPlaneError) as error:
        execute_trial(stale, managed_root=tmp_path, trusted_actions={})
    assert error.value.record.code == "TRIAL.PLAN_STALE"
    with pytest.raises(ControlPlaneError) as error:
        execute_trial(_plan("fake_unsupported"), managed_root=tmp_path)
    assert error.value.record.code == "EXECUTION.UNSUPPORTED"
    assert not (tmp_path / "trials/fake_unsupported").exists()


def test_output_escape_and_existing_trial_are_blocked(tmp_path: Path) -> None:
    def run(_config, staging: Path, _scope):
        (staging / "ok.bin").write_bytes(b"ok")
        return ["../../escape"]
    report = execute_trial(_plan("fake_escape"), managed_root=tmp_path, trusted_actions=_adapter(run)).to_dict()
    assert report["data"]["execution"]["execution_status"] == "failed"
    def good(_config, staging: Path, _scope):
        (staging / "ok.bin").write_bytes(b"ok"); return ["ok.bin"]
    execute_trial(_plan("fake_exists"), managed_root=tmp_path, trusted_actions=_adapter(good))
    with pytest.raises(ControlPlaneError) as error:
        execute_trial(_plan("fake_exists"), managed_root=tmp_path, trusted_actions=_adapter(good))
    assert error.value.record.code == "TRIAL.ROOT_EXISTS"


def test_cli_execute_never_bypasses_plan_or_registers_real_adapter(tmp_path: Path) -> None:
    import json, os, subprocess, sys
    plan_path = tmp_path / "plan.json"; plan_path.write_text(json.dumps(_plan("fake_cli")), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "neurodic.cli", "trial", "execute", "--plan", str(plan_path),
                             "--managed-root", str(tmp_path)], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "python")}, text=True, capture_output=True)
    assert result.returncode != 0
    assert json.loads(result.stdout)["errors"][0]["code"] == "EXECUTION.UNSUPPORTED"
    assert not (tmp_path / "trials/fake_cli").exists()


def test_shared_input_content_change_makes_plan_stale(tmp_path: Path) -> None:
    case = tmp_path / "case"; case.mkdir()
    for name in ("0.bmp", "1.bmp", "2.bmp"): (case / name).write_bytes(b"input-" + name.encode())
    config = tmp_path / "pin.yaml"; config.write_text("solver: pin\nmode: planar_2d\ntraining:\n  seed_iterations: 5\n", encoding="utf-8")
    paths = tmp_path / "paths.yaml"; paths.write_text(
        f"pin_2d:\n  case:\n    root: {case}\n    images_dir: .\n  output:\n    result: result/pin\n", encoding="utf-8")
    plan = plan_trial(config, case_key="pin_2d", case_paths=paths, trial_id="fake_input_stale",
                      override={"training": {"seed_iterations": 4}}).to_dict()["data"]["trial_plan"]
    (case / "0.bmp").write_bytes(b"changed")
    with pytest.raises(ControlPlaneError) as error:
        execute_trial(plan, managed_root=tmp_path, trusted_actions=_adapter(lambda _config, _stage, _scope: []))
    assert error.value.record.code == "TRIAL.PLAN_STALE"
