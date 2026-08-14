"""Native-free control tests for the guarded coarse PIN adapter."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.adapters.execution_pin import (_execution_overlay, _input_identities,
                                                    _run_pin, guarded_pin_action)
from neurodic.agent.execution import _stage_signature, execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial


ROOT = Path(__file__).resolve().parents[2]


def _plan(trial_id: str, *, seed_iterations: int = 4999, scope: dict | None = None) -> dict:
    return plan_trial(ROOT / "config/pin_2d.yaml", case_key="pin_2d", case_paths=ROOT / "config/case_paths.yaml",
                      trial_id=trial_id, scope={"selected_frame": 0} if scope is None else scope,
                      override={"training": {"seed_iterations": seed_iterations}}).to_dict()["data"]["trial_plan"]


def _install_fake_pin(monkeypatch: pytest.MonkeyPatch, *, valid: bool = True, interrupt: bool = False) -> list[dict]:
    calls: list[dict] = []
    module = types.ModuleType("neurodic.api.pin_dic")
    def fake_pin(case_root, *, config, **_kwargs):
        calls.append({"case_root": case_root, "config": config})
        output = config["output"]
        scientific = Path(output["result"]); visual = Path(output["visualization"])
        scientific.mkdir(parents=True); visual.mkdir(parents=True)
        if interrupt:
            raise KeyboardInterrupt()
        np.savez(scientific / "pin_result.npz", coordinates=np.ones((2, 2)), displacement=np.ones((2, 2)),
                 strain=np.ones((2, 3)), strain_components=np.asarray(["E_xx", "E_yy", "E_xy"]), iterations=np.asarray(1), final_loss=np.asarray(.5))
        np.savez_compressed(scientific / "diagnostics_training.npz", schema_version=np.asarray("neurodic.pin.training/v1"),
                            history=np.ones((1, 3)) if valid else np.asarray([np.nan]), history_columns=np.asarray(["phase", "phase_step", "loss"]), phase_names=np.asarray(["seed_mse", "photometric"]))
        (visual / "pin_displacement.png").write_bytes(b"png")
        return object()
    module.pin_dic = fake_pin
    monkeypatch.setitem(sys.modules, "neurodic.api.pin_dic", module)
    return calls


def _tree_identity(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_pin_capability_is_exactly_the_combined_action() -> None:
    capability = capability_for("pin.combined_solver_call")
    assert capability.execution_supported and capability.completion_scope == "combined_action"
    assert not capability_for("pin.train").execution_supported


def test_pin_adapter_fake_lifecycle_is_staging_only_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_pin(monkeypatch)
    config_before = _tree_identity(ROOT / "config")
    case_before = _tree_identity(ROOT / "case/2D/ring")
    report = execute_trial(_plan("pin_fake_publish"), managed_root=tmp_path).to_dict()["data"]["execution"]
    assert report["execution_status"] == "completed" and len(calls) == 1
    assert calls[0]["config"]["output"]["result"].startswith(str(tmp_path / "trials/pin_fake_publish/staging"))
    assert not list((tmp_path / "trials/pin_fake_publish/staging").rglob("pin_result.npz"))
    assert any(item["artifact_type"] == "pin_result" for item in report["produced_artifacts"])
    assert _tree_identity(ROOT / "config") == config_before
    assert _tree_identity(ROOT / "case/2D/ring") == case_before


def test_output_overlay_does_not_change_scientific_signature(tmp_path: Path) -> None:
    plan = _plan("pin_signature")
    action = guarded_pin_action()
    # Reconstruct the actual effective mapping through the execution path's frozen plan fields.
    from neurodic.agent.config import merge_sparse_override
    from neurodic.agent.inspect import resolve_config
    baseline = resolve_config(plan["baseline"]["config_source"], case_key="pin_2d", case_paths=plan["baseline"]["case_paths_source"])["effective_config"]
    values, _ = merge_sparse_override(baseline, plan["override"], solver="pin")
    (tmp_path / "one").mkdir(); (tmp_path / "two").mkdir()
    first = _stage_signature(plan, values, action, plan["execution_actions"][0]["covers_stages"])
    assert _execution_overlay(values, tmp_path / "one") ["output"] != _execution_overlay(values, tmp_path / "two")["output"]
    second = _stage_signature(plan, values, action, plan["execution_actions"][0]["covers_stages"])
    assert first.digest == second.digest


def test_scientific_config_scope_and_adapter_version_change_signature() -> None:
    action = guarded_pin_action(); first_plan = _plan("pin_sig_a"); second_plan = _plan("pin_sig_b", seed_iterations=4998, scope={"selected_frame": 0})
    from neurodic.agent.config import merge_sparse_override
    from neurodic.agent.inspect import resolve_config
    def signature(plan, action):
        baseline = resolve_config(plan["baseline"]["config_source"], case_key="pin_2d", case_paths=plan["baseline"]["case_paths_source"])["effective_config"]
        values, _ = merge_sparse_override(baseline, plan["override"], solver="pin")
        return _stage_signature(plan, values, action, plan["execution_actions"][0]["covers_stages"])
    one = signature(first_plan, action); two = signature(second_plan, action)
    assert one.digest != two.digest
    from neurodic.agent.execution import TrustedAction
    changed = TrustedAction(action.action_id, action.run, "neurodic.pin.full_solve/v2", action.output_contract, action.input_identities)
    assert one.digest != signature(first_plan, changed).digest


def test_input_identity_is_full_content_identity() -> None:
    values = _effective_values(_plan("pin_inputs"))
    inputs = _input_identities(_plan("pin_inputs"), values)
    assert set(inputs) == {"baseline_config", "reference_image", "deformed_image", "roi_mask"}
    assert all(inputs[key]["strength"] == "content" for key in ("reference_image", "deformed_image", "roi_mask"))


def _effective_values(plan: dict) -> dict:
    from neurodic.agent.config import merge_sparse_override
    from neurodic.agent.inspect import resolve_config
    baseline = resolve_config(plan["baseline"]["config_source"], case_key="pin_2d", case_paths=plan["baseline"]["case_paths_source"])["effective_config"]
    return merge_sparse_override(baseline, plan["override"], solver="pin")[0]


def test_invalid_output_rejects_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pin(monkeypatch, valid=False)
    report = execute_trial(_plan("pin_invalid"), managed_root=tmp_path).to_dict()["data"]["execution"]
    assert report["execution_status"] == "failed"
    assert not list((tmp_path / "trials/pin_invalid/artifacts").rglob("*"))


def test_interrupted_pin_call_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pin(monkeypatch, interrupt=True)
    report = execute_trial(_plan("pin_interrupted"), managed_root=tmp_path).to_dict()["data"]["execution"]
    assert report["execution_status"] == "interrupted"
    assert not list((tmp_path / "trials/pin_interrupted/artifacts").rglob("*"))


def test_safe_reuse_skips_the_fake_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_pin(monkeypatch)
    execute_trial(_plan("pin_source"), managed_root=tmp_path)
    reused = execute_trial(_plan("pin_target"), managed_root=tmp_path).to_dict()["data"]["execution"]
    assert len(calls) == 1 and reused["stage_attempts"][0]["status"] == "reused"


def test_tampered_managed_artifact_cannot_be_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_pin(monkeypatch)
    execute_trial(_plan("pin_tampered_source"), managed_root=tmp_path)
    artifact = next((tmp_path / "trials/pin_tampered_source/artifacts").rglob("pin_result.npz")); artifact.write_bytes(b"tampered")
    execute_trial(_plan("pin_tampered_target"), managed_root=tmp_path)
    assert len(calls) == 2


def test_missing_scope_blocks_action_plan() -> None:
    plan = _plan("pin_scope_missing", scope={})
    assert plan["plan_status"] == "blocked"
    assert plan["policy_violations"][0]["code"] == "TRIAL.SCOPE_REQUIRED"
