"""Managed-only PIN Multi evaluation: explicit evidence, never legacy discovery."""
from __future__ import annotations

import json

import pytest

from neurodic.agent.evaluate import evaluate_pin_multi_managed_result
from test_agent_execution_pin_multi_fusion import _execute, _fixture


def _evidence(fx, execution):
    report = fx["report"]
    ordered = []
    for pair, dependency in zip(report["scope"]["planned_pair_ids"], fx["deps"]):
        source = fx["managed"] / "trials" / dependency["source_trial_id"]
        manifest = json.loads((source / "manifest.json").read_text())
        attempt = dependency["source_attempt_id"]
        signature = next(item["producer_signature"] for item in manifest["stage_attempts"] if item["stage_attempt_id"] == attempt)
        ordered.append({"pair_id": pair, "source_trial_id": dependency["source_trial_id"],
                        "source_attempt_id": attempt, "producer_signature": signature})
    stage = execution["stage_attempts"][0]
    return dict(managed_root=fx["managed"], ordered_pair_results=ordered,
                fusion_result={"trial_id": execution["trial_id"], "attempt_id": stage["stage_attempt_id"]},
                selected_frame=0, expected_pair_ids=report["scope"]["planned_pair_ids"],
                expected_planned_pair_set_identity=report["planned_pair_set_identity"],
                expected_fusion_input_identity=report["fusion_input_identity"],
                expected_fusion_producer_signature=stage["producer_signature"]["digest"],
                case_key="fake", case_paths=fx["paths"])


def _evaluate(fx, execution):
    return evaluate_pin_multi_managed_result(fx["config"], **_evidence(fx, execution)).to_dict()


def test_managed_evaluation_uses_explicit_sources_and_is_deterministic(tmp_path, monkeypatch):
    fx = _fixture(tmp_path)
    execution = _execute(fx, monkeypatch, trial="c34-real")
    first = _evaluate(fx, execution)["data"]
    second = _evaluate(fx, execution)["data"]
    assert first["quality"]["schema_version"] == "neurodic.quality/v1"
    assert first["quality"]["provenance"]["managed_only"] is True
    assert first["evaluation_identity"] == second["evaluation_identity"]
    assert first["quality"]["metrics"] == second["quality"]["metrics"]
    assert next(item for item in first["quality"]["metrics"] if item["id"] == "fusion.deduplicated_points")["availability"] == "not_available"


def test_managed_evaluation_rejects_stale_c2_and_never_uses_legacy(tmp_path, monkeypatch):
    fx = _fixture(tmp_path)
    execution = _execute(fx, monkeypatch, trial="c34-stale")
    evidence = _evidence(fx, execution)
    evidence["expected_fusion_input_identity"] = "sha256:stale"
    with pytest.raises(ValueError, match="C2 identity"):
        evaluate_pin_multi_managed_result(fx["config"], **evidence)
    evidence = _evidence(fx, execution)
    quality = fx["managed"] / "trials" / evidence["ordered_pair_results"][0]["source_trial_id"] / "artifacts/pin_multi.pair_solve_quality_call/a1/scientific/pairs/cam_1__cam_2/quality/quality.json"
    quality.unlink()
    with pytest.raises(ValueError, match="contract|identity|invalid"):
        evaluate_pin_multi_managed_result(fx["config"], **evidence)


def test_managed_evaluation_rejects_pair_order_and_fusion_tamper(tmp_path, monkeypatch):
    fx = _fixture(tmp_path)
    execution = _execute(fx, monkeypatch, trial="c34-tamper")
    evidence = _evidence(fx, execution)
    evidence["ordered_pair_results"].reverse()
    with pytest.raises(ValueError, match="exact ordered"):
        evaluate_pin_multi_managed_result(fx["config"], **evidence)
    evidence = _evidence(fx, execution)
    stage = execution["stage_attempts"][0]["stage_attempt_id"]
    summary = fx["managed"] / "trials" / execution["trial_id"] / "artifacts" / "pin_multi.fusion_postprocess_call" / stage / "scientific/fused/summary.json"
    summary.write_text("{}")
    with pytest.raises(ValueError, match="content identity"):
        evaluate_pin_multi_managed_result(fx["config"], **evidence)
