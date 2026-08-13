"""Loop 9 bounded-rule tests: synthetic reports only, never solver execution."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
from neurodic.agent.diagnose import diagnose_quality_report
from neurodic.agent.recommend import recommend_from_diagnosis

ROOT=Path(__file__).resolve().parents[2]
def diagnosis(*, support="strong", cause="OPTIMIZATION.STEP_INSTABILITY", contradict=(), evidence=("training.history.finite_ratio","training.valid_pair_ratio.final"), family="TRAINING.NUMERICAL_FAILURE", solver="ndef"):
    return {"schema_version":"neurodic.diagnosis/v1","solver":solver,"scope":{"frame":1},"overall_status":"diagnosed","primary_diagnosis":"X","diagnoses":[{"code":"X","failure_stage":"ndef.deformation.train","failure_family":family,"support":support,"role":"primary","supporting_evidence":[{"metric_id":x} for x in evidence],"contradicting_evidence":[{"metric_id":x} for x in contradict],"missing_evidence":[],"candidate_causes":[{"cause_code":cause}],"next_observation":None}],"checked_stages":[],"missing_evidence":[],"notes":[]}
def rec(d): return recommend_from_diagnosis(d,ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",trial_id="r1").to_dict()["data"]["recommendation"]

def test_strong_numeric_rule_is_single_sparse_planner_validated_change(monkeypatch):
    import neurodic.agent.execution as execution
    monkeypatch.setattr(execution,"execute_trial",lambda *a,**k: pytest.fail("execution prohibited"))
    out=rec(diagnosis()); assert out["recommendation_status"]=="recommended"; assert out["sparse_override"]=={"deformation_training":{"photometric_learning_rate":0.0015}}
    assert len(out["parameter_changes"])==1 and out["planning_result"] is not None

def test_support_contradiction_missing_and_bounds_fail_closed(tmp_path):
    assert rec(diagnosis(support="weak"))["recommendation_status"]=="observation_only"
    assert rec(diagnosis(support="insufficient"))["recommendation_status"]=="observation_only"
    assert rec(diagnosis(contradict=("training.history.stable",)))["recommendation_status"]=="blocked_by_contradiction"
    assert rec(diagnosis(evidence=("training.history.finite_ratio",)))["recommendation_status"]=="insufficient_evidence"
    registry=(ROOT/"config/agent/parameter_registry.yaml").read_text(); bad=tmp_path/"registry.yaml"; bad.write_text(registry.replace("0.000375, 0.003", "0.000375, 0.001"),encoding="utf-8")
    assert recommend_from_diagnosis(diagnosis(),ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",parameter_registry=bad).to_dict()["data"]["recommendation"]["recommendation_status"]=="insufficient_evidence"

def test_protected_and_change_count_registry_rules_are_rejected(tmp_path):
    registry=(ROOT/"config/agent/parameter_registry.yaml").read_text(); protected=tmp_path/"protected.yaml"
    protected.write_text(registry.replace("auto_recommendable: false, auto_safe_range: null, direction_semantics: protected_world_scale", "auto_recommendable: true, auto_safe_range: [0.5, 2.0], direction_semantics: protected_world_scale", 1),encoding="utf-8")
    with pytest.raises(ValueError): recommend_from_diagnosis(diagnosis(),ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",parameter_registry=protected)
    rules=(ROOT/"config/agent/intervention_rules.yaml").read_text(); many=tmp_path/"many.yaml"; many.write_text(rules.replace("parameter_changes: [{path: deformation_training.photometric_learning_rate, policy: multiply, factor: 0.5}]", "parameter_changes: [{path: deformation_training.photometric_learning_rate, policy: multiply, factor: 0.5}, {path: deformation_training.photometric_learning_rate, policy: multiply, factor: 0.5}, {path: deformation_training.photometric_learning_rate, policy: multiply, factor: 0.5}]"),encoding="utf-8")
    with pytest.raises(ValueError): recommend_from_diagnosis(diagnosis(),ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",intervention_rules=many)

def test_four_solver_unsupported_and_diagnosis_numeric_taxonomy():
    for solver, config, key, family, cause in (("pin", "pin_2d.yaml", "pin_2d", "FIELD.NONFINITE", "FIELD.NONFINITE_OUTPUT"),("pin_stereo","pin_stereo.yaml","pin_stereo","RECONSTRUCTION.REPROJECTION_INCONSISTENCY","CAMERA_GEOMETRY.INCONSISTENT"),("pin_multi","pin_multi.yaml","pin_multi","PAIR.NO_VALID_RECONSTRUCTION","PAIR.RECONSTRUCTION_REJECTION"),("ndef","ndef_multi.yaml","ndef_multi","PRECALC.NO_VALID_TRACKS","CORRESPONDENCE.INSUFFICIENT_SUPPORT")):
        d=diagnosis(solver=solver,family=family,cause=cause); d["diagnoses"][0]["failure_stage"]="pin.train" if solver=="pin" else d["diagnoses"][0]["failure_stage"]
        out=recommend_from_diagnosis(d,ROOT/"config"/config,case_key=key).to_dict()["data"]["recommendation"]
        assert out["recommendation_status"]=="no_matching_rule"
    q={"schema_version":"neurodic.quality/v1","solver":"ndef","scope":{},"status":"fail","metrics":[{"id":"training.history.finite_ratio","availability":"derived","value":.5,"unit":"ratio","source":{},"scope":{}},{"id":"training.valid_pair_ratio.final","availability":"derived","value":.5,"unit":"ratio","source":{},"scope":{}}],"threshold_results":[],"findings":[]}
    assert diagnose_quality_report(q).to_dict()["primary_diagnosis"]=="NDEF.TRAINING.NUMERICAL_FAILURE"
