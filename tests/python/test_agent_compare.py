"""Loop 8 synthetic QualityReport tests; no solver or execution calls."""
from __future__ import annotations
import copy, json
from pathlib import Path
import pytest
from neurodic.agent.best import load_best, update_best
from neurodic.agent.compare import compare_quality_reports

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/comparison_profiles/default.yaml"
IDENTITY = {"case":"synthetic-cylinder","calibration":"sha256:cal","units":"mm","frame":0}

def metric(id, value, *, unit="ratio", scope=None, availability="observed", aggregation=None, sample_count=10):
    return {"id":id,"value":value,"unit":unit,"availability":availability,"source":{},"scope":scope or {},"aggregation":aggregation,"sample_count":sample_count}
def quality(solver="pin", metrics=(), *, scope=None, findings=(), kind="full", execution="completed"):
    return {"schema_version":"neurodic.quality/v1","solver":solver,"scope":scope or {"frame":0},"status":"pass","metrics":list(metrics),"threshold_results":[],"findings":list(findings),"provenance":{"scientific_identity":IDENTITY,"result_kind":kind,"execution_status":execution}}
def compare(a,b): return compare_quality_reports(a,b,profile=PROFILE).to_dict()["data"]["comparison"]

def test_direction_tolerance_missing_unit_scope_and_fixed_evaluation() -> None:
    a=quality(metrics=[metric("evaluation.photometric_residual.mean",1.0,unit="photometric_objective",scope={"evaluation_set_identity":"fixed-a"}),metric("evaluation.valid_ratio",.9)])
    b=quality(metrics=[metric("evaluation.photometric_residual.mean",.8,unit="photometric_objective",scope={"evaluation_set_identity":"fixed-a"}),metric("evaluation.valid_ratio",None,availability="not_available")])
    r=compare(a,b); assert "evaluation.photometric_residual.mean" in r["summary"]["improved_metrics"]
    assert "evaluation.valid_ratio" in r["summary"]["incomparable_metrics"]
    b["metrics"][0]["scope"]={"evaluation_set_identity":"changed"}; assert compare(a,b)["metric_comparisons"][0]["comparison_status"] == "incomparable"
    b=copy.deepcopy(a); b["metrics"][0]["unit"]="px"; assert compare(a,b)["metric_comparisons"][0]["comparison_status"] == "incomparable"
    b=copy.deepcopy(a); b["scope"]={"frame":1}; assert compare(a,b)["comparability"]["status"] == "incomparable"

def test_eligibility_and_solver_policies() -> None:
    pin_a=quality(metrics=[metric("field.displacement.finite_ratio",1),metric("evaluation.photometric_residual.mean",1,unit="photometric_objective",scope={"evaluation_set_identity":"x"}),metric("evaluation.valid_ratio",1)])
    pin_b=copy.deepcopy(pin_a); pin_b["metrics"][1]["value"]=.5; pin_b["findings"]=[{"code":"FIELD.NONFINITE"}]
    assert compare(pin_a,pin_b)["eligibility"]["status"] == "ineligible"
    stereo_a=quality("pin_stereo",[metric("reconstruction.valid_ratio",.5),metric("reconstruction.reprojection.p95",2,unit="px"),metric("stereo.evaluation.photometric_residual.mean",2,unit="photometric_objective",scope={"evaluation_set_identity":"x"})])
    stereo_b=copy.deepcopy(stereo_a); stereo_b["metrics"][1]["value"]=1; assert compare(stereo_a,stereo_b)["selection_decision"]["decision"] == "candidate_preferred"
    ndef=quality("ndef",[metric("precalculation.track_ratio",1),metric("training.valid_pair_ratio.final",1),metric("field.displacement.finite_ratio",1),metric("evaluation.photometric_residual.mean",1,unit="photometric_objective",scope={"evaluation_set_identity":"x"}),metric("evaluation.valid_ratio",1)]); bad=copy.deepcopy(ndef); bad["metrics"][3]["value"]=.5; bad["findings"]=[{"code":"FIELD.NONFINITE"}]; assert compare(ndef,bad)["eligibility"]["status"] == "ineligible"
    multi=quality("pin_multi",[metric("pin_multi.pair.valid_ratio",.5),metric("pin_multi.pair.reprojection.p95",2,unit="px")]); corrupt=copy.deepcopy(multi); corrupt["metrics"][0]["availability"]="corrupt"; corrupt["metrics"][0]["value"]=None; assert compare(multi,corrupt)["eligibility"]["status"] == "ineligible"

def test_partial_determinism_pareto_and_best_lifecycle(tmp_path: Path) -> None:
    a=quality(metrics=[metric("field.displacement.finite_ratio",1),metric("evaluation.photometric_residual.mean",1,unit="photometric_objective",scope={"evaluation_set_identity":"x"}),metric("evaluation.valid_ratio",1,scope={"evaluation_set_identity":"x"})])
    b=copy.deepcopy(a); b["metrics"][1]["value"]=.5; b["metrics"][2]["value"]=.8
    r=compare(a,b); assert r["selection_decision"]["decision"] == "retain_current" # guardrail regression
    partial=copy.deepcopy(a); partial["provenance"]["result_kind"]="stage_partial"; partial["provenance"]["execution_status"]="partial"; assert compare(a,partial)["eligibility"]["status"] == "ineligible"
    c=copy.deepcopy(a); c["metrics"][1]["value"]=.5
    first=compare(a,c); assert first == compare(a,c)
    baseline=tmp_path/"baseline.json"; baseline.write_text(json.dumps(a),encoding="utf-8")
    candidate=tmp_path/"candidate.json"; candidate.write_text(json.dumps(c),encoding="utf-8")
    promoted=update_best(first,candidate_quality=candidate,baseline_quality=baseline,managed_root=tmp_path,expected_current_best_identity=None).to_dict()["data"]["best"]
    assert load_best(tmp_path).to_dict()["data"]["best"]["best_identity"] == promoted["best_identity"]
    c["metrics"][1]["value"]=.4; candidate.write_text(json.dumps(c),encoding="utf-8")
    with pytest.raises(ValueError,match="BEST.COMPARISON_STALE"): update_best(first,candidate_quality=candidate,baseline_quality=baseline,managed_root=tmp_path,expected_current_best_identity=promoted["best_identity"])
    assert list((tmp_path/"best/history").glob("*.json"))
