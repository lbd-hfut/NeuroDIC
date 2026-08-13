"""Native-free deterministic diagnosis-rule fixtures."""
from __future__ import annotations
import json
from neurodic.agent.diagnose import diagnose_quality_report
from neurodic.agent.schemas import canonical_json

def _quality(solver, metrics, status="fail"):
    return {"schema_version":"neurodic.quality/v1","solver":solver,"scope":{"frame":1},"status":status,"metrics":[{"id":k,"availability":"derived","value":v,"unit":"ratio","source":{},"scope":{}} for k,v in metrics.items()],"threshold_results":[],"findings":[]}

def test_ndef_upstream_first_and_determinism():
    q=_quality("ndef",{"precalculation.track_ratio":0.,"training.valid_pair_ratio.final":0.,"evaluation.valid_ratio":0.,"evaluation.current_projection.positive_depth_ratio":0.,"evaluation.current_projection.in_bounds_ratio":1.,"evaluation.patch_valid_ratio":1.})
    a=diagnose_quality_report(q).to_dict(); b=diagnose_quality_report(q).to_dict()
    assert a["primary_diagnosis"]=="NDEF.PRECALC.NO_VALID_TRACKS" and canonical_json(a)==canonical_json(b)
    assert all("recommend" not in canonical_json(a).lower() and "rerun" not in canonical_json(a).lower() for _ in [0])

def test_stereo_contradiction_and_missing_evidence():
    q=_quality("pin_stereo",{"reconstruction.valid_ratio":.2,"reconstruction.reason.reprojection_error_ratio":.8,"reconstruction.reference_reprojection.p95":1.,"reconstruction.current_reprojection.p95":9.})
    report=diagnose_quality_report(q).to_dict(); item=next(x for x in report["diagnoses"] if x["code"]=="STEREO.RECONSTRUCTION.REPROJECTION_INCONSISTENCY")
    assert item["support"]=="moderate" and item["contradicting_evidence"]
    missing=diagnose_quality_report(_quality("pin",{})).to_dict(); assert missing["overall_status"]=="insufficient_evidence" and missing["missing_evidence"]

def test_pin_and_multi_conservative_rules():
    pin=diagnose_quality_report(_quality("pin",{"field.displacement.finite_ratio":.9,"evaluation.valid_ratio":0.})).to_dict()
    assert {x["code"] for x in pin["diagnoses"]}=={"PIN.FIELD.NONFINITE","PIN.EVALUATION.NO_VALID_OBSERVATIONS"}
    multi=diagnose_quality_report(_quality("pin_multi",{"pin_multi.pair.valid_ratio":0.,"fusion.preselection.overlap_group_count":0.})).to_dict()
    assert multi["primary_diagnosis"]=="PIN_MULTI.PAIR.NO_VALID_RECONSTRUCTION"
