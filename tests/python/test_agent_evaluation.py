"""Loop 3 evidence/quality tests use existing metadata only."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import pytest
from neurodic.agent.evaluate import evaluate_result
from neurodic.agent.schemas import Availability, MetricRecord

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("config,key,solver",[("pin_2d.yaml","pin_2d","pin"),("pin_stereo.yaml","pin_stereo","pin_stereo"),("pin_multi.yaml","pin_multi","pin_multi"),("ndef_multi.yaml","ndef_multi","ndef")])
def test_all_solver_quality_reports_are_native_free(config,key,solver):
    q=evaluate_result(ROOT/"config"/config,case_key=key,case_paths=ROOT/"config/case_paths.yaml").to_dict()["data"]["quality"]
    assert q["solver"]==solver and q["schema_version"]=="neurodic.quality/v1"
    assert q["status"] in {"unknown","pass","warning","fail"}
    assert all(item["availability"] in {x.value for x in Availability} for item in q["metrics"])

def test_ndef_and_pin_multi_extract_real_evidence_without_checkpoint_loading():
    for config,key,required in [("ndef_multi.yaml","ndef_multi",{"precalculation.track_ratio","training.loss.final","field.displacement.finite_ratio"}),("pin_multi.yaml","pin_multi",{"pin_multi.pair.valid_ratio"})]:
        q=evaluate_result(ROOT/"config"/config,case_key=key,case_paths=ROOT/"config/case_paths.yaml").to_dict()["data"]["quality"]
        assert required <= {x["id"] for x in q["metrics"]}

    ndef=evaluate_result(ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",case_paths=ROOT/"config/case_paths.yaml").to_dict()["data"]["quality"]
    values={x["id"]:x["value"] for x in ndef["metrics"]}
    assert values["precalculation.track_ratio"] < values["precalculation.inlier_ratio"]

def test_pin_known_limitation_is_not_promoted_to_pass():
    q=evaluate_result(ROOT/"config/pin_2d.yaml",case_key="pin_2d",case_paths=ROOT/"config/case_paths.yaml").to_dict()["data"]["quality"]
    fixed=next(x for x in q["metrics"] if x["id"]=="evaluation.photometric_residual.mean")
    assert fixed["availability"]=="not_available" and q["status"]=="unknown"

def test_metric_contract_rejects_missing_available_value_and_strict_json():
    with pytest.raises(ValueError): MetricRecord("x",Availability.OBSERVED,"ratio",{},None)
    metric=MetricRecord("x",Availability.NOT_AVAILABLE,"ratio",{},None)
    assert json.loads(json.dumps(metric.to_dict(),allow_nan=False))["availability"]=="not_available"

def test_evaluate_is_read_only_and_cli_quality_failure_is_exit_zero():
    case=ROOT/"case"/"Multi"/"CylinderDIC"
    before=sorted((str(x.relative_to(case)),x.stat().st_mtime_ns) for x in case.rglob("*") if x.is_file())
    evaluate_result(ROOT/"config/ndef_multi.yaml",case_key="ndef_multi",case_paths=ROOT/"config/case_paths.yaml")
    after=sorted((str(x.relative_to(case)),x.stat().st_mtime_ns) for x in case.rglob("*") if x.is_file())
    assert before==after
    env={**os.environ,"PYTHONPATH":str(ROOT/"python")}
    result=subprocess.run([sys.executable,"-m","neurodic.cli","evaluate","--config","config/pin_2d.yaml","--case-key","pin_2d","--format","json"],cwd=ROOT,env=env,text=True,capture_output=True)
    payload=json.loads(result.stdout)
    assert result.returncode==0 and payload["status"]=="ok" and payload["data"]["quality"]["status"]=="unknown" and result.stderr==""
