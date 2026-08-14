"""Loop 10B CLI and guarded-capability compatibility checks; native-free."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import pytest
from neurodic.agent.execution import TrustedAction, execute_trial
from neurodic.agent.trials import plan_trial

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
def cli(*args: str): return subprocess.run([sys.executable, "-m", "neurodic.cli", *args], cwd=ROOT, env=ENV, text=True, capture_output=True)
def plan(trial_id: str, solver="pin"):
    config = ROOT / ("config/pin_2d.yaml" if solver == "pin" else "config/pin_multi.yaml")
    key = "pin_2d" if solver == "pin" else "pin_multi"
    override = {"training":{"seed_iterations":4999}} if solver == "pin" else {"pair_roi":{"max_features":11999}}
    scope = {"selected_frame": 0} if solver == "pin" else {"pair_id":"cam_0__cam_1"}
    return plan_trial(config, case_key=key, case_paths=ROOT/"config/case_paths.yaml", override=override, trial_id=trial_id, scope=scope).to_dict()["data"]["trial_plan"]

def test_help_and_read_only_imports_are_native_free():
    commands = [(), ("inspect",), ("evaluate",), ("diagnose",), ("recommend",), ("trial",), ("trial","plan"), ("trial","execute"), ("compare",), ("best",), ("best","show"), ("best","evaluate"), ("best","promote")]
    for command in commands: assert cli(*command, "--help").returncode == 0
    import neurodic.agent.compare, neurodic.agent.diagnose, neurodic.agent.recommend, neurodic.agent.trials
    assert "neurodic._neurodic" not in sys.modules and "torch" not in sys.modules

def test_json_stdout_exit_semantics_and_structured_errors(tmp_path: Path):
    inspected = cli("inspect","config","--config","config/pin_2d.yaml","--case-key","pin_2d","--format","json")
    assert inspected.returncode == 0 and json.loads(inspected.stdout)["status"] == "ok"
    evaluated = cli("evaluate","--config","config/pin_2d.yaml","--case-key","pin_2d","--format","json")
    assert evaluated.returncode == 0 and json.loads(evaluated.stdout)["operation"] == "evaluate.result"
    quality={"schema_version":"neurodic.quality/v1","solver":"pin","scope":{},"status":"fail","metrics":[{"id":"field.displacement.finite_ratio","availability":"derived","value":0.,"unit":"ratio","source":{},"scope":{}},{"id":"evaluation.valid_ratio","availability":"derived","value":0.,"unit":"ratio","source":{},"scope":{}}],"threshold_results":[],"findings":[]}
    quality_path=tmp_path/"quality.json"; quality_path.write_text(json.dumps(quality),encoding="utf-8")
    diagnosed=cli("diagnose","--quality",str(quality_path),"--format","json")
    assert diagnosed.returncode == 0 and json.loads(diagnosed.stdout)["data"]["diagnosis"]["primary_diagnosis"] is not None
    diagnosis = {"schema_version":"neurodic.diagnosis/v1","solver":"pin","scope":{},"overall_status":"diagnosed","primary_diagnosis":"X","diagnoses":[{"code":"X","failure_stage":"pin.train","failure_family":"FIELD.NONFINITE","support":"strong","role":"primary","supporting_evidence":[],"contradicting_evidence":[],"missing_evidence":[],"candidate_causes":[{"cause_code":"FIELD.NONFINITE_OUTPUT"}]}],"checked_stages":[],"missing_evidence":[],"notes":[]}
    path=tmp_path/"diagnosis.json"; path.write_text(json.dumps(diagnosis),encoding="utf-8")
    recommended=cli("recommend","--diagnosis",str(path),"--config","config/pin_2d.yaml","--case-key","pin_2d","--format","json")
    assert recommended.returncode == 0 and json.loads(recommended.stdout)["data"]["recommendation"]["recommendation_status"] == "no_matching_rule"
    candidate=dict(quality); candidate["provenance"]={"scientific_identity":{"case":"x"},"result_kind":"full","execution_status":"completed"}; baseline=dict(candidate); baseline["metrics"]=[dict(x) for x in candidate["metrics"]]; baseline["metrics"][0]["value"]=1.
    base_path=tmp_path/"baseline.json"; candidate_path=tmp_path/"candidate.json"; base_path.write_text(json.dumps(baseline),encoding="utf-8"); candidate_path.write_text(json.dumps(candidate),encoding="utf-8")
    compared=cli("compare","--baseline",str(base_path),"--candidate",str(candidate_path),"--format","json")
    assert compared.returncode == 0 and json.loads(compared.stdout)["operation"] == "compare.quality"
    bad=cli("inspect","config","--config","missing.yaml","--format","json")
    payload=json.loads(bad.stdout); assert bad.returncode != 0 and payload["status"] == "error" and {"code","message"}.issubset(payload["errors"][0])

def test_runtime_capabilities_unsupported_fail_closed_and_partial_reuse(tmp_path: Path):
    multi=plan("compat_multi", "pin"); pin_actions={x["action_id"]:x for x in multi["execution_actions"]}
    assert pin_actions["pin.combined_solver_call"]["execution_supported"] is True
    assert pin_actions["pin.combined_solver_call"]["completion_scope"] == "combined_action"
    pair=plan("compat_pair", "pin_multi"); actions={x["action_id"]:x for x in pair["execution_actions"]}
    assert actions["pin_multi.separate_pair_roi_call"]["execution_supported"] is True
    assert actions["pin_multi.separate_pair_roi_call"]["completion_scope"] == "requested_action_only"
    assert actions["pin_multi.combined_solver_call"]["execution_supported"] is False
    plan_path=tmp_path/"unsupported.json"; plan_path.write_text(json.dumps(multi),encoding="utf-8")
    rejected=cli("trial","execute","--plan",str(plan_path),"--managed-root",str(tmp_path),"--action","pin.not_an_approved_action","--format","json")
    assert rejected.returncode != 0 and json.loads(rejected.stdout)["errors"][0]["code"] == "EXECUTION.UNSUPPORTED"
    calls=0
    def run(_values, staging, _scope):
        nonlocal calls; calls += 1; (staging/"ok.json").write_text("{}",encoding="utf-8"); return ["ok.json"]
    action={"pin_multi.separate_pair_roi_call": TrustedAction("pin_multi.separate_pair_roi_call",run,"neurodic.test.compat/v1")}
    first=execute_trial(pair,managed_root=tmp_path, trusted_actions=action, action_id="pin_multi.separate_pair_roi_call").to_dict()["data"]["execution"]
    assert first["execution_status"] == "partial" and first["stage_attempts"][0]["status"] == "completed"
    second_plan=plan("compat_pair_reuse", "pin_multi")
    second=execute_trial(second_plan,managed_root=tmp_path, trusted_actions=action, action_id="pin_multi.separate_pair_roi_call").to_dict()["data"]["execution"]
    assert calls == 1 and second["execution_status"] == "partial" and second["stage_attempts"][0]["status"] == "reused" and second["reused_artifacts"]

def test_skills_reference_real_cli_and_schema_versions():
    texts="\n".join(path.read_text(encoding="utf-8") for path in (ROOT/"skills").rglob("SKILL.md"))
    assert "PYTHONPATH=python python -m neurodic.cli" in texts
    from neurodic.agent.compare import COMPARISON_SCHEMA_VERSION
    from neurodic.agent.execution import EXECUTION_SCHEMA_VERSION
    from neurodic.agent.recommend import RECOMMENDATION_SCHEMA_VERSION
    from neurodic.agent.schemas import DIAGNOSIS_SCHEMA_VERSION, QUALITY_SCHEMA_VERSION
    from neurodic.agent.trials import TRIAL_PLAN_SCHEMA_VERSION
    from neurodic.agent.best import BEST_SCHEMA_VERSION
    assert (QUALITY_SCHEMA_VERSION, DIAGNOSIS_SCHEMA_VERSION, RECOMMENDATION_SCHEMA_VERSION, TRIAL_PLAN_SCHEMA_VERSION, EXECUTION_SCHEMA_VERSION, COMPARISON_SCHEMA_VERSION, BEST_SCHEMA_VERSION) == ("neurodic.quality/v1","neurodic.diagnosis/v1","neurodic.recommendation/v1","neurodic.trial_plan/v1","neurodic.execution/v1","neurodic.comparison/v1","neurodic.best/v1")
