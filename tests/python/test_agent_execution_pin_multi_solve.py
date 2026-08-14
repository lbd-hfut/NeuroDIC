"""Native-free guarded lifecycle coverage for the C1 PIN Multi action."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
from neurodic.agent.artifacts import content_identity
from neurodic.agent.execution import (ProducerSignature, TrustedAction, _stage_signature,
                                      _verified_reuse, execute_trial)
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.errors import ControlPlaneError
from neurodic.agent.trials import plan_trial
from neurodic.agent.adapters.execution_pin_multi import _solve_outputs

ROOT = Path(__file__).resolve().parents[2]


def _plan(managed: Path, trial_id: str):
    location = "artifacts/pin_multi.separate_pair_roi_call/a1/left_mask.npy"; source = managed / "trials/source"
    mask = source / location; mask.parent.mkdir(parents=True, exist_ok=True); mask.write_bytes(b"managed-roi")
    signature = {"stage_id":"pin_multi.separate_pair_roi_call", "scope":{"pair_id":"cam_0__cam_1"}, "digest":"sha256:fake-roi"}
    artifact = {"location":location,"stage_attempt_id":"a1","producer_action_id":"pin_multi.separate_pair_roi_call","producer_signature":signature,"identity":content_identity(mask).to_dict()}
    (source / "manifest.json").write_text(json.dumps({"trial_id":"source", "stage_attempts":[{"stage_attempt_id":"a1","status":"completed","action_id":"pin_multi.separate_pair_roi_call","producer_signature":signature}], "produced_artifacts":[artifact]}))
    dep = {"dependency_id":"pair_roi","producer_action_id":"pin_multi.separate_pair_roi_call","producer_signature":signature,"scope":{"pair_id":"cam_0__cam_1"},"source_trial_id":"source","source_attempt_id":"a1","required_artifacts":[{"relative_path":location,"identity":artifact["identity"]}]}
    return plan_trial(ROOT/"config/pin_multi.yaml",case_key="pin_multi",case_paths=ROOT/"config/case_paths.yaml",trial_id=trial_id,restore_missing=True,scope={"pair_id":"cam_0__cam_1","selected_frame":0},upstream_dependencies=[dep]).to_dict()["data"]["trial_plan"]


def _fake(monkeypatch, *, metadata=None, omit=None, interrupt=False, corrupt=None, high_error=False):
    import neurodic.api.pin_multi_slover_dic as api
    def solve(_values, **kwargs):
        if interrupt: raise KeyboardInterrupt()
        assert kwargs["pair_id"] == "cam_0__cam_1" and kwargs["selected_frame"] == 0
        root = Path(kwargs["result_root"]); pair = kwargs["pair_id"]
        for item in _solve_outputs(pair):
            if item.path == omit: continue
            path = root.parent / item.path; path.parent.mkdir(parents=True, exist_ok=True)
            if item.path.endswith("pair_metadata.json"):
                path.write_text(json.dumps(metadata or {"pair_id":pair,"reference_camera":"cam_0","secondary_camera":"cam_1","selected_frame":0}))
            elif item.path.endswith(".npz"):
                if "/disp/" in item.path:
                    if corrupt == "npz" and item.path.endswith("reference_disparity.npz"): path.write_bytes(b"not-an-npz"); continue
                    if corrupt == "missing_key" and item.path.endswith("reference_disparity.npz"): np.savez(path,coordinates=np.ones((1,2)),iterations=np.array(1),final_loss=np.array(.1)); continue
                    coordinates=np.ones((2,2)) if corrupt == "shape" and item.path.endswith("reference_disparity.npz") else np.ones((1,2))
                    displacement=np.ones((1,2));
                    if corrupt == "nan" and item.path.endswith("reference_disparity.npz"): displacement[0,0]=np.nan
                    if corrupt == "inf" and item.path.endswith("reference_disparity.npz"): displacement[0,0]=np.inf
                    np.savez(path,coordinates=coordinates,displacement=displacement,iterations=np.array(1),final_loss=np.array(.1))
                elif "/reconstruct/" in item.path: np.savez(path,left_coordinates=np.ones((1,2)),right_coordinates=np.ones((1,2)),points=np.ones((1,3)),valid=np.array([False]),reprojection_error=np.ones(1))
                else: np.savez(path,coordinates=np.ones((1,2)),reference_points=np.ones((1,3)),current_points=np.ones((1,3)),displacement=np.ones((1,3)),valid=np.array([False]))
            elif item.path.endswith("reason_codes.npy"): np.save(path,np.array([[1.0]]) if corrupt == "reason" else np.array([1],dtype=np.int8))
            elif item.path.endswith("quality.json"):
                if corrupt == "json": path.write_text("{")
                else:
                    quality={"total_points":1,"valid_points":0,"valid_ratio":0.0,"reason_codes":{"valid":0},"mean_reprojection_error_px":1e30 if high_error else None}
                    if corrupt == "quality_key": quality.pop("valid_ratio")
                    path.write_text(json.dumps(quality))
            else: path.write_text("{}")
    monkeypatch.setattr(api, "solve_pin_multi_pair", solve)


def test_pair_solve_quality_guarded_success_and_reuse(tmp_path, monkeypatch):
    _fake(monkeypatch); first=execute_trial(_plan(tmp_path,"c1-first"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert first["execution_status"] == "partial" and first["stage_attempts"][0]["status"] == "completed"
    _fake(monkeypatch, interrupt=True); second=execute_trial(_plan(tmp_path,"c1-reuse"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert second["stage_attempts"][0]["status"] == "reused"


def test_pair_metadata_and_missing_output_fail_without_publish(tmp_path, monkeypatch):
    _fake(monkeypatch, metadata={"pair_id":"cam_0__cam_2","reference_camera":"cam_0","secondary_camera":"cam_2","selected_frame":0})
    bad=execute_trial(_plan(tmp_path,"c1-wrong"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert bad["execution_status"] == "failed" and not list((tmp_path/"trials/c1-wrong/artifacts").rglob("*"))
    _fake(monkeypatch, omit=_solve_outputs("cam_0__cam_1")[0].path)
    bad=execute_trial(_plan(tmp_path,"c1-missing"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert bad["execution_status"] == "failed"


@pytest.mark.parametrize(("metadata"), [
    {"pair_id":"cam_0__cam_1","reference_camera":"cam_1","secondary_camera":"cam_0","selected_frame":0},
    {"pair_id":"cam_0__cam_1","reference_camera":"cam_0","secondary_camera":"cam_1","selected_frame":1},
])
def test_pair_solve_rejects_reversed_roles_or_wrong_frame(tmp_path, monkeypatch, metadata):
    _fake(monkeypatch, metadata=metadata)
    result=execute_trial(_plan(tmp_path,"c1-meta"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["execution_status"] == "failed" and not list((tmp_path/"trials/c1-meta/artifacts").rglob("*"))


@pytest.mark.parametrize("index", [3, 5, 7, 8])
def test_pair_solve_rejects_each_missing_required_output(tmp_path, monkeypatch, index):
    _fake(monkeypatch, omit=_solve_outputs("cam_0__cam_1")[index].path)
    result=execute_trial(_plan(tmp_path,f"c1-missing-{index}"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["execution_status"] == "failed"


def test_pair_solve_runtime_error_and_interrupt_do_not_publish(tmp_path, monkeypatch):
    import neurodic.api.pin_multi_slover_dic as api
    monkeypatch.setattr(api,"solve_pin_multi_pair",lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError("fake native failure")))
    failed=execute_trial(_plan(tmp_path,"c1-runtime"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert failed["execution_status"] == "failed" and not list((tmp_path/"trials/c1-runtime/artifacts").rglob("*"))
    _fake(monkeypatch, interrupt=True)
    interrupted=execute_trial(_plan(tmp_path,"c1-interrupt"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert interrupted["execution_status"] == "interrupted" and not list((tmp_path/"trials/c1-interrupt/artifacts").rglob("*"))


def _assert_corrupt_output_rejected(tmp_path, monkeypatch, corrupt):
    _fake(monkeypatch, corrupt=corrupt)
    result=execute_trial(_plan(tmp_path,f"c1-corrupt-{corrupt}"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["execution_status"] == "failed"
    assert not list((tmp_path/f"trials/c1-corrupt-{corrupt}/artifacts").rglob("*"))


def test_pair_solve_rejects_corrupt_required_npz(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "npz")


def test_pair_solve_rejects_npz_missing_required_key(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "missing_key")


def test_pair_solve_rejects_corrupt_quality_json(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "json")


def test_pair_solve_rejects_quality_json_missing_required_field(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "quality_key")


def test_pair_solve_rejects_nan_in_required_numeric_array(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "nan")


def test_pair_solve_rejects_inf_in_required_numeric_array(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "inf")


def test_pair_solve_rejects_incompatible_required_array_shape(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "shape")


def test_pair_solve_rejects_invalid_reason_codes_shape_or_dtype(tmp_path, monkeypatch):
    _assert_corrupt_output_rejected(tmp_path, monkeypatch, "reason")


def test_pair_solve_zero_valid_points_is_execution_success(tmp_path, monkeypatch):
    _fake(monkeypatch)
    result=execute_trial(_plan(tmp_path,"c1-zero-valid"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"
    assert result["execution_status"] == "partial"
    assert capability_for("pin_multi.pair_solve_quality_call").completion_scope == "combined_action"


def test_pair_solve_high_error_but_structurally_valid_completes(tmp_path, monkeypatch):
    _fake(monkeypatch, high_error=True)
    result=execute_trial(_plan(tmp_path,"c1-high-error"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"
    assert result["execution_status"] == "partial"


def snapshot_tree(root: Path):
    return {str(path.relative_to(root)):(path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob("*")) if path.is_file()}


def _baseline(tmp_path: Path) -> Path:
    root=tmp_path/"baseline"
    for name in ("config/config.yaml","case/input.dat","legacy_result/old_result.json","calibration/calibration.json","roi_manifest.json","pin_multi_manifest.json","fusion_manifest.json"):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(name)
    return root


def test_pair_solve_guarded_success_is_baseline_zero_write(tmp_path, monkeypatch):
    baseline=_baseline(tmp_path); before=snapshot_tree(baseline)
    sentinels={name:hashlib.sha256((baseline/name).read_bytes()).hexdigest() for name in ("roi_manifest.json","pin_multi_manifest.json","fusion_manifest.json")}
    _fake(monkeypatch); execute_trial(_plan(tmp_path,"c1-zero-write-ok"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call")
    assert snapshot_tree(baseline) == before
    assert {name:hashlib.sha256((baseline/name).read_bytes()).hexdigest() for name in sentinels} == sentinels


def test_pair_solve_guarded_failure_is_baseline_zero_write(tmp_path, monkeypatch):
    baseline=_baseline(tmp_path); before=snapshot_tree(baseline)
    import neurodic.api.pin_multi_slover_dic as api
    monkeypatch.setattr(api,"solve_pin_multi_pair",lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError("fake")))
    execute_trial(_plan(tmp_path,"c1-zero-write-fail"),managed_root=tmp_path,action_id="pin_multi.pair_solve_quality_call")
    assert snapshot_tree(baseline) == before


def _signature_inputs(_plan, values):
    return values["signature_inputs"]


def _signature_plan():
    dependency = {"dependency_id":"pair_roi", "producer_action_id":"pin_multi.separate_pair_roi_call",
                  "producer_signature":{"stage_id":"pin_multi.separate_pair_roi_call", "scope":{"pair_id":"cam_0__cam_1"}, "digest":"sha256:roi-producer"},
                  "scope":{"pair_id":"cam_0__cam_1"}, "source_trial_id":"source-a", "source_attempt_id":"a1",
                  "required_artifacts":[{"relative_path":"artifacts/roi/left_mask.npy", "identity":{"digest":"roi-bytes"}}]}
    return {"solver":"pin_multi", "scope":{"pair_id":"cam_0__cam_1", "selected_frame":0},
            "baseline":{"effective_config_identity":"sha256:baseline"}, "upstream_dependencies":[dependency]}


def _signature_values():
    return {"reconstruction":{"min_views":2}, "signature_inputs":{"calibration":{"digest":"calibration-a"},
            "images":{"reference_image":{"digest":"ref-a"}, "secondary_reference_image":{"digest":"secondary-a"},
                      "current_image":{"digest":"current-a"}, "secondary_current_image":{"digest":"secondary-current-a"}}}}


def _pair_signature(plan, values, implementation="neurodic.pin_multi.pair_solve_quality/v1"):
    action = TrustedAction("pin_multi.pair_solve_quality_call", lambda *_args: (), implementation, input_identities=_signature_inputs)
    return _stage_signature(plan, values, action, ("pin_multi.pair_solve", "pin_multi.pair_quality"))


def test_pair_solve_signature_is_deterministic():
    plan, values = _signature_plan(), _signature_values()
    assert _pair_signature(plan, values).to_dict() == _pair_signature(plan, values).to_dict()


def test_pair_solve_signature_matrix():
    baseline_plan, baseline_values = _signature_plan(), _signature_values(); baseline = _pair_signature(baseline_plan, baseline_values).digest
    def changed_plan(mutator):
        plan = json.loads(json.dumps(baseline_plan)); mutator(plan); return _pair_signature(plan, baseline_values).digest
    def changed_values(mutator):
        values = json.loads(json.dumps(baseline_values)); mutator(values); return _pair_signature(baseline_plan, values).digest
    assert changed_plan(lambda value: value["scope"].update({"pair_id":"cam_1__cam_0"})) != baseline
    assert changed_plan(lambda value: value["scope"].update({"selected_frame":1})) != baseline
    assert changed_values(lambda value: value["signature_inputs"]["images"]["reference_image"].update({"digest":"ref-b"})) != baseline
    assert changed_values(lambda value: value["signature_inputs"]["images"]["secondary_reference_image"].update({"digest":"secondary-b"})) != baseline
    assert changed_values(lambda value: value["signature_inputs"]["images"]["current_image"].update({"digest":"current-b"})) != baseline
    assert changed_values(lambda value: value["signature_inputs"]["calibration"].update({"digest":"calibration-b"})) != baseline
    assert changed_plan(lambda value: value["upstream_dependencies"][0]["producer_signature"].update({"digest":"sha256:roi-other"})) != baseline
    assert changed_plan(lambda value: value["upstream_dependencies"][0]["required_artifacts"][0]["identity"].update({"digest":"roi-other"})) != baseline
    assert changed_values(lambda value: value["reconstruction"].update({"min_views":3})) != baseline
    assert _pair_signature(baseline_plan, baseline_values, "neurodic.pin_multi.pair_solve_quality/v2").digest != baseline
    assert changed_plan(lambda value: value["upstream_dependencies"][0].update({"source_trial_id":"source-b"})) == baseline
    assert changed_plan(lambda value: value["upstream_dependencies"][0].update({"source_attempt_id":"a2"})) == baseline
    assert changed_plan(lambda value: value["upstream_dependencies"][0].update({"managed_absolute_path":"/different/managed/root"})) == baseline
    assert changed_values(lambda value: value.update({"result_root":"/different/result"})) == baseline
    assert changed_values(lambda value: value.update({"visualization_root":"/different/visualization"})) == baseline
    assert changed_values(lambda value: value.update({"staging_root":"/different/staging"})) == baseline


def _published_manifest(root: Path, trial_id: str):
    path = root / "trials" / trial_id / "manifest.json"
    return path, json.loads(path.read_text())


def test_executor_revalidation_rejects_post_plan_dependency_tamper(tmp_path, monkeypatch):
    plan = json.loads(json.dumps(_plan(tmp_path, "c1-post-plan-content")))
    (tmp_path / "trials/source/artifacts/pin_multi.separate_pair_roi_call/a1/left_mask.npy").write_bytes(b"tampered")
    import neurodic.api.pin_multi_slover_dic as api
    monkeypatch.setattr(api, "solve_pin_multi_pair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("downstream scientific action must not execute")))
    with pytest.raises(ControlPlaneError): execute_trial(plan, managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call")


def test_executor_revalidation_rejects_post_plan_dependency_manifest_tamper(tmp_path, monkeypatch):
    plan = json.loads(json.dumps(_plan(tmp_path, "c1-post-plan-manifest"))); manifest = tmp_path / "trials/source/manifest.json"
    value = json.loads(manifest.read_text()); value["produced_artifacts"][0]["identity"]["digest"] = "tampered"; manifest.write_text(json.dumps(value))
    import neurodic.api.pin_multi_slover_dic as api
    monkeypatch.setattr(api, "solve_pin_multi_pair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("downstream scientific action must not execute")))
    with pytest.raises(ControlPlaneError): execute_trial(plan, managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call")


def _completed_pair_solve(tmp_path, monkeypatch, trial_id="c1-first"):
    _fake(monkeypatch)
    result = execute_trial(_plan(tmp_path, trial_id), managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"


def test_pair_solve_reuse_rejects_tampered_required_artifact(tmp_path, monkeypatch):
    _completed_pair_solve(tmp_path, monkeypatch); path, manifest = _published_manifest(tmp_path, "c1-first")
    artifact = next(item for item in manifest["produced_artifacts"] if item["artifact_type"] == "pin_multi_field.reference_disparity")
    (path.parent / artifact["location"]).write_bytes(b"tampered")
    _fake(monkeypatch); result = execute_trial(_plan(tmp_path, "c1-tampered-artifact"), managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"


def test_pair_solve_reuse_rejects_tampered_artifact_manifest_identity(tmp_path, monkeypatch):
    _completed_pair_solve(tmp_path, monkeypatch); path, manifest = _published_manifest(tmp_path, "c1-first")
    manifest["produced_artifacts"][0]["identity"]["digest"] = "tampered"; path.write_text(json.dumps(manifest))
    _fake(monkeypatch); result = execute_trial(_plan(tmp_path, "c1-tampered-manifest"), managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"


def test_pair_solve_reuse_rejects_tampered_producer_signature(tmp_path, monkeypatch):
    _completed_pair_solve(tmp_path, monkeypatch); path, manifest = _published_manifest(tmp_path, "c1-first")
    manifest["produced_artifacts"][0]["producer_signature"]["digest"] = "sha256:tampered"; path.write_text(json.dumps(manifest))
    _fake(monkeypatch); result = execute_trial(_plan(tmp_path, "c1-tampered-signature"), managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert result["stage_attempts"][0]["status"] == "completed"


def test_pair_solve_wrong_scope_is_never_reused(tmp_path, monkeypatch):
    _completed_pair_solve(tmp_path, monkeypatch); _path, manifest = _published_manifest(tmp_path, "c1-first")
    signature = dict(manifest["stage_attempts"][0]["producer_signature"]); signature["scope"] = {"pair_id":"cam_0__cam_2", "selected_frame":0}
    candidate = ProducerSignature(signature["stage_id"], signature["implementation"], signature["stage_config_identity"], signature["input_identities"], signature["scope"], signature["output_contract"])
    assert _verified_reuse(tmp_path, candidate) is None


def test_reusable_upstream_roi_does_not_imply_pair_solve_reuse(tmp_path, monkeypatch):
    calls = []
    _fake(monkeypatch)
    import neurodic.api.pin_multi_slover_dic as api
    original = api.solve_pin_multi_pair
    def counted(*args, **kwargs): calls.append(True); return original(*args, **kwargs)
    monkeypatch.setattr(api, "solve_pin_multi_pair", counted)
    result = execute_trial(_plan(tmp_path, "c1-no-downstream-reuse"), managed_root=tmp_path, action_id="pin_multi.pair_solve_quality_call").to_dict()["data"]["execution"]
    assert calls and result["stage_attempts"][0]["status"] == "completed"
