"""C3.1 guarded executor lifecycle tests; real fusion/native code is forbidden."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.adapters.execution_pin_multi import _solve_outputs
from neurodic.agent.artifacts import content_identity
from neurodic.agent.execution import execute_trial
from neurodic.agent.errors import ControlPlaneError
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.pair_set_readiness import inspect_pin_multi_pair_set_readiness
from neurodic.agent.trials import plan_trial


PAIRS = ["cam_1__cam_2", "cam_0__cam_1"]


def _write_c1_outputs(base: Path, pair: str, frame: int = 0) -> None:
    for item in _solve_outputs(pair):
        path = base / item.path; path.parent.mkdir(parents=True, exist_ok=True)
        if "/disp/" in item.path:
            np.savez(path, coordinates=np.ones((1,2)), displacement=np.ones((1,2)), iterations=np.array(1), final_loss=np.array(.1))
        elif "/reconstruct/" in item.path:
            np.savez(path, left_coordinates=np.ones((1,2)), right_coordinates=np.ones((1,2)), points=np.ones((1,3)), valid=np.array([True]), reprojection_error=np.ones(1))
        elif item.path.endswith("initial_to_current.npz"):
            np.savez(path, coordinates=np.ones((1,2)), reference_points=np.ones((1,3)), current_points=np.ones((1,3)), displacement=np.zeros((1,3)), valid=np.array([True]))
        elif item.path.endswith("reason_codes.npy"): np.save(path, np.array([0], np.int8))
        elif item.path.endswith("quality.json"): path.write_text(json.dumps({"total_points":1,"valid_points":1,"valid_ratio":1.0,"reason_codes":{"valid":1}}))
        elif item.path.endswith("pair_metadata.json"):
            left,right=pair.split("__"); path.write_text(json.dumps({"pair_id":pair,"reference_camera":left,"secondary_camera":right,"selected_frame":frame}))
        else: path.write_text("{}")


def _source(managed: Path, calibration: Path, pair: str) -> dict:
    trial_id="c1-"+pair.replace("__","-"); attempt="a1"; trial=managed/"trials"/trial_id
    base=trial/"artifacts"/"pin_multi.pair_solve_quality_call"/attempt; _write_c1_outputs(base,pair)
    signature={"stage_id":"pin_multi.pair_solve_quality_call","scope":{"pair_id":pair,"selected_frame":0},
      "digest":"sha256:"+hashlib.sha256(pair.encode()).hexdigest(),"output_contract":"neurodic.pin_multi.pair-solve-quality-artifacts/v1",
      "implementation":{"adapter":"neurodic.pin_multi.pair_solve_quality/v1"},"input_identities":{"calibration":content_identity(calibration).to_dict()}}
    records=[]
    for output in _solve_outputs(pair):
        path=base/output.path
        records.append({"artifact_type":output.artifact_type,"location":str(path.relative_to(trial)),"stage_attempt_id":attempt,
          "producer_action_id":"pin_multi.pair_solve_quality_call","producer_signature":signature,"identity":content_identity(path).to_dict()})
    trial.mkdir(parents=True,exist_ok=True)
    (trial/"manifest.json").write_text(json.dumps({"trial_id":trial_id,"stage_attempts":[{"stage_attempt_id":attempt,"status":"completed","action_id":"pin_multi.pair_solve_quality_call","producer_signature":signature}],"produced_artifacts":records}))
    wanted=[x for x in records if x["artifact_type"] in {"pin_multi_reconstruction.reference","pin_multi_reconstruction.current"}]
    return {"dependency_id":f"pair/{pair}","producer_action_id":"pin_multi.pair_solve_quality_call","producer_signature":signature,
      "scope":{"pair_id":pair,"selected_frame":0},"source_trial_id":trial_id,"source_attempt_id":attempt,
      "required_artifacts":[{"relative_path":x["location"],"identity":x["identity"]} for x in wanted]}


def _fixture(tmp_path: Path):
    case=tmp_path/"case"; managed=tmp_path/"managed"; managed.mkdir(); (managed/"trials").mkdir()
    for cam in ("cam_0","cam_1","cam_2"):
        d=case/"images"/cam; d.mkdir(parents=True)
        (d/"00.bmp").write_bytes(b"reference"); (d/"01.bmp").write_bytes(b"current")
    calibration=case/"calibration.json"; calibration.write_text(json.dumps({"cameras":[]}))
    for relative in ("result/pin_multi_slover/manifest.json", "result/pin_multi_slover/fused/legacy.json",
                     "result/pin_multi_slover/pair_roi/legacy_manifest.json", "result/pin_multi_slover/pairs/legacy/pair.json"):
        sentinel=case/relative; sentinel.parent.mkdir(parents=True,exist_ok=True); sentinel.write_text(relative)
    config=tmp_path/"pin_multi.yaml"; paths=tmp_path/"case_paths.yaml"
    config.write_text("solver: pin_multi_slover\nmode: pairwise_multiview\ncamera_pairs:\n  selection: manual\n  manual:\n    - [cam_1, cam_2]\n    - [cam_0, cam_1]\nfusion:\n  enabled: true\n  voxel_size: 0.1\nreconstruction:\n  world_scale: 1.0\ntraditional_strain:\n  neighbors: 12\n")
    paths.write_text(f"fake:\n  case:\n    root: {case}\n    images: images\n    calibration: calibration.json\n    frame: -1\n")
    deps=[_source(managed,calibration,pair) for pair in PAIRS]
    report=inspect_pin_multi_pair_set_readiness(config,managed_root=managed,selected_frame=0,case_key="fake",case_paths=paths).data
    assert report["status"]=="ready" and report["fusion_input_identity"]
    return {"case":case,"managed":managed,"config":config,"paths":paths,"deps":deps,"report":report}


def _plan(fx, trial="c3-first", *, deps=None, scope_change=None):
    report=fx["report"]; scope={"selected_frame":0,"planned_pair_ids":list(report["scope"]["planned_pair_ids"]),"pair_set_status":report["status"],
      "planned_pair_set_identity":report["planned_pair_set_identity"],"fusion_input_identity":report["fusion_input_identity"]}
    if scope_change: scope_change(scope)
    return plan_trial(fx["config"],case_key="fake",case_paths=fx["paths"],trial_id=trial,restore_missing=True,scope=scope,
                      upstream_dependencies=fx["deps"] if deps is None else deps).to_dict()["data"]["trial_plan"]


def write_valid_fusion_outputs(result_root: Path, pairs=PAIRS, *, n=2, corrupt=None):
    fused=result_root/"fused"; fused.mkdir(parents=True,exist_ok=True); names=np.asarray(pairs); source=np.arange(n)%len(pairs); points=np.arange(n*3,dtype=float).reshape(n,3); reproj=np.ones(n)
    common={"valid":np.ones(n,bool),"reprojection_error":reproj,"source_pair":source,"pair_names":names,"voxel_size":.1}
    np.savez(fused/"reference_surface.npz",points=points,**common); np.savez(fused/"current_surface.npz",points=points+1,**common)
    np.savez(fused/"deformation.npz",coordinates=points,reference_points=points,current_points=points+1,displacement=np.ones((n,3)),valid=np.ones(n,bool),source_pair=source,pair_names=names,voxel_size=.1)
    np.savez(fused/"strain.npz",coordinates=points,strain=np.ones((n,6)),valid=np.ones(n,bool),source_pair=source,pair_names=names,voxel_size=.1)
    (fused/"summary.json").write_text(json.dumps({"selected_points":n}))
    if corrupt:
        path=fused/corrupt
        if path.exists(): path.unlink()


def _install_fake(monkeypatch, *, n=2, error=None, omit=None, mutate=None, calls=None):
    import neurodic.pin_multi_fusion as fusion
    def fake(_values, *, ordered_pair_inputs, result_root, visualization_root):
        if calls is not None: calls.append([x["pair_id"] for x in ordered_pair_inputs])
        assert [x["pair_id"] for x in ordered_pair_inputs]==PAIRS
        assert all("/trials/c1-" in str(x["reference_reconstruction"]) for x in ordered_pair_inputs)
        assert "/staging/" in str(result_root) and "/staging/" in str(visualization_root)
        if error is KeyboardInterrupt: raise KeyboardInterrupt()
        if error: raise RuntimeError("fake fusion failure")
        write_valid_fusion_outputs(Path(result_root),n=n,corrupt=omit)
        if mutate: mutate(Path(result_root)/"fused")
        return {"selected_points":n}
    monkeypatch.setattr(fusion,"fuse_pin_multi_managed_pairs",fake)
    monkeypatch.setattr(fusion,"fuse_pin_multi_surfaces",lambda *_a,**_k: (_ for _ in ()).throw(AssertionError("legacy fusion called")))


def _execute(fx, monkeypatch, *, trial="c3-first", **fake):
    _install_fake(monkeypatch,**fake)
    return execute_trial(_plan(fx,trial),managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call").data["execution"]


def test_fusion_plan_ready_for_real_c2_ready_report(tmp_path):
    fx=_fixture(tmp_path); plan=_plan(fx); assert plan["plan_status"]=="ready" and [x["action_id"] for x in plan["execution_actions"]]==["pin_multi.fusion_postprocess_call"]


@pytest.mark.parametrize("change",[
    lambda s:s.update(pair_set_status="not_ready"), lambda s:s.update(fusion_input_identity=None),
    lambda s:s.update(planned_pair_ids=[PAIRS[0],PAIRS[0]]), lambda s:s.update(planned_pair_ids=[]),
])
def test_fusion_plan_invalid_pair_set_is_blocked(tmp_path,change):
    fx=_fixture(tmp_path); plan=_plan(fx,scope_change=change); assert plan["plan_status"]=="blocked" and plan["policy_violations"][-1]["code"]=="PAIR_SET.NOT_READY"


def test_fusion_postprocess_guarded_success(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); calls=[]; result=_execute(fx,monkeypatch,calls=calls)
    assert result["stage_attempts"][0]["status"]=="completed" and calls==[PAIRS]
    assert capability_for("pin_multi.fusion_postprocess_call").completion_scope=="combined_action"
    assert len(result["produced_artifacts"])==5


def test_fusion_adapter_restores_c2_planned_pair_order(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); fx["deps"].reverse(); calls=[]; result=_execute(fx,monkeypatch,calls=calls)
    assert result["stage_attempts"][0]["status"]=="completed" and calls==[PAIRS]


@pytest.mark.parametrize("kind",["missing","extra","wrong_mapping","wrong_frame","wrong_action","wrong_signature","legacy"])
def test_fusion_dependency_negative_matrix(tmp_path,monkeypatch,kind):
    fx=_fixture(tmp_path); deps=copy.deepcopy(fx["deps"])
    if kind=="missing": deps.pop()
    elif kind=="extra": deps.append({**copy.deepcopy(deps[0]),"dependency_id":"pair/cam_9__cam_8"})
    elif kind=="wrong_mapping": deps[0]["scope"]["pair_id"]=PAIRS[1]
    elif kind=="wrong_frame": deps[0]["scope"]["selected_frame"]=1
    elif kind=="wrong_action": deps[0]["producer_action_id"]="legacy.action"
    elif kind=="wrong_signature": deps[0]["producer_signature"]["digest"]="sha256:wrong"
    else: deps[0]={"dependency_id":deps[0]["dependency_id"],"source_trial_id":"legacy","source_attempt_id":"a","producer_action_id":"legacy","producer_signature":{},"scope":deps[0]["scope"],"required_artifacts":[]}
    plan=_plan(fx,deps=deps); calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    with pytest.raises(ControlPlaneError): execute_trial(plan,managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call")
    assert calls==[]


def test_fusion_rejects_duplicate_semantic_pair_dependency(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); deps=copy.deepcopy(fx["deps"]); deps[1]["scope"]=dict(deps[0]["scope"]); deps[1]["producer_signature"]=copy.deepcopy(deps[0]["producer_signature"])
    plan=_plan(fx,deps=deps); calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    with pytest.raises(ControlPlaneError): execute_trial(plan,managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call")
    assert calls==[]


@pytest.mark.parametrize("missing",["reference.npz","current.npz"])
def test_fusion_rejects_missing_reconstruction(tmp_path,monkeypatch,missing):
    fx=_fixture(tmp_path); deps=copy.deepcopy(fx["deps"]); deps[0]["required_artifacts"]=[x for x in deps[0]["required_artifacts"] if Path(x["relative_path"]).name!=missing]
    calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    result=execute_trial(_plan(fx,deps=deps),managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call").data["execution"]
    assert result["stage_attempts"][0]["status"]=="failed" and calls==[]


def test_fusion_rejects_reversed_pair_roles(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); plan=_plan(fx); dep=fx["deps"][0]; trial=fx["managed"]/"trials"/dep["source_trial_id"]; manifest=trial/"manifest.json"; value=json.loads(manifest.read_text())
    record=next(x for x in value["produced_artifacts"] if x["artifact_type"]=="pin_multi_pair_metadata"); path=trial/record["location"]
    metadata=json.loads(path.read_text()); metadata["reference_camera"],metadata["secondary_camera"]=metadata["secondary_camera"],metadata["reference_camera"]; path.write_text(json.dumps(metadata)); record["identity"]=content_identity(path).to_dict(); manifest.write_text(json.dumps(value))
    calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    with pytest.raises(ControlPlaneError): execute_trial(plan,managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call")
    assert calls==[]


@pytest.mark.parametrize("which",["reference","current","manifest"])
def test_fusion_executor_rejects_post_plan_tamper(tmp_path,monkeypatch,which):
    fx=_fixture(tmp_path); plan=_plan(fx); dep=fx["deps"][0]; trial=fx["managed"]/"trials"/dep["source_trial_id"]
    if which in {"reference","current"}:
        item=next(x for x in dep["required_artifacts"] if Path(x["relative_path"]).name==f"{which}.npz"); (trial/item["relative_path"]).write_bytes(b"tampered")
    else:
        path=trial/"manifest.json"; value=json.loads(path.read_text()); value["produced_artifacts"][0]["identity"]["digest"]="tampered"; path.write_text(json.dumps(value))
    calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    with pytest.raises(ControlPlaneError): execute_trial(plan,managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call")
    assert calls==[]


def test_fusion_executor_rejects_post_plan_fusion_identity_change(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); plan=_plan(fx); dep=fx["deps"][0]; trial=fx["managed"]/"trials"/dep["source_trial_id"]; manifest=trial/"manifest.json"; value=json.loads(manifest.read_text())
    record=next(x for x in value["produced_artifacts"] if x["artifact_type"]=="pin_multi_reconstruction.reference"); path=trial/record["location"]
    with np.load(path,allow_pickle=False) as data: arrays={key:np.asarray(data[key]) for key in data.files}
    arrays["points"]=arrays["points"]+1; np.savez(path,**arrays); record["identity"]=content_identity(path).to_dict(); manifest.write_text(json.dumps(value))
    changed=inspect_pin_multi_pair_set_readiness(fx["config"],managed_root=fx["managed"],selected_frame=0,case_key="fake",case_paths=fx["paths"]).data
    assert changed["status"]=="ready" and changed["fusion_input_identity"]!=fx["report"]["fusion_input_identity"]
    calls=[]; _install_fake(monkeypatch,error=AssertionError,calls=calls)
    with pytest.raises(ControlPlaneError): execute_trial(plan,managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call")
    assert calls==[]


@pytest.mark.parametrize("omit",["reference_surface.npz","current_surface.npz","deformation.npz","strain.npz","summary.json"])
def test_fusion_rejects_each_missing_required_output(tmp_path,monkeypatch,omit):
    fx=_fixture(tmp_path); result=_execute(fx,monkeypatch,omit=omit); assert result["stage_attempts"][0]["status"]=="failed" and not result["produced_artifacts"]


def _bad_npz(fused: Path,mode: str):
    if mode=="corrupt": (fused/"reference_surface.npz").write_bytes(b"bad")
    elif mode=="key": np.savez(fused/"reference_surface.npz",points=np.ones((2,3)))
    elif mode=="json": (fused/"summary.json").write_text("{")
    elif mode=="summary": (fused/"summary.json").write_text("{}")
    elif mode in {"nan","inf"}:
        bad=np.nan if mode=="nan" else np.inf; np.savez(fused/"reference_surface.npz",points=np.full((2,3),bad),valid=np.ones(2,bool),reprojection_error=np.ones(2),source_pair=np.array([0,1]),pair_names=np.asarray(PAIRS),voxel_size=.1)
    elif mode=="shape": np.savez(fused/"current_surface.npz",points=np.ones((1,3)),valid=np.ones(1,bool),reprojection_error=np.ones(1),source_pair=np.array([0]),pair_names=np.asarray(PAIRS),voxel_size=.1)
    elif mode in {"source_hi","source_neg"}:
        source=np.array([2,1]) if mode=="source_hi" else np.array([-1,1]); np.savez(fused/"deformation.npz",coordinates=np.ones((2,3)),reference_points=np.ones((2,3)),current_points=np.ones((2,3)),displacement=np.ones((2,3)),valid=np.ones(2,bool),source_pair=source,pair_names=np.asarray(PAIRS),voxel_size=.1)
    elif mode in {"order","unknown"}:
        names=np.asarray(list(reversed(PAIRS)) if mode=="order" else [PAIRS[0],"cam_x__cam_y"]); np.savez(fused/"strain.npz",coordinates=np.ones((2,3)),strain=np.ones((2,6)),valid=np.ones(2,bool),source_pair=np.array([0,1]),pair_names=names,voxel_size=.1)
    else:
        data=np.load(fused/"strain.npz"); np.savez(fused/"strain.npz",**{k:data[k] for k in data.files if k!="voxel_size"},voxel_size=.2)


@pytest.mark.parametrize("mode",["corrupt","key","json","summary","nan","inf","shape","source_hi","source_neg","order","unknown","voxel"])
def test_fusion_output_validation_negative_matrix(tmp_path,monkeypatch,mode):
    fx=_fixture(tmp_path); result=_execute(fx,monkeypatch,mutate=lambda fused:_bad_npz(fused,mode)); assert result["stage_attempts"][0]["status"]=="failed" and not result["produced_artifacts"]


def test_fusion_zero_points_is_execution_success(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); result=_execute(fx,monkeypatch,n=0); assert result["stage_attempts"][0]["status"]=="completed"


def test_fusion_high_error_but_structurally_valid_completes(tmp_path,monkeypatch):
    fx=_fixture(tmp_path)
    def high(fused):
        for name,key in (("reference_surface.npz","points"),("current_surface.npz","points")):
            path=fused/name
            with np.load(path,allow_pickle=False) as data: values={item:np.asarray(data[item]) for item in data.files}
            values[key]=np.full((2,3),1e30); values["reprojection_error"]=np.full(2,1e30); np.savez(path,**values)
    result=_execute(fx,monkeypatch,mutate=high); assert result["stage_attempts"][0]["status"]=="completed"


@pytest.mark.parametrize(("error","status"),[(RuntimeError,"failed"),(KeyboardInterrupt,"interrupted")])
def test_fusion_failure_interrupt_no_publish(tmp_path,monkeypatch,error,status):
    fx=_fixture(tmp_path); result=_execute(fx,monkeypatch,error=error); assert result["stage_attempts"][0]["status"]==status and not result["produced_artifacts"]


def test_fusion_partial_staging_then_error_does_not_publish(tmp_path,monkeypatch):
    fx=_fixture(tmp_path)
    import neurodic.pin_multi_fusion as fusion
    def partial(_values,*,ordered_pair_inputs,result_root,visualization_root):
        path=Path(result_root)/"fused/reference_surface.npz"; path.parent.mkdir(parents=True); path.write_bytes(b"partial"); raise RuntimeError("after partial staging")
    monkeypatch.setattr(fusion,"fuse_pin_multi_managed_pairs",partial)
    result=execute_trial(_plan(fx),managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call").data["execution"]
    assert result["stage_attempts"][0]["status"]=="failed" and not result["produced_artifacts"] and not list((fx["managed"]/"trials/c3-first/artifacts").rglob("*.npz"))


def test_fusion_guarded_success_atomic_publish(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); result=_execute(fx,monkeypatch)
    assert len(result["produced_artifacts"])==5
    assert all((fx["managed"]/"trials/c3-first"/x["location"]).is_file() for x in result["produced_artifacts"])


def test_fusion_guarded_success_and_safe_reuse(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); first=_execute(fx,monkeypatch); assert first["stage_attempts"][0]["status"]=="completed"
    import neurodic.pin_multi_fusion as fusion
    monkeypatch.setattr(fusion,"fuse_pin_multi_managed_pairs",lambda *_a,**_k: (_ for _ in ()).throw(AssertionError("fusion callable must not execute during valid reuse")))
    second=execute_trial(_plan(fx,"c3-second"),managed_root=fx["managed"],action_id="pin_multi.fusion_postprocess_call").data["execution"]
    assert second["stage_attempts"][0]["status"]=="reused"


@pytest.mark.parametrize("tamper",["artifact","identity","producer"])
def test_fusion_reuse_rejects_downstream_tamper(tmp_path,monkeypatch,tamper):
    fx=_fixture(tmp_path); first=_execute(fx,monkeypatch); trial=fx["managed"]/"trials/c3-first"; manifest=trial/"manifest.json"; value=json.loads(manifest.read_text())
    if tamper=="artifact": (trial/value["produced_artifacts"][0]["location"]).write_bytes(b"tampered")
    elif tamper=="identity": value["produced_artifacts"][0]["identity"]["digest"]="tampered"; manifest.write_text(json.dumps(value))
    else: value["produced_artifacts"][0]["producer_signature"]["digest"]="tampered"; manifest.write_text(json.dumps(value))
    result=_execute(fx,monkeypatch,trial="c3-second"); assert result["stage_attempts"][0]["status"]=="completed"


def test_fusion_reuse_rejects_wrong_pair_order_or_scope(tmp_path,monkeypatch):
    fx=_fixture(tmp_path); _execute(fx,monkeypatch)
    with pytest.raises(ControlPlaneError): _plan(fx,"c3-second",scope_change=lambda s:s.update(selected_frame=1))


def _snapshot(root: Path):
    return {str(p.relative_to(root)):(p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("fails",[False,True])
def test_fusion_baseline_and_source_manifests_zero_write(tmp_path,monkeypatch,fails):
    fx=_fixture(tmp_path); before=_snapshot(fx["case"]); sources={d["source_trial_id"]:(fx["managed"]/"trials"/d["source_trial_id"] / "manifest.json").read_bytes() for d in fx["deps"]}
    _execute(fx,monkeypatch,error=RuntimeError if fails else None)
    assert _snapshot(fx["case"])==before and all((fx["managed"]/"trials"/name/"manifest.json").read_bytes()==data for name,data in sources.items())
