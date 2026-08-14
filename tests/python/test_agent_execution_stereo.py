"""Native-free fake lifecycle tests for the atomic guarded Stereo adapter."""
from __future__ import annotations
import json, sys, types
from pathlib import Path
import numpy as np
import pytest
from neurodic.agent.execution import execute_trial
from neurodic.agent.execution_registry import capability_for
from neurodic.agent.trials import plan_trial

ROOT=Path(__file__).resolve().parents[2]
FIELDS=("reference_disparity","left_temporal","deformed_disparity")
def fixture(tmp):
    case=tmp/"case"; (case/"left").mkdir(parents=True,exist_ok=True); (case/"right").mkdir(exist_ok=True)
    for folder,names in (("left",("00.bmp","01.bmp")),("right",("00.bmp","01.bmp"))):
        for name in names: (case/folder/name).write_bytes(name.encode())
    (case/"roi.bmp").write_bytes(b"roi"); (case/"camera.json").write_text("{}")
    paths=tmp/"paths.yaml"; paths.write_text(f"pin_stereo:\n  case:\n    root: {case}\n    left_images: left\n    right_images: right\n    roi: roi.bmp\n    camera_pair: camera.json\n  output:\n    result: result/stereo\n    visualization: visualization/stereo\n")
    return paths
def plan(tid, tmp, scope={"selected_frame":0}): return plan_trial(ROOT/"config/pin_stereo.yaml",case_key="pin_stereo",case_paths=fixture(tmp),trial_id=tid,restore_missing=True,scope=scope).to_dict()["data"]["trial_plan"]
def fake(monkeypatch, *, missing=None, interrupt=False, wrong_field=False, missing_geometry=False):
    calls=[]; module=types.ModuleType("neurodic.api.pin_stereo_dic")
    def run(config):
        calls.append(config); root=Path(config["output"]["result"]); vis=Path(config["output"]["visualization"])
        if interrupt: raise KeyboardInterrupt()
        for p in (root/"disp",root/"reconstruct",root/"deformation",root/"diagnostics",vis): p.mkdir(parents=True,exist_ok=True)
        for field in FIELDS:
            if field==missing: continue
            np.savez(root/f"disp/{field}.npz",coordinates=np.ones((2,2)),displacement=np.ones((2,2)),iterations=np.array(1),final_loss=np.array(.1),training_history=np.ones((1,3)),training_history_columns=np.array(["a"]),training_history_schema_version=np.array("neurodic.pin.training/v1"))
        for name in ("initial","last"): np.savez(root/f"reconstruct/{name}.npz",left_coordinates=np.ones((2,2)),right_coordinates=np.ones((2,2)),points=np.ones((2,3)),valid=np.array([True,True]),reprojection_error=np.ones(2))
        np.savez(root/"deformation/initial_to_last.npz",coordinates=np.ones((2,2)),reference_points=np.ones((2,3)),current_points=np.ones((2,3)),displacement=np.ones((2,3)),strain=np.ones((2,6)),strain_components=np.array(["x"]),valid=np.array([True,True]))
        (root/"deformation/initial_to_last_summary.json").write_text("{}")
        if not missing_geometry:
            np.savez(root/"diagnostics/stereo_geometry.npz",schema_version=np.array("neurodic.stereo_geometry/v1"),reason_code=np.zeros(2),reason_names=np.array(["ok"]),valid=np.array([True,True]),reference_reprojection_error=np.ones(2),current_reprojection_error=np.ones(2),reference_positive_depth=np.array([True,True]),current_positive_depth=np.array([True,True]))
            (root/"diagnostics/stereo_geometry.json").write_text("{}")
        if wrong_field: (root/"diagnostics/field_provenance.json").write_text('{"schema_version":"neurodic.stereo.fields/v1","fields":{"reference_disparity":{"reference":"reference_left","target":"deformed_left"}}}')
    module.pin_stereo_dic=run; monkeypatch.setitem(sys.modules,"neurodic.api.pin_stereo_dic",module); return calls

def test_capability_and_scope(tmp_path):
    c=capability_for("pin_stereo.combined_solver_call"); assert c.execution_supported and c.scope_requirement=="scope.selected_frame" and c.completion_scope=="combined_action"
    assert plan("stereo-missing",tmp_path,{})["plan_status"]=="blocked"
def test_fake_atomic_publish_and_reuse(tmp_path,monkeypatch):
    calls=fake(monkeypatch); first=execute_trial(plan("stereo-first",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]
    assert first["execution_status"]=="completed" and len(calls)==1 and all(a["producer_action_id"]=="pin_stereo.combined_solver_call" for a in first["produced_artifacts"])
    second=execute_trial(plan("stereo-reuse",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]
    assert len(calls)==1 and second["stage_attempts"][0]["status"]=="reused"
def test_missing_field_fails_without_publish(tmp_path,monkeypatch):
    fake(monkeypatch,missing="left_temporal"); result=execute_trial(plan("stereo-missing-field",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]
    assert result["execution_status"]=="failed" and not list((tmp_path/"trials/stereo-missing-field/artifacts").rglob("*"))
def test_wrong_field_or_missing_geometry_fails(tmp_path,monkeypatch):
    fake(monkeypatch,wrong_field=True); assert execute_trial(plan("stereo-wrong",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]["execution_status"]=="failed"
    fake(monkeypatch,missing_geometry=True); assert execute_trial(plan("stereo-geometry",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]["execution_status"]=="failed"
def test_interrupted_is_not_published(tmp_path,monkeypatch):
    fake(monkeypatch,interrupt=True); result=execute_trial(plan("stereo-interrupted",tmp_path),managed_root=tmp_path).to_dict()["data"]["execution"]
    assert result["execution_status"]=="interrupted" and not list((tmp_path/"trials/stereo-interrupted/artifacts").rglob("*"))
