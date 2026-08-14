"""Direct C3 producer-signature determinant matrix; native-free."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
from neurodic.agent.adapters.execution_pin_multi import guarded_fusion_postprocess_action
from neurodic.agent.execution import TrustedAction, _stage_signature


def _case(tmp_path):
    root=tmp_path/"case"; root.mkdir(); calibration=root/"cal.json"; calibration.write_bytes(b"calibration")
    values={"case":{"root":str(root),"calibration":"cal.json"},"fusion":{"enabled":True,"voxel_size":.1},"traditional_strain":{"neighbors":12}}
    signature={"stage_id":"pin_multi.pair_solve_quality_call","scope":{"pair_id":"a__b","selected_frame":0},"digest":"sha256:c1"}
    deps=[]
    for pair in ("b__c","a__b"):
        sig=copy.deepcopy(signature); sig["scope"]["pair_id"]=pair
        deps.append({"dependency_id":f"pair/{pair}","producer_action_id":"pin_multi.pair_solve_quality_call","producer_signature":sig,"scope":{"pair_id":pair,"selected_frame":0},"source_trial_id":"source","source_attempt_id":"a1","required_artifacts":[{"relative_path":f"artifacts/{pair}/reference.npz","identity":{"digest":f"ref-{pair}"}},{"relative_path":f"artifacts/{pair}/current.npz","identity":{"digest":f"cur-{pair}"}}]})
    plan={"solver":"pin_multi","scope":{"selected_frame":0,"planned_pair_ids":["b__c","a__b"],"pair_set_status":"ready","planned_pair_set_identity":"sha256:planned","fusion_input_identity":"sha256:fusion"},"baseline":{"effective_config_identity":"sha256:base"},"upstream_dependencies":deps}
    return plan,values,calibration


def _sig(plan,values,action=None): return _stage_signature(plan,values,action or guarded_fusion_postprocess_action(),("pin_multi.fusion","pin_multi.postprocess")).digest


@pytest.mark.parametrize("change",["order","membership","frame","reference","current","c1","fusion_config","calibration","planned_identity","fusion_identity","implementation"])
def test_fusion_producer_signature_scientific_changes_differ(tmp_path,change):
    plan,values,calibration=_case(tmp_path); baseline=_sig(plan,values); p=copy.deepcopy(plan); v=copy.deepcopy(values); action=None
    if change=="order": p["scope"]["planned_pair_ids"].reverse(); p["upstream_dependencies"].reverse()
    elif change=="membership": p["scope"]["planned_pair_ids"][0]="x__y"
    elif change=="frame": p["scope"]["selected_frame"]=1
    elif change=="reference": p["upstream_dependencies"][0]["required_artifacts"][0]["identity"]["digest"]="changed"
    elif change=="current": p["upstream_dependencies"][0]["required_artifacts"][1]["identity"]["digest"]="changed"
    elif change=="c1": p["upstream_dependencies"][0]["producer_signature"]["digest"]="changed"
    elif change=="fusion_config": v["fusion"]["voxel_size"]=.2
    elif change=="calibration": calibration.write_bytes(b"changed calibration")
    elif change=="planned_identity": p["scope"]["planned_pair_set_identity"]="changed"
    elif change=="fusion_identity": p["scope"]["fusion_input_identity"]="changed"
    else:
        base=guarded_fusion_postprocess_action(); action=TrustedAction(base.action_id,base.run,"neurodic.pin_multi.fusion_postprocess/v2",base.output_contract,base.input_identities)
    assert _sig(p,v,action)!=baseline


@pytest.mark.parametrize("change",["none","source_trial","source_attempt","managed_path","result_root","visualization_root","staging_root"])
def test_fusion_producer_signature_management_changes_same(tmp_path,change):
    plan,values,_calibration=_case(tmp_path); baseline=_sig(plan,values); p=copy.deepcopy(plan); v=copy.deepcopy(values)
    if change=="source_trial": p["upstream_dependencies"][0]["source_trial_id"]="elsewhere"
    elif change=="source_attempt": p["upstream_dependencies"][0]["source_attempt_id"]="elsewhere"
    elif change=="managed_path": p["upstream_dependencies"][0]["managed_absolute_path"]="/different"
    elif change!="none": v[change]="/different"
    assert _sig(p,v)==baseline


def test_fusion_producer_signature_quality_only_same(tmp_path):
    plan,values,_calibration=_case(tmp_path); baseline=_sig(plan,values); changed=copy.deepcopy(plan)
    changed["upstream_dependencies"][0]["quality_evidence"]={"identity":{"digest":"quality-only-change"},"valid_ratio":.01}
    assert _sig(changed,values)==baseline
