"""Native-free C2 readiness coverage; fixtures are managed files only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurodic.agent.adapters.execution_pin_multi import _solve_outputs
from neurodic.agent.artifacts import content_identity
from neurodic.agent.pair_set_readiness import inspect_pin_multi_pair_set_readiness


def _values(root):
    return {"case": {"root": str(root), "calibration": "calibration.json"},
            "camera_pairs": {"selection": "manual", "manual": [["cam_0", "cam_1"]]}, "fusion": {"enabled": True}}


def _install(monkeypatch, root, pairs=(("cam_0", "cam_1", {}),), *, fusion=True):
    import neurodic.agent.pair_set_readiness as c2
    values = _values(root); values["fusion"]["enabled"] = fusion
    (root / "calibration.json").write_text(json.dumps({"cameras": []}))
    monkeypatch.setattr(c2, "resolve_config", lambda *a, **k: {"effective_config": values})
    monkeypatch.setattr(c2, "_options_from_config", lambda _v: (type("O", (), {"mode": "manual"})(), None))
    monkeypatch.setattr(c2, "select_pin_multi_pairs", lambda *_a: list(pairs))


def _publish(root, *, pair="cam_0__cam_1", frame=0, status="completed", mutate=None, trial="source"):
    attempt = "a1"; base = root / "trials" / trial / "artifacts" / "pin_multi.pair_solve_quality_call" / attempt
    signature = {"stage_id": "pin_multi.pair_solve_quality_call", "scope": {"pair_id": pair, "selected_frame": frame},
                 "digest": "sha256:fixed", "output_contract": "neurodic.pin_multi.pair-solve-quality-artifacts/v1",
                 "implementation": {"adapter": "neurodic.pin_multi.pair_solve_quality/v1"},
                 "input_identities": {"calibration": content_identity(root / "calibration.json").to_dict()}}
    records = []
    for output in _solve_outputs(pair):
        path = base / output.path; path.parent.mkdir(parents=True, exist_ok=True)
        if "/disp/" in output.path:
            np.savez(path, coordinates=np.ones((1, 2)), displacement=np.ones((1, 2)), iterations=np.array(1), final_loss=np.array(.1))
        elif "/reconstruct/" in output.path:
            np.savez(path, left_coordinates=np.ones((1,2)), right_coordinates=np.ones((1,2)), points=np.ones((1,3)), valid=np.array([False]), reprojection_error=np.ones(1))
        elif output.path.endswith("initial_to_current.npz"):
            np.savez(path, coordinates=np.ones((1,2)), reference_points=np.ones((1,3)), current_points=np.ones((1,3)), displacement=np.ones((1,3)), valid=np.array([False]))
        elif output.path.endswith("reason_codes.npy"): np.save(path, np.array([1], dtype=np.int8))
        elif output.path.endswith("quality.json"):
            path.write_text(json.dumps({"total_points": 1, "valid_points": 0, "valid_ratio": 0.0, "reason_codes": {"valid": 0}}))
        elif output.path.endswith("pair_metadata.json"):
            left, right = pair.split("__"); path.write_text(json.dumps({"pair_id": pair, "reference_camera": left, "secondary_camera": right, "selected_frame": frame}))
        else: path.write_text("{}")
        records.append({"artifact_type": output.artifact_type, "location": str(path.relative_to(base.parents[2])), "stage_attempt_id": attempt,
                        "producer_action_id": "pin_multi.pair_solve_quality_call", "producer_signature": signature, "identity": content_identity(path).to_dict()})
    if mutate: mutate(base, records, signature)
    manifest = {"trial_id": trial, "stage_attempts": [{"stage_attempt_id": attempt, "status": status, "action_id": "pin_multi.pair_solve_quality_call", "producer_signature": signature}], "produced_artifacts": records}
    path = base.parents[2] / "manifest.json"; path.write_text(json.dumps(manifest)); return path


def _report(monkeypatch, tmp_path, **kwargs):
    _install(monkeypatch, tmp_path, **kwargs.pop("install", {}))
    return inspect_pin_multi_pair_set_readiness("ignored.yaml", managed_root=tmp_path, selected_frame=0).to_dict()["data"]


def test_complete_reused_and_zero_valid_are_ready(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path); _publish(tmp_path, status="reused")
    report = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert report["status"] == "ready" and report["pairs"][0]["state"] == "reused" and report["fusion_input_identity"]


@pytest.mark.parametrize(("status", "state"), [("failed", "failed"), ("interrupted", "interrupted")])
def test_failed_and_interrupted_block(monkeypatch, tmp_path, status, state):
    _install(monkeypatch, tmp_path); _publish(tmp_path, status=status)
    report = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert report["status"] == "not_ready" and report["pairs"][0]["state"] == state


def test_missing_duplicate_and_disabled(monkeypatch, tmp_path):
    report = _report(monkeypatch, tmp_path); assert report["pairs"][0]["state"] == "missing"
    _install(monkeypatch, tmp_path, pairs=(("cam_0", "cam_1", {}), ("cam_0", "cam_1", {}))); _publish(tmp_path)
    assert "PAIR_SET.DUPLICATE_PAIR" in [x["code"] for x in inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data["blocking_reasons"]]
    report = _report(monkeypatch, tmp_path, install={"fusion": False}); assert report["status"] == "not_applicable"


@pytest.mark.parametrize("kind", ["tamper", "roles", "frame", "signature"])
def test_invalid_managed_candidates_block(monkeypatch, tmp_path, kind):
    _install(monkeypatch, tmp_path)
    def mutate(base, records, signature):
        if kind == "tamper": (base / _solve_outputs("cam_0__cam_1")[0].path).write_bytes(b"tampered")
        elif kind == "roles": (base / "scientific/pairs/cam_0__cam_1/pair_metadata.json").write_text(json.dumps({"pair_id":"cam_0__cam_1","reference_camera":"cam_1","secondary_camera":"cam_0","selected_frame":0}))
        elif kind == "frame": signature["scope"]["selected_frame"] = 1
        else: signature["implementation"]["adapter"] = "neurodic.pin_multi.pair_solve_quality/unreviewed"
    _publish(tmp_path, mutate=mutate)
    report = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert report["status"] == "not_ready"


def test_identity_order_and_management_location(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path); _publish(tmp_path)
    first = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    _install(monkeypatch, tmp_path, pairs=(("cam_1", "cam_0", {}), ("cam_0", "cam_1", {})))
    second = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert first["planned_pair_set_identity"] != second["planned_pair_set_identity"]


def test_quality_is_evidence_but_fusion_inputs_change_identity(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path); manifest = _publish(tmp_path)
    first = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    value = json.loads(manifest.read_text()); trial = manifest.parent
    quality = next(x for x in value["produced_artifacts"] if x["artifact_type"] == "pin_multi_pair_quality")
    qpath = trial / quality["location"]; q = json.loads(qpath.read_text()); q["valid_ratio"] = 0.01; qpath.write_text(json.dumps(q)); quality["identity"] = content_identity(qpath).to_dict(); manifest.write_text(json.dumps(value))
    quality_changed = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert quality_changed["status"] == "ready" and quality_changed["fusion_input_identity"] == first["fusion_input_identity"]
    value = json.loads(manifest.read_text()); recon = next(x for x in value["produced_artifacts"] if x["artifact_type"] == "pin_multi_reconstruction.current")
    rpath = trial / recon["location"]
    with np.load(rpath, allow_pickle=False) as before:
        np.savez(rpath, **{key: np.asarray(before[key]) * (2 if key == "points" else 1) for key in before.files})
    recon["identity"] = content_identity(rpath).to_dict(); manifest.write_text(json.dumps(value))
    changed = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert changed["status"] == "ready" and changed["fusion_input_identity"] != first["fusion_input_identity"]


def test_planned_identity_tracks_config_and_calibration(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path); _publish(tmp_path)
    first = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    import neurodic.agent.pair_set_readiness as c2
    values = _values(tmp_path); values["camera_pairs"]["selection"] = "changed"
    monkeypatch.setattr(c2, "resolve_config", lambda *a, **k: {"effective_config": values})
    changed = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert changed["planned_pair_set_identity"] != first["planned_pair_set_identity"]
    (tmp_path / "calibration.json").write_text(json.dumps({"cameras": ["changed"]}))
    recalibrated = inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0).data
    assert recalibrated["planned_pair_set_identity"] != changed["planned_pair_set_identity"] and recalibrated["status"] == "not_ready"


def test_readiness_is_zero_write(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path); _publish(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inspect_pin_multi_pair_set_readiness("x", managed_root=tmp_path, selected_frame=0)
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
