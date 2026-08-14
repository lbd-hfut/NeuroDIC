"""Hostile provenance coverage for managed upstream dependencies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neurodic.agent.artifacts import content_identity
from neurodic.agent.execution import _resolve_dependencies, execute_trial
from neurodic.agent.errors import ControlPlaneError
from neurodic.agent.schemas import canonical_json
from neurodic.agent.trials import plan_trial


ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path: Path):
    root = tmp_path / "managed"; location = "artifacts/pin_multi.pair_roi/a1/left_mask.npy"
    path = root / "trials" / "source" / location; path.parent.mkdir(parents=True); path.write_bytes(b"roi")
    signature = {"stage_id": "pin_multi.separate_pair_roi_call", "scope": {"pair_id": "cam_0__cam_1"}, "digest": "sha256:test"}
    artifact = {"location": location, "stage_attempt_id": "a1", "producer_action_id": "pin_multi.separate_pair_roi_call", "producer_signature": signature, "identity": content_identity(path).to_dict()}
    manifest = {"trial_id": "source", "stage_attempts": [{"stage_attempt_id": "a1", "status": "completed", "action_id": "pin_multi.separate_pair_roi_call", "producer_signature": signature}], "produced_artifacts": [artifact]}
    (root / "trials" / "source" / "manifest.json").write_text(json.dumps(manifest))
    dep = {"dependency_id": "pair_roi", "producer_action_id": "pin_multi.separate_pair_roi_call", "producer_signature": signature,
           "scope": {"pair_id": "cam_0__cam_1"}, "source_trial_id": "source", "source_attempt_id": "a1",
           "required_artifacts": [{"relative_path": location, "identity": artifact["identity"]}]}
    return root, path, dep


def _plan(dep):
    return {"scope": {"pair_id": "cam_0__cam_1"}, "upstream_dependencies": [dep]}


def _manifest(root: Path):
    path = root / "trials" / "source" / "manifest.json"
    value = json.loads(path.read_text())
    return path, value


def _assert_rejected(root: Path, dep):
    with pytest.raises(ControlPlaneError):
        _resolve_dependencies(_plan(dep), root)


def test_exact_dependency_resolves_and_tamper_fails(tmp_path):
    root, path, dep = _fixture(tmp_path)
    assert Path(_resolve_dependencies(_plan(dep), root)["pair_roi"]["files"]["left_mask.npy"]).samefile(path)
    path.write_bytes(b"tampered")
    _assert_rejected(root, dep)


def test_dependency_rejects_external_directory(tmp_path):
    root, _path, dep = _fixture(tmp_path); external = tmp_path / "external"; external.mkdir(); (external / "left_mask.npy").write_bytes(b"roi")
    bad = copy.deepcopy(dep); bad["source_trial_id"] = "../external"
    _assert_rejected(root, bad)


def test_dependency_rejects_legacy_artifact_without_provenance(tmp_path):
    root, _path, dep = _fixture(tmp_path); legacy = root / "trials" / "legacy"; legacy.mkdir(parents=True)
    (legacy / "left_mask.npy").write_bytes(b"roi"); (legacy / "meta.json").write_text("{}")
    (legacy / "manifest.json").write_text(json.dumps({"trial_id": "legacy", "produced_artifacts": []}))
    bad = copy.deepcopy(dep); bad["source_trial_id"] = "legacy"
    _assert_rejected(root, bad)


def test_dependency_rejects_symlink_escape(tmp_path):
    root, path, dep = _fixture(tmp_path); outside = tmp_path / "outside-mask.npy"; outside.write_bytes(b"roi")
    path.unlink(); path.symlink_to(outside)
    _assert_rejected(root, dep)


def test_dependency_rejects_required_artifact_traversal(tmp_path):
    root, _path, dep = _fixture(tmp_path); bad = copy.deepcopy(dep); bad["required_artifacts"][0]["relative_path"] = "../outside/left_mask.npy"
    _assert_rejected(root, bad)


def test_dependency_rejects_source_trial_mismatch(tmp_path):
    root, _path, dep = _fixture(tmp_path); source = root / "trials" / "source"; target = root / "trials" / "declared"; source.rename(target)
    manifest = json.loads((target / "manifest.json").read_text()); manifest["trial_id"] = "source"; (target / "manifest.json").write_text(json.dumps(manifest))
    bad = copy.deepcopy(dep); bad["source_trial_id"] = "declared"
    _assert_rejected(root, bad)


def test_dependency_rejects_source_attempt_mismatch(tmp_path):
    root, _path, dep = _fixture(tmp_path); manifest_path, manifest = _manifest(root)
    manifest["stage_attempts"][0]["stage_attempt_id"] = "a2"; manifest["produced_artifacts"][0]["stage_attempt_id"] = "a2"; manifest_path.write_text(json.dumps(manifest))
    _assert_rejected(root, dep)


def test_dependency_rejects_producer_action_mismatch(tmp_path):
    root, _path, dep = _fixture(tmp_path); bad = copy.deepcopy(dep); bad["producer_action_id"] = "other.action"
    _assert_rejected(root, bad)


def test_dependency_rejects_producer_signature_mismatch(tmp_path):
    root, _path, dep = _fixture(tmp_path); bad = copy.deepcopy(dep); bad["producer_signature"]["digest"] = "sha256:other"
    _assert_rejected(root, bad)


def test_dependency_rejects_scope_mismatch(tmp_path):
    root, _path, dep = _fixture(tmp_path); bad = copy.deepcopy(dep); bad["scope"] = {"pair_id": "cam_0__cam_2"}; bad["producer_signature"]["scope"] = dict(bad["scope"])
    _assert_rejected(root, bad)


def test_dependency_accepts_legacy_roi_scope_projection(tmp_path):
    root, path, dep = _fixture(tmp_path); dep["producer_signature"]["scope"]["selected_frame"] = -1; dep["producer_signature"]["stage_id"] = "pin_multi.pair_roi"
    manifest_path, manifest = _manifest(root)
    manifest["stage_attempts"][0]["producer_signature"]["scope"]["selected_frame"] = -1; manifest["stage_attempts"][0]["producer_signature"]["stage_id"] = "pin_multi.pair_roi"
    manifest["produced_artifacts"][0]["producer_signature"]["scope"]["selected_frame"] = -1; manifest["produced_artifacts"][0]["producer_signature"]["stage_id"] = "pin_multi.pair_roi"; manifest["produced_artifacts"][0].pop("producer_action_id")
    manifest_path.write_text(json.dumps(manifest))
    assert Path(_resolve_dependencies(_plan(dep), root)["pair_roi"]["files"]["left_mask.npy"]).samefile(path)


def test_dependency_rejects_missing_required_artifact(tmp_path):
    root, path, dep = _fixture(tmp_path); path.unlink()
    _assert_rejected(root, dep)


def test_dependency_rejects_manifest_artifact_identity_tamper(tmp_path):
    root, _path, dep = _fixture(tmp_path); manifest_path, manifest = _manifest(root); manifest["produced_artifacts"][0]["identity"]["digest"] = "not-the-file"; manifest_path.write_text(json.dumps(manifest))
    _assert_rejected(root, dep)


def test_dependency_rejects_producer_manifest_identity_tamper(tmp_path):
    root, _path, dep = _fixture(tmp_path); manifest_path, manifest = _manifest(root); manifest["produced_artifacts"][0]["producer_action_id"] = "other.action"; manifest_path.write_text(json.dumps(manifest))
    _assert_rejected(root, dep)


def _c1_plan(dep, trial_id: str):
    return plan_trial(ROOT / "config/pin_multi.yaml", case_key="pin_multi", case_paths=ROOT / "config/case_paths.yaml", trial_id=trial_id,
                      restore_missing=True, scope={"pair_id": "cam_0__cam_1", "selected_frame": 0}, upstream_dependencies=[dep]).to_dict()["data"]["trial_plan"]


@pytest.mark.parametrize("mutate", [
    lambda root, dep: dep.update({"source_trial_id": "../external"}),
    lambda root, dep: dep["required_artifacts"][0].update({"relative_path": "../outside/left_mask.npy"}),
    lambda root, dep: (root / "trials" / "source" / dep["required_artifacts"][0]["relative_path"]).write_bytes(b"tampered"),
    lambda root, dep: dep["producer_signature"].update({"digest": "sha256:wrong"}),
])
def test_dependency_negative_execution_never_invokes_adapter(tmp_path, monkeypatch, mutate):
    root, _path, dep = _fixture(tmp_path); mutate(root, dep)
    import neurodic.api.pin_multi_slover_dic as api
    monkeypatch.setattr(api, "solve_pin_multi_pair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("downstream scientific action must not execute")))
    with pytest.raises(ControlPlaneError):
        execute_trial(_c1_plan(dep, "dependency-negative"), managed_root=root, action_id="pin_multi.pair_solve_quality_call")


def test_upstream_dependencies_survive_trial_plan_round_trip(tmp_path):
    _root, _path, dep = _fixture(tmp_path); original = _c1_plan(dep, "dependency-roundtrip")
    loaded = json.loads(canonical_json(original))
    assert loaded["upstream_dependencies"] == original["upstream_dependencies"]
    assert loaded["plan_identity"] == original["plan_identity"]


def test_dependency_scientific_change_changes_plan_identity(tmp_path):
    _root, _path, dep = _fixture(tmp_path); baseline = _c1_plan(dep, "dependency-identity")
    for field, value in (("digest", "sha256:changed"),):
        changed = copy.deepcopy(dep); changed["producer_signature"][field] = value
        assert _c1_plan(changed, "dependency-identity")["plan_identity"] != baseline["plan_identity"]
    changed = copy.deepcopy(dep); changed["scope"]["pair_id"] = "cam_0__cam_2"; changed["producer_signature"]["scope"] = dict(changed["scope"])
    assert _c1_plan(changed, "dependency-identity")["plan_identity"] != baseline["plan_identity"]
    changed = copy.deepcopy(dep); changed["required_artifacts"][0]["identity"]["digest"] = "changed"
    assert _c1_plan(changed, "dependency-identity")["plan_identity"] != baseline["plan_identity"]


def test_dependency_management_reference_may_change_plan_identity(tmp_path):
    _root, _path, dep = _fixture(tmp_path); baseline = _c1_plan(dep, "dependency-management")
    changed = copy.deepcopy(dep); changed["source_trial_id"] = "other-source"
    assert _c1_plan(changed, "dependency-management")["plan_identity"] != baseline["plan_identity"]
