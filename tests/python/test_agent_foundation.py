"""Loop 1 contracts are solver-free and must remain read-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurodic.agent.artifacts import (
    ArtifactRecord,
    IdentityStrength,
    canonical_path,
    content_identity,
    metadata_identity,
    path_within,
    require_path_within,
)
from neurodic.agent.errors import ControlPlaneError, ErrorRecord, error_envelope
from neurodic.agent.schemas import Availability, CapabilityRecord, Envelope, QualityReport, canonical_json, is_utc_timestamp, utc_now
from neurodic.agent.state import RunRecord, StageRecord, StageStatus, TrialRecord, atomic_write_json, valid_stage_transition


def _snapshot(root: Path) -> list[tuple[str, int, int]]:
    return sorted((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in root.rglob("*") if path.is_file())


def test_envelope_error_contract_and_strict_round_trip() -> None:
    error = ErrorRecord("FILESYSTEM.NOT_FOUND", "Surface file is missing", True,
                        stage="ndef.surface", path="surface.npz", details={"attempt": 1})
    envelope = error_envelope("inspect.artifact", error, request_id="req_test")
    payload = envelope.to_dict()
    assert payload["schema_version"] == "neurodic.agent/v1"
    assert payload["status"] == "error"
    assert json.loads(canonical_json(payload)) == payload
    assert payload["errors"][0]["details"] == {"attempt": 1}
    with pytest.raises(ValueError, match="requires at least one error"):
        Envelope(status="error", operation="x", request_id="req_x", data={})
    with pytest.raises(ValueError, match="Unknown control-plane error"):
        ErrorRecord("TRAINING.DIVERGED", "not Loop 1", False)


def test_quality_foundation_preserves_missing_evidence() -> None:
    report = QualityReport(solver="pin", scope={"frame": 1}, metrics=(
        {"id": "loss", "availability": Availability.NOT_AVAILABLE, "value": None},
    ))
    payload = report.to_dict()
    assert payload["schema_version"] == "neurodic.quality/v1"
    assert payload["status"] == "unknown"
    assert payload["metrics"][0]["availability"] == "not_available"


def test_canonical_json_is_strict_and_paths_are_serialized() -> None:
    assert canonical_json({"b": 1, "a": Path("x")}) == '{"a":"x","b":1}'
    with pytest.raises(ValueError, match="NaN"):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError, match="infinity"):
        canonical_json({"value": float("inf")})
    assert is_utc_timestamp(utc_now())
    assert not is_utc_timestamp("2026-08-13T00:00:00+08:00")


def test_metadata_and_content_identity_are_distinct_and_location_is_not_identity(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    first, second = root / "first.npz", root / "second.npz"
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")
    first_metadata = metadata_identity(first, root=root)
    assert first_metadata.strength is IdentityStrength.METADATA
    assert first_metadata.basis["path"] == "first.npz"
    assert metadata_identity(first, root=root).digest == first_metadata.digest
    first.write_bytes(b"changed-content")
    changed = metadata_identity(first, root=root)
    assert changed.digest != first_metadata.digest
    content = content_identity(second)
    assert content.strength is IdentityStrength.CONTENT
    assert content.algorithm == "sha256"
    record = ArtifactRecord.from_file(second, artifact_type="surface", artifact_schema="npz/v1",
                                      producer_stage="ndef.surface.fuse", root=root)
    assert record.location == "second.npz"
    assert record.artifact_id != record.identity.digest
    assert record.identity.digest != content.digest


def test_metadata_identity_changes_when_mtime_changes(tmp_path: Path) -> None:
    path = tmp_path / "product.bin"
    path.write_bytes(b"contents")
    before = metadata_identity(path)
    stat = path.stat()
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert metadata_identity(path).digest != before.digest


def test_canonical_containment_and_escape_rejection(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    product = root / "outputs" / "result.json"
    product.parent.mkdir()
    product.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    assert canonical_path(product) == product.resolve()
    assert path_within(product, root)
    assert require_path_within(root / "outputs" / ".." / "outputs" / "result.json", root, require_exists=True) == product.resolve()
    assert not path_within(outside, root)
    with pytest.raises(ControlPlaneError) as caught:
        require_path_within(root / ".." / "outside.json", root, require_exists=True)
    assert caught.value.record.code == "FILESYSTEM.OUTSIDE_ROOT"
    with pytest.raises(ControlPlaneError) as missing:
        require_path_within(root / "missing.json", root, require_exists=True)
    assert missing.value.record.code == "FILESYSTEM.NOT_FOUND"


def test_read_only_helpers_do_not_mutate_fixture_tree(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    product = root / "result" / "artifact.npz"
    product.parent.mkdir()
    product.write_bytes(b"fixture")
    before = _snapshot(root)
    artifact = ArtifactRecord.from_file(product, artifact_type="fixture", artifact_schema="npz/v1",
                                        producer_stage="fixture.stage", root=root)
    canonical_path(product, require_exists=True)
    metadata_identity(product, root=root)
    content_identity(product)
    require_path_within(product, root, require_exists=True)
    assert artifact.size_bytes == len(b"fixture")
    assert _snapshot(root) == before


def test_existing_pin_multi_and_ndef_artifacts_fit_v1_record_without_mutation() -> None:
    repository = Path(__file__).resolve().parents[2]
    case = repository / "case" / "Multi" / "CylinderDIC"
    products = (
        (case / "result" / "pin_multi_slover" / "manifest.json", "pin_multi_manifest", "json/v1", "pin_multi.pair_solve"),
        (case / "result" / "ndef_multi_slover" / "diagnostics" / "summary.json", "ndef_summary", "json/v1", "ndef.deformation.infer"),
    )
    before = _snapshot(case)
    for path, artifact_type, artifact_schema, producer_stage in products:
        if not path.is_file():
            pytest.skip(f"Representative artifact is absent in this checkout: {path}")
        record = ArtifactRecord.from_file(path, artifact_type=artifact_type, artifact_schema=artifact_schema,
                                          producer_stage=producer_stage, root=case)
        assert record.identity.strength is IdentityStrength.METADATA
        assert not Path(record.location).is_absolute()
        assert json.loads(canonical_json(record.to_dict())) == record.to_dict()
    assert _snapshot(case) == before


def test_stage_state_machine_preserves_interrupted_and_terminal_immutability() -> None:
    pending = StageRecord("ndef.deformation.train", StageStatus.PENDING, 1, "adapter/v1")
    assert valid_stage_transition(StageStatus.PENDING, StageStatus.RUNNING)
    running = pending.transition(StageStatus.RUNNING, timestamp="2026-08-13T00:00:00Z")
    interrupted = running.transition(StageStatus.INTERRUPTED, timestamp="2026-08-13T00:01:00Z")
    assert interrupted.status is StageStatus.INTERRUPTED
    assert interrupted.finished_at == "2026-08-13T00:01:00Z"
    with pytest.raises(ValueError, match="Illegal stage transition"):
        interrupted.transition(StageStatus.RUNNING)
    completed = running.transition(StageStatus.COMPLETED, timestamp="2026-08-13T00:01:00Z")
    with pytest.raises(ValueError, match="Illegal stage transition"):
        completed.transition(StageStatus.RUNNING)
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        StageRecord("stage", StageStatus.RUNNING, 1, "v1", started_at="2026-08-13T08:00:00+08:00")


def test_run_trial_capability_and_atomic_json_foundation(tmp_path: Path) -> None:
    capabilities = CapabilityRecord(reuse_supported=True, cache_supported=False, resume_supported=False,
                                     notes="resume is not implemented")
    run = RunRecord("run_a", "ndef", "case_meta_digest", str(tmp_path), {"frame": 1}, {"id": "default"},
                    created_at="2026-08-13T00:00:00Z", capabilities=capabilities)
    trial = TrialRecord("baseline", run.run_id, "effective_digest", {"runtime": 23},
                        created_at="2026-08-13T00:00:00Z")
    target = tmp_path / "record.json"
    atomic_write_json(target, {"run": run.to_dict(), "trial": trial.to_dict()}, root=tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run"]["capabilities"]["resume_supported"] is False
    assert payload["trial"]["parent_trial_id"] is None
    assert not list(tmp_path.glob(".record.json.*.tmp"))
    with pytest.raises(ControlPlaneError) as outside:
        atomic_write_json(tmp_path.parent / "outside.json", {}, root=tmp_path)
    assert outside.value.record.code == "FILESYSTEM.OUTSIDE_ROOT"


def test_schema_files_are_valid_json_and_close_v1_records() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema_root = repository / "schemas" / "agent" / "v1"
    for path in sorted(schema_root.glob("*.schema.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    artifact = json.loads((schema_root / "artifact.schema.json").read_text(encoding="utf-8"))
    quality = json.loads((schema_root / "quality.schema.json").read_text(encoding="utf-8"))
    assert artifact["additionalProperties"] is False
    assert quality["additionalProperties"] is False
