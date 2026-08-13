"""Explicit, atomic best-reference management; no trial mutation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from .artifacts import require_path_within
from .compare import compare_quality_reports, quality_identity
from .schemas import Envelope, canonical_json, utc_now

BEST_SCHEMA_VERSION = "neurodic.best/v1"

def _load(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value.get("data", {}).get("quality", value.get("quality", value))
def _atomic(path: Path, data: Mapping[str, Any], root: Path) -> None:
    path = require_path_within(path, root); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(canonical_json(data) + "\n", encoding="utf-8"); os.replace(tmp, path)
def _best_identity(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json({k:v for k,v in value.items() if k != "best_identity"}).encode()).hexdigest()

def load_best(managed_root: str | Path) -> Envelope:
    root = Path(managed_root).resolve(); path = root / "best/current.json"
    return Envelope(status="ok", operation="best.show", data={"best": _load(path) if path.is_file() else None})

def evaluate_best_candidate(candidate_quality: str | Path, *, managed_root: str | Path, profile: str | Path = "config/comparison_profiles/default.yaml") -> Envelope:
    """Read-only current-best versus candidate comparison; never promotes."""
    current = load_best(managed_root).data["best"]
    if current is None: return Envelope(status="ok", operation="best.evaluate", data={"comparison": None, "decision": "no_current_best"})
    baseline = _load(current["result_ref"]["quality_path"])
    return compare_quality_reports(baseline, _load(candidate_quality), profile=profile)

def update_best(comparison: Mapping[str, Any], *, candidate_quality: str | Path, managed_root: str | Path,
                baseline_quality: str | Path, expected_current_best_identity: str | None = None) -> Envelope:
    """Promote only an already-computed eligible preferred candidate explicitly."""
    if comparison.get("schema_version") != "neurodic.comparison/v1": raise ValueError("Invalid comparison report")
    if comparison.get("eligibility", {}).get("status") != "eligible" or comparison.get("selection_decision", {}).get("decision") not in {"candidate_preferred", "no_current_best"}: raise ValueError("BEST.PROMOTION_BLOCKED")
    candidate = _load(candidate_quality); expected_quality = comparison["candidate_identity"]["quality_identity"]
    if quality_identity(candidate) != expected_quality: raise ValueError("BEST.COMPARISON_STALE")
    if quality_identity(_load(baseline_quality)) != comparison["baseline_identity"]["quality_identity"]: raise ValueError("BEST.COMPARISON_STALE")
    root = Path(managed_root).resolve(); best = root / "best"; history = best / "history"; history.mkdir(parents=True, exist_ok=True)
    current_path = best / "current.json"; previous = _load(current_path) if current_path.is_file() else None; previous_id = previous.get("best_identity") if previous else None
    if expected_current_best_identity != previous_id: raise ValueError("BEST.STATE_CHANGED")
    record = {"schema_version": BEST_SCHEMA_VERSION, "scope_key": {"solver": comparison["candidate_identity"]["solver"], "scientific_identity": comparison["candidate_identity"]["scientific_identity"], "scope": candidate.get("scope", {})}, "comparison_profile_identity": comparison["comparison_profile_identity"], "result_ref": {"kind": "quality_report", "quality_identity": expected_quality, "quality_path": str(Path(candidate_quality).resolve()), "trial_id": candidate.get("provenance", {}).get("trial_id")}, "comparison_identity": comparison["comparison_identity"]}
    record["best_identity"] = _best_identity(record)
    event = {"schema_version": "neurodic.best_promotion/v1", "previous_best": previous_id, "new_best": record["best_identity"], "comparison_identity": comparison["comparison_identity"], "reason": comparison["selection_decision"]["reasons"], "promoted_at": utc_now()}
    _atomic(history / f"{event['promoted_at'].replace(':', '').replace('-', '')}_{record['best_identity'][7:19]}.json", event, root); _atomic(current_path, record, root)
    return Envelope(status="ok", operation="best.promote", data={"best": record, "promotion": event})
