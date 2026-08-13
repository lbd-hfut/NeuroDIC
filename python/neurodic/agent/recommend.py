"""Bounded, native-free intervention hypotheses. Never executes a trial."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from .config import effective_config_identity, lookup, owner_stages
from .inspect import resolve_config
from .parameters import is_parameter_safe, load_intervention_rules, load_parameter_registry, parameter_for
from .schemas import Envelope, canonical_json
from .trials import plan_trial

RECOMMENDATION_SCHEMA_VERSION = "neurodic.recommendation/v1"
_SUPPORT = {"strong": 3, "moderate": 2, "weak": 1, "insufficient": 0}

@dataclass(frozen=True)
class ParameterChangeRecommendation:
    path: str; old_value: Any; new_value: Any; rule_id: str; reason: str
    expected_mechanism: str; owner_stage: str; risk_class: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)

@dataclass(frozen=True)
class RecommendationReport:
    solver: str; scope: Mapping[str, Any]; recommendation_status: str; parameter_changes: tuple[Mapping[str, Any], ...]
    sparse_override: Mapping[str, Any]; source_diagnosis_identity: str; rule_set_identity: str; parameter_registry_identity: str
    schema_version: str = RECOMMENDATION_SCHEMA_VERSION
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)

def diagnosis_identity(diagnosis: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(diagnosis).encode()).hexdigest()
def _nested(path: str, value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}; target = result
    parts = path.split(".")
    for key in parts[:-1]: target = target.setdefault(key, {})
    target[parts[-1]] = value; return result
def _unwrap(value: Mapping[str, Any], key: str) -> Mapping[str, Any]: return value.get("data", {}).get(key, value.get(key, value))

def recommend_from_diagnosis(diagnosis: Mapping[str, Any], solver_config: str | Path, *, case_key: str | None = None,
                            case_paths: str | Path = "config/case_paths.yaml", trial_id: str | None = None,
                            parameter_registry: str | Path = "config/agent/parameter_registry.yaml",
                            intervention_rules: str | Path = "config/agent/intervention_rules.yaml") -> Envelope:
    """Produce at most one planner-validated sparse override; never imports execution."""
    diagnosis = _unwrap(diagnosis, "diagnosis")
    if diagnosis.get("schema_version") != "neurodic.diagnosis/v1": raise ValueError("recommend_from_diagnosis requires DiagnosisReport")
    solver = diagnosis.get("solver"); resolved = resolve_config(solver_config, case_key=case_key, case_paths=case_paths, solver=solver)
    if resolved["solver"] != solver: raise ValueError("Diagnosis/config solver mismatch")
    registry, registry_id = load_parameter_registry(parameter_registry); rules, rules_id = load_intervention_rules(intervention_rules)
    primary = next((x for x in diagnosis.get("diagnoses", []) if x.get("role") == "primary"), None)
    report: dict[str, Any] = {"schema_version": RECOMMENDATION_SCHEMA_VERSION, "solver": solver, "scope": diagnosis.get("scope", {}), "source_diagnosis_identity": diagnosis_identity(diagnosis), "effective_config_identity": effective_config_identity(resolved["effective_config"]), "rule_set_identity": rules_id, "parameter_registry_identity": registry_id, "recommendation_status": "no_matching_rule", "execution_status": "not_performed", "matched_rules": [], "selected_rule": None, "parameter_changes": [], "sparse_override": {}, "planning_result": None, "blocked_reasons": [], "missing_evidence": [], "contradictions": [], "notes": ["A recommendation is a bounded intervention hypothesis, not an outcome guarantee.", "Execution was not performed."]}
    if not primary: report["recommendation_status"] = "insufficient_evidence"; report["blocked_reasons"].append("primary diagnosis unavailable"); return _envelope(report)
    support = primary.get("support", "insufficient")
    if support in {"weak", "insufficient"}: report["recommendation_status"] = "observation_only"; report["blocked_reasons"].append("diagnosis support too weak for automatic override"); return _envelope(report)
    causes = {x.get("cause_code") for x in primary.get("candidate_causes", [])}
    candidates = [x for x in rules["rules"] if x["solver"] == solver and primary.get("failure_family") in x["applicable_failure_families"] and causes.intersection(x["applicable_candidate_causes"])]
    report["matched_rules"] = [x["rule_id"] for x in sorted(candidates, key=lambda x: (x["priority"], x["rule_id"]))]
    if not candidates: return _envelope(report)
    eligible: list[Mapping[str, Any]] = []
    evidence = {x.get("metric_id") for x in primary.get("supporting_evidence", [])}; contradictions = {x.get("metric_id") for x in primary.get("contradicting_evidence", [])}
    for rule in candidates:
        if _SUPPORT[support] < _SUPPORT[rule["minimum_support"]] or (support == "moderate" and not rule.get("allows_moderate", False)):
            continue
        missing = sorted(set(rule["required_evidence"]) - evidence)
        blocked = sorted(set(rule.get("contradicting_evidence", [])) & contradictions)
        if missing: report["missing_evidence"].extend(missing); continue
        if blocked: report["contradictions"].extend(blocked); continue
        eligible.append(rule)
    if not eligible:
        report["recommendation_status"] = "blocked_by_contradiction" if report["contradictions"] else "insufficient_evidence"; return _envelope(report)
    rule = sorted(eligible, key=lambda x: (x["priority"], len(x["parameter_changes"]), x["rule_id"]))[0]
    changes = []
    for change in rule["parameter_changes"]:
        metadata = parameter_for(registry, solver, change["path"])
        if metadata is None or not is_parameter_safe(metadata, solver) or rule["stage_ownership"] not in (owner_stages(solver, change["path"]) or ()) or metadata["owner_stage"] != rule["stage_ownership"]: report["recommendation_status"] = "protected_parameter_only"; report["blocked_reasons"].append(change["path"]); return _envelope(report)
        old = lookup(resolved["effective_config"], change["path"])
        if change["policy"] != "multiply" or not isinstance(old, (int, float)): report["recommendation_status"] = "insufficient_evidence"; report["blocked_reasons"].append("unsupported step policy"); return _envelope(report)
        new = old * change["factor"]; hard = metadata["config_valid_range"]; safe = metadata.get("auto_safe_range")
        if not hard[0] <= new <= hard[1] or not safe or not safe[0] <= new <= safe[1] or new == old:
            report["recommendation_status"] = "insufficient_evidence"; report["blocked_reasons"].append("automatic safe bound reached"); return _envelope(report)
        changes.append({"path": change["path"], "old_value": old, "new_value": new, "rule_id": rule["rule_id"], "reason": f"diagnosis {primary['code']} matched reviewed intervention rule", "expected_mechanism": rule["expected_mechanism"], "owner_stage": rule["stage_ownership"], "risk_class": rule["risk_class"]})
    if len(changes) != 1: raise ValueError("Single-change policy violation")
    override = _nested(changes[0]["path"], changes[0]["new_value"])
    planned = plan_trial(solver_config, case_key=case_key, case_paths=case_paths, override=override, trial_id=trial_id).to_dict()["data"]["trial_plan"]
    report.update({"selected_rule": {k: rule[k] for k in ("rule_id", "version", "evidence_level", "risk_class", "stage_ownership", "expected_mechanism", "source_rationale")}, "parameter_changes": changes, "sparse_override": override, "planning_result": planned})
    report["recommendation_status"] = "plan_blocked" if planned["plan_status"] == "blocked" else "recommended"
    if planned["plan_status"] == "blocked": report["blocked_reasons"].append("Loop 6 planner blocked sparse override")
    return _envelope(report)

def _envelope(report: Mapping[str, Any]) -> Envelope:
    core = {k:v for k,v in report.items() if k != "recommendation_identity"}
    value = dict(report); value["recommendation_identity"] = "sha256:" + hashlib.sha256(canonical_json(core).encode()).hexdigest()
    return Envelope(status="ok", operation="recommend.from_diagnosis", data={"recommendation": value})
