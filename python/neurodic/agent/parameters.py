"""Auditable parameter and intervention-rule registries for Loop 9."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import yaml
from .config import owner_stages, protected_violations

PARAMETER_REGISTRY_VERSION = "neurodic.parameter_registry/v1"
RULE_SET_VERSION = "neurodic.intervention_rules/v1"

@dataclass(frozen=True)
class ParameterMetadata:
    path: str; solver: str; owner_stage: str; type: str; config_valid_range: Sequence[float | int]
    trial_modifiable: bool; auto_recommendable: bool; auto_safe_range: Sequence[float | int] | None
    direction_semantics: str; default_step_policy: str | None; coupled_with: Sequence[str]; risk_class: str; protected: bool; notes: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)

@dataclass(frozen=True)
class InterventionRule:
    rule_id: str; version: int; solver: str; applicable_failure_families: Sequence[str]
    applicable_candidate_causes: Sequence[str]; minimum_support: str; required_evidence: Sequence[str]
    parameter_changes: Sequence[Mapping[str, Any]]; coupling_policy: str; stage_ownership: str
    expected_mechanism: str; risk_class: str; stop_conditions: Sequence[str]; evidence_level: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)

def _load(path: str | Path, schema: str) -> tuple[Mapping[str, Any], str]:
    raw = Path(path).read_bytes(); value = yaml.safe_load(raw)
    if not isinstance(value, Mapping) or value.get("schema_version") != schema: raise ValueError("Invalid registry schema")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()

def load_parameter_registry(path: str | Path = "config/agent/parameter_registry.yaml") -> tuple[Mapping[str, Any], str]:
    value, identity = _load(path, PARAMETER_REGISTRY_VERSION)
    params = value.get("parameters", [])
    if not isinstance(params, list) or any(not isinstance(x, Mapping) for x in params): raise ValueError("Invalid parameter registry entries")
    for item in params:
        if not all(key in item for key in ("path", "solver", "owner_stage", "type", "config_valid_range", "trial_modifiable", "auto_recommendable", "risk_class", "protected")): raise ValueError("Incomplete parameter metadata")
        owners = owner_stages(item["solver"], item["path"])
        if not item["protected"] and item["owner_stage"] not in (owners or ()): raise ValueError("Parameter owner stage conflicts with Loop 6 ownership")
        if item["protected"] and owners and item["owner_stage"] not in owners: raise ValueError("Protected parameter owner stage conflicts with Loop 6 ownership")
        if item["auto_recommendable"] and (item["protected"] or not item["trial_modifiable"] or item["risk_class"] != "low"): raise ValueError("Unsafe automatic parameter metadata")
    return value, identity

def load_intervention_rules(path: str | Path = "config/agent/intervention_rules.yaml") -> tuple[Mapping[str, Any], str]:
    value, identity = _load(path, RULE_SET_VERSION)
    rules = value.get("rules", [])
    if not isinstance(rules, list) or any(not isinstance(x, Mapping) for x in rules): raise ValueError("Invalid intervention rule entries")
    for item in rules:
        needed = {"rule_id", "version", "priority", "solver", "applicable_failure_families", "applicable_candidate_causes", "minimum_support", "required_evidence", "parameter_changes", "coupling_policy", "stage_ownership", "risk_class", "evidence_level", "stop_conditions"}
        if not needed.issubset(item): raise ValueError("Incomplete intervention rule")
        if len(item["parameter_changes"]) > 2 or (len(item["parameter_changes"]) != 1 and item["coupling_policy"] != "explicit_coupled"): raise ValueError("Invalid automatic change count")
    return value, identity

def parameter_for(registry: Mapping[str, Any], solver: str, path: str) -> Mapping[str, Any] | None:
    return next((x for x in registry["parameters"] if x["solver"] == solver and x["path"] == path), None)

def is_parameter_safe(metadata: Mapping[str, Any], solver: str) -> bool:
    path = str(metadata["path"])
    return bool(metadata.get("trial_modifiable") and metadata.get("auto_recommendable") and not metadata.get("protected") and metadata.get("risk_class") == "low" and not protected_violations(solver, (path,)) and owner_stages(solver, path))
