"""Native-free deterministic QualityReport comparison (Loop 8)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import Envelope, canonical_json

COMPARISON_SCHEMA_VERSION = "neurodic.comparison/v1"
_KNOWN_METRICS = {
    "pin": {"field.displacement.finite_ratio", "evaluation.photometric_residual.mean", "evaluation.valid_ratio", "training.loss.final"},
    "pin_stereo": {"reconstruction.valid_ratio", "reconstruction.reprojection.p95", "stereo.evaluation.photometric_residual.mean", "evaluation.photometric_residual.mean"},
    "pin_multi": {"pin_multi.pair.valid_ratio", "pin_multi.pair.reprojection.p95", "fusion.selected_points", "fusion.preselection.overlap_group_count", "fusion.preselection.displacement_disagreement_p95"},
    "ndef": {"precalculation.track_ratio", "training.valid_pair_ratio.final", "field.displacement.finite_ratio", "evaluation.photometric_residual.mean", "evaluation.valid_ratio", "evaluation.current_projection.positive_depth_ratio", "evaluation.current_projection.in_bounds_ratio", "evaluation.view_residual.p95", "evaluation.cross_view.residual_spread_p95"},
}


def quality_identity(quality: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(quality).encode()).hexdigest()


def _profile(path: str | Path, solver: str) -> tuple[Mapping[str, Any], str]:
    raw = Path(path).read_bytes(); data = yaml.safe_load(raw)
    if not isinstance(data, Mapping) or data.get("schema_version") != "neurodic.comparison_profile/v1":
        raise ValueError("Invalid comparison profile schema")
    if solver not in data.get("solvers", {}): raise ValueError("Comparison profile does not support solver")
    definition = data["solvers"][solver]
    if not set(definition.get("metrics", {})).issubset(_KNOWN_METRICS[solver]): raise ValueError("Comparison profile declares an unknown metric")
    tolerance = {**data.get("default_tolerance", {}), **definition.get("tolerance", {})}
    if not all(isinstance(tolerance.get(key), (int, float)) and math.isfinite(tolerance[key]) and tolerance[key] >= 0 for key in ("atol", "rtol")): raise ValueError("Comparison profile tolerance must be finite and non-negative")
    if any(rule.get("direction") not in {"lower_is_better", "higher_is_better", "neutral"} for rule in definition.get("metrics", {}).values()): raise ValueError("Comparison profile has invalid metric direction")
    return data, "sha256:" + hashlib.sha256(raw).hexdigest()


def _identity(q: Mapping[str, Any]) -> Mapping[str, Any]:
    # Provenance is deliberately report-owned; paths never establish scientific identity.
    return {"quality_identity": quality_identity(q), "solver": q.get("solver"), "scope": q.get("scope", {}), "scientific_identity": q.get("provenance", {}).get("scientific_identity"),
            "result_kind": q.get("provenance", {}).get("result_kind", "full"), "execution_status": q.get("provenance", {}).get("execution_status")}


def _scope(value: Mapping[str, Any]) -> str: return canonical_json(value.get("scope", {}))
def _evaluation(scope: Mapping[str, Any]) -> Any:
    return scope.get("evaluation_set_identity") or scope.get("evaluation_identity")
def _metric_key(metric: Mapping[str, Any]) -> tuple[str, str, str | None, str]:
    scope = dict(metric.get("scope", {})); scope.pop("evaluation_set_identity", None); scope.pop("evaluation_identity", None)
    return str(metric.get("id")), canonical_json(scope), metric.get("aggregation"), str(metric.get("unit"))
def _available(metric: Mapping[str, Any]) -> bool: return metric.get("availability") in {"observed", "derived"} and isinstance(metric.get("value"), (int, float)) and math.isfinite(metric["value"])


def _metric_comparison(base: Mapping[str, Any] | None, cand: Mapping[str, Any] | None, rule: Mapping[str, Any], tolerance: Mapping[str, Any]) -> Mapping[str, Any]:
    metric = cand or base or {}; direction = rule.get("direction")
    result = {"metric_id": metric.get("id"), "scope": metric.get("scope", {}), "aggregation": metric.get("aggregation"), "unit": metric.get("unit"),
              "direction": direction, "baseline_value": base.get("value") if base else None, "candidate_value": cand.get("value") if cand else None,
              "baseline_availability": base.get("availability") if base else "not_available", "candidate_availability": cand.get("availability") if cand else "not_available",
              "delta": None, "relative_delta": None, "comparison_status": "incomparable", "comparison_support": "insufficient", "notes": []}
    if base is None or cand is None or not _available(base) or not _available(cand):
        if base and cand and base.get("availability") == cand.get("availability") == "not_applicable": result["comparison_status"] = "not_applicable"
        elif (cand or {}).get("availability") == "corrupt" or (base or {}).get("availability") == "corrupt": result["comparison_status"] = "unusable"
        else: result["notes"].append("missing metric is not a regression")
        return result
    if base.get("unit") != cand.get("unit"): result["notes"].append("unit mismatch"); return result
    if rule.get("fixed_evaluation") and (not _evaluation(base.get("scope", {})) or _evaluation(base.get("scope", {})) != _evaluation(cand.get("scope", {}))):
        result["notes"].append("fixed evaluation identity mismatch"); return result
    b, c = float(base["value"]), float(cand["value"]); delta = c-b
    result["delta"] = delta
    if abs(b) > float(tolerance["atol"]): result["relative_delta"] = delta / b
    equal = abs(delta) <= max(float(tolerance["atol"]), float(tolerance["rtol"]) * max(abs(b), abs(c)))
    result["comparison_support"] = "strong" if base.get("sample_count") == cand.get("sample_count") else "moderate"
    if direction == "neutral": result["comparison_status"] = "neutral"; return result
    if equal: result["comparison_status"] = "unchanged"; return result
    result["comparison_status"] = "improved" if ((direction == "lower_is_better" and delta < 0) or (direction == "higher_is_better" and delta > 0)) else "regressed"
    return result


def compare_quality_reports(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, profile: str | Path = "config/comparison_profiles/default.yaml") -> Envelope:
    """Read-only O(M) comparison. It never reads raw solver outputs or executes a trial."""
    solver = baseline.get("solver")
    profile_data, profile_id = _profile(profile, str(solver))
    reasons: list[str] = []
    if solver != candidate.get("solver"): reasons.append("solver mismatch")
    if _scope(baseline) != _scope(candidate): reasons.append("scope mismatch")
    bi, ci = _identity(baseline), _identity(candidate)
    if not bi["scientific_identity"] or not ci["scientific_identity"]: reasons.append("protected scientific identity unavailable")
    elif canonical_json(bi["scientific_identity"]) != canonical_json(ci["scientific_identity"]): reasons.append("protected scientific identity mismatch")
    rules = profile_data["solvers"][solver]; tolerance = {**profile_data.get("default_tolerance", {}), **rules.get("tolerance", {})}
    bm = {_metric_key(x): x for x in baseline.get("metrics", [])}; cm = {_metric_key(x): x for x in candidate.get("metrics", [])}
    comparisons = []
    for key in sorted(set(bm) | set(cm)):
        identifier = key[0]; rule = rules.get("metrics", {}).get(identifier)
        if rule is None: continue # no explicit direction => never selection evidence
        comparisons.append(_metric_comparison(bm.get(key), cm.get(key), rule, tolerance))
    comparable = [x for x in comparisons if x["comparison_status"] not in {"incomparable", "unusable", "not_applicable"}]
    comparability = "incomparable" if reasons or not comparable else ("comparable" if len(comparable) == len(comparisons) else "partially_comparable")
    hard = set(rules.get("hard_findings", [])); findings = {x.get("code") for x in candidate.get("findings", [])}
    guardrails = [x for x in comparisons if rules.get("metrics", {}).get(x["metric_id"], {}).get("role") == "guardrail"]
    required = set(rules.get("required_metrics", [])); by_id = {x.get("id"): x for x in candidate.get("metrics", [])}
    missing = sorted(item for item in required if not _available(by_id.get(item, {})))
    hard_failure = sorted(hard & findings) + sorted(x["metric_id"] for x in comparisons if x["comparison_status"] == "unusable" and x["metric_id"] in required)
    partial = ci["result_kind"] != "full" or ci["execution_status"] in {"partial", "failed", "interrupted"}
    eligibility = {"status": "ineligible" if hard_failure or partial else "insufficient_evidence" if missing else "eligible",
                   "hard_failures": hard_failure + (["RESULT.PARTIAL"] if partial else []), "missing_required_metrics": missing}
    primary = [x for x in comparisons if rules.get("metrics", {}).get(x["metric_id"], {}).get("role") == "primary"]
    primary_improved = any(x["comparison_status"] == "improved" for x in primary)
    primary_regressed = any(x["comparison_status"] == "regressed" for x in primary)
    guardrail_regressed = any(x["comparison_status"] == "regressed" for x in guardrails)
    if eligibility["status"] != "eligible": decision, why = "candidate_ineligible", "candidate is not eligible"
    elif comparability == "incomparable": decision, why = "incomparable", "results are not scientifically comparable"
    elif primary_improved and not primary_regressed and not guardrail_regressed: decision, why = "candidate_preferred", "strict primary improvement without guardrail regression"
    else: decision, why = "retain_current", "tie or Pareto conflict retains current"
    summary = {"improved_metrics": [x["metric_id"] for x in comparisons if x["comparison_status"] == "improved"], "regressed_metrics": [x["metric_id"] for x in comparisons if x["comparison_status"] == "regressed"], "unchanged_metrics": [x["metric_id"] for x in comparisons if x["comparison_status"] == "unchanged"], "incomparable_metrics": [x["metric_id"] for x in comparisons if x["comparison_status"] in {"incomparable", "unusable"}]}
    core = {"schema_version": COMPARISON_SCHEMA_VERSION, "baseline_identity": bi, "candidate_identity": ci, "comparison_profile_identity": profile_id,
            "comparability": {"status": comparability, "reasons": reasons}, "metric_comparisons": comparisons, "guardrails": guardrails, "eligibility": eligibility,
            "summary": summary, "selection_decision": {"decision": decision, "reasons": [why], "tie_policy": "retain_current"}}
    core["comparison_identity"] = "sha256:" + hashlib.sha256(canonical_json(core).encode()).hexdigest()
    return Envelope(status="ok", operation="compare.quality", data={"comparison": core})


def compare_results(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, profile: str | Path = "config/comparison_profiles/default.yaml") -> Envelope:
    """Alias for callers that compare named baseline/current-best results."""
    return compare_quality_reports(baseline, candidate, profile=profile)


def select_best_candidate(comparison: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read-only deterministic selection already encoded by a ComparisonReport."""
    return dict(comparison.get("selection_decision", {"decision": "incomparable", "reasons": ["invalid comparison"]}))
