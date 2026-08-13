"""Deterministic, native-free interpretation of an existing QualityReport."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evaluate import evaluate_result
from .schemas import Availability, DiagnosisRecord, DiagnosisReport, Envelope

RULESET_VERSION = "neurodic-diagnosis-rules/v1"
_RANK = {"strong": 0, "moderate": 1, "weak": 2, "insufficient": 3}
_STAGE_ORDER = {"pin.train": 1, "stereo.reconstruct": 2, "pin_multi.pair_quality": 2,
                "ndef.precalculation": 1, "ndef.deformation.train": 2, "ndef.evaluate": 3}

def _available(metric: Mapping[str, Any] | None) -> bool:
    return bool(metric and metric.get("availability") in {Availability.OBSERVED.value, Availability.DERIVED.value})

def _ref(metric: Mapping[str, Any], effect: str | None = None) -> dict[str, Any]:
    value = {"metric_id": metric["id"], "scope": metric.get("scope", {}), "observed_value": metric.get("value"),
             "availability": metric.get("availability"), "source": metric.get("source", {})}
    if effect: value["effect"] = effect
    return value

def _missing(identifier: str, impact: str) -> dict[str, str]:
    return {"evidence_id": identifier, "reason": "not_available", "impact": impact}

def _record(code: str, stage: str, family: str, support: str, evidence: list[Mapping[str, Any]], *,
            causes: list[tuple[str, str]] = [], contradict: list[Mapping[str, Any]] = [], missing: list[Mapping[str, Any]] = [],
            next_observation: str | None = None, role: str = "secondary") -> DiagnosisRecord:
    return DiagnosisRecord(code, stage, family, support, tuple(evidence), tuple(contradict), tuple(missing),
                           tuple({"cause_code": c, "label": label, "supporting_evidence_refs": [x["metric_id"] for x in evidence]} for c, label in causes), next_observation, role)

def diagnose_quality_report(quality: Mapping[str, Any]) -> DiagnosisReport:
    """Diagnose one normalized report only; no artifact, solver, or GPU access."""
    if quality.get("schema_version") != "neurodic.quality/v1":
        raise ValueError("diagnose_quality_report requires neurodic.quality/v1")
    solver = str(quality["solver"]); metrics = {x["id"]: x for x in quality.get("metrics", []) if isinstance(x, Mapping)}
    diagnoses: list[DiagnosisRecord] = []; missing: list[Mapping[str, Any]] = []
    def metric(key: str): return metrics.get(key)
    def val(key: str): return metric(key).get("value") if _available(metric(key)) else None
    def add_missing(key: str, impact: str):
        item = _missing(key, impact); missing.append(item); return item
    if solver == "pin":
        finite, evaluation = val("field.displacement.finite_ratio"), val("evaluation.valid_ratio")
        if finite is None: add_missing("field.displacement.finite_ratio", "cannot classify field integrity")
        elif finite < 1: diagnoses.append(_record("PIN.FIELD.NONFINITE", "pin.train", "FIELD.NONFINITE", "strong", [_ref(metric("field.displacement.finite_ratio"))], causes=[("FIELD.NONFINITE_OUTPUT", "nonfinite field output")]))
        if evaluation is None: add_missing("evaluation.valid_ratio", "cannot classify evaluation support")
        elif evaluation == 0: diagnoses.append(_record("PIN.EVALUATION.NO_VALID_OBSERVATIONS", "pin.train", "EVALUATION.NO_VALID_OBSERVATIONS", "strong", [_ref(metric("evaluation.valid_ratio"))], causes=[("OBSERVATION.SUPPORT_LOSS", "insufficient valid photometric support"), ("FIELD.WARP_INVALIDITY", "field projection or warping invalidity")]))
    elif solver == "pin_stereo":
        valid = val("reconstruction.valid_ratio")
        reason_ids = {name: f"reconstruction.reason.{name}_ratio" for name in ("invalid_field", "outside_roi", "out_of_bounds", "negative_depth", "reprojection_error")}
        available = {name: val(key) for name, key in reason_ids.items()}
        if valid is None: add_missing("reconstruction.valid_ratio", "cannot classify reconstruction")
        elif valid == 0:
            absent = [add_missing(key, "cannot determine dominant reconstruction rejection") for key in reason_ids.values() if val(key) is None]
            support = "strong" if not absent else "moderate"
            diagnoses.append(_record("STEREO.RECONSTRUCTION.NO_VALID_POINTS", "stereo.reconstruct", "RECONSTRUCTION.NO_VALID_POINTS", support, [_ref(metric("reconstruction.valid_ratio"))], missing=absent, causes=[("GEOMETRY.RECONSTRUCTION_REJECTION", "complete geometric rejection")]))
        for name, family, causes in (("invalid_field", "RECONSTRUCTION.INVALID_PLANAR_INPUT", [("PLANAR_FIELD.INVALID", "one or more planar displacement fields are invalid")]), ("out_of_bounds", "RECONSTRUCTION.CURRENT_PROJECTION_OUT_OF_DOMAIN", [("VIEW.DOMAIN_LOSS", "current geometry leaves observable image domain"), ("PLANAR_FIELD.INCONSISTENT", "upstream planar field is incompatible with stereo support")]), ("negative_depth", "RECONSTRUCTION.DEPTH_VALIDITY_FAILURE", [("CORRESPONDENCE.INCONSISTENT", "incompatible correspondences"), ("GEOMETRY.TRIANGULATION_INCONSISTENT", "triangulation geometry inconsistency")]), ("reprojection_error", "RECONSTRUCTION.REPROJECTION_INCONSISTENCY", [("CORRESPONDENCE.INCONSISTENT", "correspondence inconsistency"), ("CAMERA_GEOMETRY.INCONSISTENT", "camera geometry inconsistency")])):
            ratio = available[name]
            if ratio is not None and ratio > 0.5:
                contradiction=[]; ref=val("reconstruction.reference_reprojection.p95"); cur=val("reconstruction.current_reprojection.p95")
                if name == "reprojection_error" and ref is not None and cur is not None and ref <= 5 < cur: contradiction=[_ref(metric("reconstruction.reference_reprojection.p95"), "reference state remains lower-error than current state")]
                diagnoses.append(_record(f"STEREO.{family}", "stereo.reconstruct", family, "moderate" if contradiction else "strong", [_ref(metric(reason_ids[name]))], causes=causes, contradict=contradiction, next_observation="camera-specific triangulation diagnostics"))
    elif solver == "ndef":
        tracks, train, projection = val("precalculation.track_ratio"), val("training.valid_pair_ratio.final"), val("evaluation.valid_ratio")
        if tracks is None: add_missing("precalculation.track_ratio", "cannot classify precalculation support")
        elif tracks == 0: diagnoses.append(_record("NDEF.PRECALC.NO_VALID_TRACKS", "ndef.precalculation", "PRECALC.NO_VALID_TRACKS", "strong", [_ref(metric("precalculation.track_ratio"))], causes=[("CORRESPONDENCE.INSUFFICIENT_SUPPORT", "insufficient correspondence support"), ("OBSERVATION.VISIBILITY_SUPPORT_LOSS", "insufficient texture or visibility support")]))
        if train is None: add_missing("training.valid_pair_ratio.final", "cannot classify training observations")
        elif train == 0: diagnoses.append(_record("NDEF.TRAINING.NO_VALID_OBSERVATIONS", "ndef.deformation.train", "TRAINING.NO_VALID_OBSERVATIONS", "strong", [_ref(metric("training.valid_pair_ratio.final"))], causes=[("VIEW.OBSERVATION_SUPPORT_LOSS", "insufficient visible current observations")], role="consequent" if tracks == 0 else "secondary"))
        finite_history = val("training.history.finite_ratio")
        if finite_history is None: add_missing("training.history.finite_ratio", "cannot classify numerical training stability")
        elif finite_history < 1 and train is not None and train > 0:
            diagnoses.append(_record("NDEF.TRAINING.NUMERICAL_FAILURE", "ndef.deformation.train", "TRAINING.NUMERICAL_FAILURE", "strong", [_ref(metric("training.history.finite_ratio")), _ref(metric("training.valid_pair_ratio.final"))], causes=[("OPTIMIZATION.STEP_INSTABILITY", "nonfinite training history despite available observations")]))
        if projection is None: add_missing("evaluation.valid_ratio", "cannot classify current projection")
        elif projection == 0:
            components=[("evaluation.current_projection.positive_depth_ratio", "VIEW.CURRENT_DEPTH_INVALID"), ("evaluation.current_projection.in_bounds_ratio", "VIEW.CURRENT_OUT_OF_BOUNDS"), ("evaluation.patch_valid_ratio", "VIEW.PATCH_SUPPORT_INVALID")]
            observed=[(key,family,val(key)) for key,family in components if val(key) is not None]
            if observed:
                key,family,value=min(observed,key=lambda item:item[2]); diagnoses.append(_record(f"NDEF.{family}", "ndef.evaluate", family, "strong", [_ref(metric("evaluation.valid_ratio")), _ref(metric(key))], causes=[("VIEW.GEOMETRIC_VALIDITY_LOSS", "current view observation validity collapse")]))
            else: diagnoses.append(_record("NDEF.VIEW.CURRENT_VALIDITY_FAILURE", "ndef.evaluate", "VIEW.CURRENT_VALIDITY_FAILURE", "moderate", [_ref(metric("evaluation.valid_ratio"))], missing=[add_missing("evaluation.current_projection.*", "cannot distinguish depth, bounds, and patch support")]))
    elif solver == "pin_multi":
        ratio = val("pin_multi.pair.valid_ratio")
        if ratio is None: add_missing("pin_multi.pair.valid_ratio", "cannot classify pair reconstruction")
        elif ratio == 0: diagnoses.append(_record("PIN_MULTI.PAIR.NO_VALID_RECONSTRUCTION", "pin_multi.pair_quality", "PAIR.NO_VALID_RECONSTRUCTION", "strong", [_ref(metric("pin_multi.pair.valid_ratio"))], causes=[("PAIR.RECONSTRUCTION_REJECTION", "all pair reconstructions lack valid points")]))
        groups=val("fusion.preselection.overlap_group_count")
        if groups is None: add_missing("fusion.preselection.overlap_group_count", "cannot classify cross-pair overlap")
        elif groups == 0: diagnoses.append(_record("PIN_MULTI.FUSION.INSUFFICIENT_MULTI_PAIR_OVERLAP", "pin_multi.fusion", "FUSION.INSUFFICIENT_MULTI_PAIR_OVERLAP", "moderate", [_ref(metric("fusion.preselection.overlap_group_count"))], causes=[("FUSION.INSUFFICIENT_OVERLAP", "insufficient multi-pair spatial overlap")]))
    stages = {"pin": ["pin.train"], "pin_stereo": ["stereo.reconstruct"], "pin_multi": ["pin_multi.pair_quality", "pin_multi.fusion"], "ndef": ["ndef.precalculation", "ndef.deformation.train", "ndef.evaluate"]}[solver]
    diagnoses.sort(key=lambda x: (_STAGE_ORDER.get(x.failure_stage, 99), _RANK[x.support], x.code))
    if diagnoses:
        primary = diagnoses[0]; diagnoses[0] = DiagnosisRecord(**{**primary.__dict__, "role": "primary"})
        status = "diagnosed" if primary.support in {"strong", "moderate"} else "partial"
        primary_code = diagnoses[0].code
    else: status = "insufficient_evidence" if missing or quality.get("status") == "unknown" else "no_failure_detected"; primary_code = None
    return DiagnosisReport(solver, quality.get("scope", {}), status, tuple(diagnoses), primary_code, tuple(stages), tuple(missing), (), RULESET_VERSION)

def diagnose_result(solver_config: str | Path, **kwargs: Any) -> Envelope:
    quality = evaluate_result(solver_config, **kwargs).to_dict()["data"]["quality"]
    report = diagnose_quality_report(quality)
    return Envelope(status="ok", operation="diagnose.result", data={"diagnosis": report.to_dict(), "quality": quality})

def load_quality_report(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise ValueError("Quality report must be an object")
    return value
