"""Read-only evidence extraction and quality reporting; no solver imports or execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .inspect import inspect_case, resolve_config
from .schemas import Availability, Envelope, FindingRecord, MetricRecord, QualityReport, ThresholdResult, canonical_json


EVALUATOR_VERSION = "neurodic-evaluator/v1"


def _metric(identifier: str, value: int | float | None, unit: str, availability: Availability, path: Path | None,
            field: str | None = None, *, aggregation: str | None = None, count: int | None = None,
            scope: Mapping[str, Any] | None = None, notes: str | None = None) -> MetricRecord:
    source = {"path": str(path) if path is not None else None, "field": field, "evaluator_version": EVALUATOR_VERSION}
    return MetricRecord(identifier, availability, unit, source, value, scope or {}, aggregation, count, notes)


def _stats(values: np.ndarray) -> tuple[int, float | None, float | None, float | None, float | None]:
    values = np.asarray(values).reshape(-1)
    finite = values[np.isfinite(values)]
    if not finite.size: return int(values.size), None, None, None, None
    return int(values.size), float(finite.mean()), float(np.median(finite)), float(np.percentile(finite, 95)), float(finite.size / values.size)


def _npz(path: Path) -> np.lib.npyio.NpzFile | None:
    try: return np.load(path, allow_pickle=False)
    except (OSError, ValueError): return None


def _json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, Mapping) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError): return None


def _add_npz_field(metrics: list[MetricRecord], identifier: str, path: Path, key: str, unit: str, *, aggregation: str = "mean") -> None:
    payload = _npz(path)
    if payload is None:
        metrics.append(_metric(identifier, None, unit, Availability.CORRUPT, path, key)); return
    try:
        if key not in payload.files: metrics.append(_metric(identifier, None, unit, Availability.NOT_AVAILABLE, path, key)); return
        values = payload[key]; count, mean, median, p95, finite = _stats(values)
        value = {"mean": mean, "median": median, "p95": p95}.get(aggregation)
        metrics.append(_metric(identifier, value, unit, Availability.DERIVED if value is not None else Availability.CORRUPT, path, key, aggregation=aggregation, count=count))
    finally: payload.close()


def _pin(root: Path, values: Mapping[str, Any]) -> list[MetricRecord]:
    output = root / values.get("output", {}).get("result", "result/pin_2d")
    paths = [output / "pin_result.npz", root / "result/pin/pin_result.npz"]
    path = next((item for item in paths if item.is_file()), None); metrics = []
    if path is None:
        return [_metric("field.displacement.finite_ratio", None, "ratio", Availability.NOT_AVAILABLE, None), _metric("training.fixed_evaluation_residual", None, "normalized_loss", Availability.NOT_AVAILABLE, None, notes="No fixed evaluation residual is exported by current PIN workflow")]
    data = _npz(path)
    if data is None: return [_metric("field.displacement.finite_ratio", None, "ratio", Availability.CORRUPT, path)]
    try:
        for key, identifier, unit in (("iterations", "training.iterations", "count"), ("final_loss", "training.loss.final", "normalized_loss")):
            metrics.append(_metric(identifier, float(np.asarray(data[key])), unit, Availability.OBSERVED if key in data.files else Availability.NOT_AVAILABLE, path, key))
        for key, identifier in (("displacement", "field.displacement.finite_ratio"), ("strain", "field.strain.finite_ratio")):
            if key in data.files:
                array = np.asarray(data[key]); metrics.append(_metric(identifier, float(np.isfinite(array).all(axis=1).mean()), "ratio", Availability.DERIVED, path, key, aggregation="row_all_finite", count=len(array)))
            else: metrics.append(_metric(identifier, None, "ratio", Availability.NOT_AVAILABLE, path, key))
    finally: data.close()
    evaluation = path.parent / "diagnostics_evaluation.json"
    payload = _json(evaluation)
    if payload is None:
        metrics.append(_metric("evaluation.photometric_residual.mean", None, "photometric_objective", Availability.NOT_AVAILABLE, evaluation, notes="Fixed evaluation artifact is absent"))
        metrics.append(_metric("evaluation.valid_ratio", None, "ratio", Availability.NOT_AVAILABLE, evaluation))
    else:
        summary = payload.get("summary", {})
        value = summary.get("mean")
        valid_ratio = payload.get("valid_ratio")
        availability = Availability.OBSERVED if isinstance(value, (int, float)) and np.isfinite(value) else Availability.CORRUPT
        metrics.append(_metric("evaluation.photometric_residual.mean", value if availability is Availability.OBSERVED else None, "photometric_objective", availability, evaluation, "summary.mean", scope=payload.get("evaluation_set", {}), aggregation="mean_per_valid_window"))
        metrics.append(_metric("evaluation.valid_ratio", valid_ratio if isinstance(valid_ratio, (int, float)) else None, "ratio", Availability.OBSERVED if isinstance(valid_ratio, (int, float)) else Availability.CORRUPT, evaluation, "valid_ratio", scope=payload.get("evaluation_set", {})))
    return metrics


def _stereo(root: Path, values: Mapping[str, Any]) -> list[MetricRecord]:
    base = root / values.get("output", {}).get("result", "result/pin_stereo"); legacy = root / "result"; base = base if base.exists() else legacy
    path = base / "reconstruct/initial.npz"; metrics = []
    data = _npz(path)
    if data is None: return [_metric("reconstruction.valid_ratio", None, "ratio", Availability.NOT_AVAILABLE, path), _metric("reconstruction.reprojection.p95", None, "px", Availability.NOT_AVAILABLE, path)]
    try:
        valid = np.asarray(data["valid"], dtype=bool) if "valid" in data.files else None
        if valid is None: metrics.append(_metric("reconstruction.valid_ratio", None, "ratio", Availability.NOT_AVAILABLE, path, "valid"))
        else: metrics.append(_metric("reconstruction.valid_ratio", float(valid.mean()), "ratio", Availability.DERIVED, path, "valid", aggregation="mean", count=len(valid)))
        error = np.asarray(data["reprojection_error"]) if "reprojection_error" in data.files else None
        if error is None: metrics.append(_metric("reconstruction.reprojection.p95", None, "px", Availability.NOT_AVAILABLE, path, "reprojection_error"))
        else: metrics.append(_metric("reconstruction.reprojection.p95", _stats(error)[3], "px", Availability.DERIVED, path, "reprojection_error", aggregation="p95", count=len(error)))
    finally: data.close()
    fields = ("reference_disparity", "left_temporal", "deformed_disparity")
    observed = []
    for field in fields:
        evaluation = base / "disp" / f"{field}_evaluation.json"
        payload = _json(evaluation)
        identifier = "evaluation.photometric_residual.mean"
        if payload is None:
            metrics.append(_metric(identifier, None, "photometric_objective", Availability.NOT_AVAILABLE, evaluation, scope={"field": field}))
            continue
        value = payload.get("summary", {}).get("mean")
        availability = Availability.OBSERVED if isinstance(value, (int, float)) and np.isfinite(value) else Availability.CORRUPT
        metrics.append(_metric(identifier, value if availability is Availability.OBSERVED else None, "photometric_objective", availability, evaluation, "summary.mean", scope={"field": field, **payload.get("evaluation_set", {})}, aggregation="mean_per_valid_window"))
        if availability is Availability.OBSERVED: observed.append(float(value))
    metrics.append(_metric("stereo.evaluation.photometric_residual.mean", float(np.mean(observed)) if len(observed) == 3 else None, "photometric_objective", Availability.DERIVED if len(observed) == 3 else Availability.NOT_AVAILABLE, None, aggregation="mean_over_three_planar_fields", count=len(observed)))
    geometry = base / "diagnostics/stereo_geometry.npz"; payload = _npz(geometry)
    if payload is not None:
        try:
            codes = np.asarray(payload["reason_code"]) if "reason_code" in payload.files else np.empty(0, int)
            names = [str(x) for x in payload["reason_names"]] if "reason_names" in payload.files else []
            for code, name in enumerate(names): metrics.append(_metric(f"reconstruction.reason.{name}_ratio", float((codes == code).mean()) if len(codes) else None, "ratio", Availability.DERIVED if len(codes) else Availability.CORRUPT, geometry, "reason_code", scope={"reason": name}, count=len(codes)))
            for state in ("reference", "current"):
                key=f"{state}_reprojection_error"
                if key in payload.files: metrics.append(_metric(f"reconstruction.{state}_reprojection.p95", _stats(np.asarray(payload[key]))[3], "px", Availability.DERIVED, geometry, key, aggregation="p95"))
        finally: payload.close()
    return metrics


def _pin_multi(root: Path, values: Mapping[str, Any]) -> list[MetricRecord]:
    configured = root / values.get("output", {}).get("result", "result/pin_multi"); base = configured if configured.exists() else root / "result/pin_multi_slover"; metrics = []
    quality_paths = sorted((base / "pairs").glob("*/quality/quality.json")) if (base / "pairs").is_dir() else []
    ratios = []
    for path in quality_paths:
        data = _json(path)
        if data is None: metrics.append(_metric("pin_multi.pair.valid_ratio", None, "ratio", Availability.CORRUPT, path)); continue
        pair = path.parents[2].name
        for field, identifier, unit in (("valid_ratio", "pin_multi.pair.valid_ratio", "ratio"), ("p95_reprojection_error_px", "pin_multi.pair.reprojection.p95", "px")):
            value = data.get(field); availability = Availability.OBSERVED if isinstance(value, (int, float)) else Availability.NOT_AVAILABLE
            metrics.append(_metric(identifier, value if availability is Availability.OBSERVED else None, unit, availability, path, field, scope={"pair": pair}))
            if identifier.endswith("valid_ratio") and availability is Availability.OBSERVED: ratios.append(float(value))
        for name, count in data.get("reason_codes", {}).items(): metrics.append(_metric(f"pin_multi.reason_code.{name}.ratio", float(count / data["total_points"]) if data.get("total_points", 0) else None, "ratio", Availability.DERIVED if data.get("total_points", 0) else Availability.CORRUPT, path, f"reason_codes.{name}", scope={"pair": pair}))
    metrics.append(_metric("pin_multi.pair.valid_ratio", float(np.mean(ratios)) if ratios else None, "ratio", Availability.DERIVED if ratios else Availability.NOT_AVAILABLE, None, aggregation="mean_over_observed_pairs", count=len(ratios)))
    fusion = _json(base / "fused/summary.json")
    consistency = _json(base / "fused/preselection_consistency.json")
    if consistency is not None:
        metrics.append(_metric("fusion.preselection.overlap_group_count", consistency.get("overlap_group_count"), "count", Availability.OBSERVED, base / "fused/preselection_consistency.json", "overlap_group_count"))
        for key in ("disagreement_median", "disagreement_p95"):
            value=consistency.get("summary",{}).get(key); metrics.append(_metric(f"fusion.preselection.displacement_{key}", value, "calibration_world_unit", Availability.OBSERVED if isinstance(value,(int,float)) else Availability.NOT_AVAILABLE, base / "fused/preselection_consistency.json", f"summary.{key}"))
    if not values.get("fusion", {}).get("enabled", False): metrics.append(_metric("fusion.selected_points", None, "count", Availability.NOT_APPLICABLE, None))
    elif fusion is None: metrics.append(_metric("fusion.selected_points", None, "count", Availability.NOT_AVAILABLE, base / "fused/summary.json"))
    else:
        metrics.append(_metric("fusion.selected_points", fusion.get("selected_points"), "count", Availability.OBSERVED, base / "fused/summary.json", "selected_points"))
        metrics.append(_metric("fusion.deduplicated_points", fusion.get("deduplicated_points"), "count", Availability.OBSERVED, base / "fused/summary.json", "deduplicated_points"))
    return metrics


def _ndef(root: Path, values: Mapping[str, Any]) -> list[MetricRecord]:
    output = values.get("output", {}); configured = root / output.get("result", "result") / output.get("ndef_subdir", "ndef"); base = configured if configured.exists() else root / "result/ndef_multi_slover"; metrics=[]
    scale_path=base/"precalculation/sparse_scale.json"; scale=_json(scale_path)
    if scale is None: metrics += [_metric("precalculation.track_ratio",None,"ratio",Availability.NOT_AVAILABLE,scale_path), _metric("precalculation.inlier_ratio",None,"ratio",Availability.NOT_AVAILABLE,scale_path)]
    else:
        tracks, inliers=scale.get("n_tracks"),scale.get("n_inliers")
        requested=sum(int(item.get("requested_seeds",0)) for item in scale.get("per_camera",[]) if isinstance(item,Mapping))
        track_ratio=float(tracks/requested) if isinstance(tracks,int) and requested else None
        inlier_ratio=float(inliers/tracks) if isinstance(tracks,int) and isinstance(inliers,int) and tracks else None
        metrics += [_metric("precalculation.track_count",tracks,"count",Availability.OBSERVED,scale_path,"n_tracks"),_metric("precalculation.requested_seed_count",requested if requested else None,"count",Availability.DERIVED if requested else Availability.NOT_AVAILABLE,scale_path,"per_camera.requested_seeds",aggregation="sum"),_metric("precalculation.track_ratio",track_ratio,"ratio",Availability.DERIVED if track_ratio is not None else Availability.CORRUPT,scale_path,"n_tracks/sum(per_camera.requested_seeds)"),_metric("precalculation.inlier_ratio",inlier_ratio,"ratio",Availability.DERIVED if inlier_ratio is not None else Availability.CORRUPT,scale_path,"n_inliers/n_tracks")]
    tracks_path=base/"precalculation/sparse_tracks.npz"; _add_npz_field(metrics,"precalculation.reprojection.p95",tracks_path,"current_reprojection_error","px",aggregation="p95")
    training_path=base/"diagnostics/training.npz"; data=_npz(training_path)
    if data is None: metrics += [_metric("training.loss.final",None,"normalized_loss",Availability.NOT_AVAILABLE,training_path),_metric("training.valid_pair_ratio.final",None,"ratio",Availability.NOT_AVAILABLE,training_path)]
    else:
        try:
            history=np.asarray(data["history"]) if "history" in data.files else np.empty((0,8)); finite=np.isfinite(history).all(axis=1) if history.ndim==2 else np.array([],bool)
            if history.size and finite.any():
                final=history[finite][-1]; best=float(np.min(history[finite,2])); ratio=float(final[5]/final[6]) if final[6]>0 else 0.0
                metrics += [_metric("training.loss.final",float(final[2]),"normalized_loss",Availability.OBSERVED,training_path,"history",aggregation="last_finite",count=len(history),notes="Sampled training loss, not fixed evaluation loss"),_metric("training.loss.best",best,"normalized_loss",Availability.DERIVED,training_path,"history",aggregation="min",count=len(history)),_metric("training.valid_pair_ratio.final",ratio,"ratio",Availability.DERIVED,training_path,"history.valid_pairs/supervised_pairs",aggregation="last_finite",count=len(history)),_metric("training.history.finite_ratio",float(finite.mean()),"ratio",Availability.DERIVED,training_path,"history",aggregation="row_all_finite",count=len(history))]
            else: metrics += [_metric("training.loss.final",None,"normalized_loss",Availability.CORRUPT,training_path,"history"),_metric("training.valid_pair_ratio.final",None,"ratio",Availability.CORRUPT,training_path,"history")]
        finally: data.close()
    field_path=base/"deformation/reference_to_current.npz"; data=_npz(field_path)
    if data is None: metrics.append(_metric("field.displacement.finite_ratio",None,"ratio",Availability.NOT_AVAILABLE,field_path))
    else:
        try:
            disp=np.asarray(data["displacement"]) if "displacement" in data.files else None
            if disp is None: metrics.append(_metric("field.displacement.finite_ratio",None,"ratio",Availability.NOT_AVAILABLE,field_path,"displacement"))
            else: metrics.append(_metric("field.displacement.finite_ratio",float(np.isfinite(disp).all(axis=1).mean()),"ratio",Availability.DERIVED,field_path,"displacement",aggregation="row_all_finite",count=len(disp)))
        finally: data.close()
    evaluation_path = base / "diagnostics/evaluation.json"; evaluation = _json(evaluation_path)
    if evaluation is None:
        metrics += [_metric("evaluation.photometric_residual.mean",None,"photometric_objective",Availability.NOT_AVAILABLE,evaluation_path), _metric("evaluation.valid_ratio",None,"ratio",Availability.NOT_AVAILABLE,evaluation_path)]
    else:
        value=evaluation.get("summary",{}).get("mean"); ratio=evaluation.get("valid_ratio")
        availability=Availability.OBSERVED if isinstance(value,(int,float)) and np.isfinite(value) else Availability.CORRUPT
        metrics += [_metric("evaluation.photometric_residual.mean",value if availability is Availability.OBSERVED else None,"photometric_objective",availability,evaluation_path,"summary.mean",scope=evaluation.get("evaluation_set",{}),aggregation="visible_count_weighted_mean_per_valid_pair"), _metric("evaluation.valid_ratio",ratio if isinstance(ratio,(int,float)) else None,"ratio",Availability.OBSERVED if isinstance(ratio,(int,float)) else Availability.CORRUPT,evaluation_path,"valid_ratio",scope=evaluation.get("evaluation_set",{}))]
        table=_npz(base/"diagnostics/evaluation.npz")
        if table is not None:
            try:
                for key, identifier in (("observation_positive_depth","evaluation.current_projection.positive_depth_ratio"),("observation_in_bounds","evaluation.current_projection.in_bounds_ratio"),("observation_patch_valid","evaluation.patch_valid_ratio")):
                    if key in table.files:
                        x=np.asarray(table[key],bool); metrics.append(_metric(identifier,float(x.mean()) if len(x) else None,"ratio",Availability.DERIVED if len(x) else Availability.CORRUPT,base/"diagnostics/evaluation.npz",key,count=len(x)))
            finally: table.close()
        for camera, stats in evaluation.get("cameras", {}).items():
            value=stats.get("residual_p95"); metrics.append(_metric("evaluation.view_residual.p95",value,"photometric_objective",Availability.OBSERVED if isinstance(value,(int,float)) else Availability.NOT_AVAILABLE,evaluation_path,f"cameras.{camera}.residual_p95",scope={"camera":camera},aggregation="p95"))
        for key in ("residual_spread_median", "residual_spread_p95"):
            value=evaluation.get("cross_view",{}).get(key); metrics.append(_metric(f"evaluation.cross_view.{key}",value,"photometric_objective",Availability.OBSERVED if isinstance(value,(int,float)) else Availability.NOT_AVAILABLE,evaluation_path,f"cross_view.{key}"))
    return metrics


def _profile(path: str | Path, solver: str) -> Mapping[str, Any]:
    import yaml
    payload=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or solver not in payload.get("solvers",{}): raise ValueError("Profile does not support selected solver")
    return {"id":payload["profile_id"],"version":payload["version"],**payload["solvers"][solver]}


def _compare(value: float, operator: str, threshold: Any) -> bool:
    return {">=":value>=threshold,">":value>threshold,"<=":value<=threshold,"<":value<threshold}[operator]


def evaluate_result(solver_config: str | Path, *, case_key: str | None = None, case_paths: str | Path = "config/case_paths.yaml", case_root: str | Path | None = None, solver: str | None = None, profile: str | Path = "config/quality_profiles/default.yaml") -> Envelope:
    """Evaluate existing artifacts only; report remains in memory and no solver is imported."""
    inspected=inspect_case(solver_config,case_key=case_key,case_paths=case_paths,case_root=case_root,solver=solver).to_dict()["data"]
    values=inspected["config"]["effective_config"]; root=Path(inspected["resolved_case_root"]); selected=inspected["solver"]
    metrics={"pin":_pin,"pin_stereo":_stereo,"pin_multi":_pin_multi,"ndef":_ndef}[selected](root,values); definition=_profile(profile,selected); index={item.id:item for item in metrics}; results=[]; findings=[]; required_missing=False; required_failed=False
    thresholded=set()
    for rule in definition.get("thresholds",[]):
        thresholded.add(rule["metric_id"])
        metric=index.get(rule["metric_id"]); available=metric is not None and metric.availability in {Availability.OBSERVED,Availability.DERIVED}; passed=_compare(float(metric.value),rule["operator"],rule["threshold"]) if available else None
        required=bool(rule.get("required",False)); reason=None if available else "metric_not_available"; results.append(ThresholdResult(rule["metric_id"],rule["operator"],rule["threshold"],metric.availability if metric else Availability.NOT_AVAILABLE,available,passed,required,metric.value if available else None,reason))
        if required and not available: required_missing=True
        if required and passed is False: required_failed=True
        if available and passed is False: findings.append(FindingRecord(rule["finding"],"error" if required else "warning",rule["stage"],[rule["metric_id"]],f"Observed {rule['metric_id']} violates profile threshold", "threshold"))
    for identifier in definition.get("required_metrics", []):
        if identifier in thresholded: continue
        metric=index.get(identifier)
        available=metric is not None and metric.availability in {Availability.OBSERVED, Availability.DERIVED}
        results.append(ThresholdResult(identifier,"required",None,metric.availability if metric else Availability.NOT_AVAILABLE,available,None,True,metric.value if available else None,None if available else "metric_not_available"))
        if not available: required_missing=True
    status="unknown" if required_missing else "fail" if required_failed else "warning" if findings else "pass"
    protected = {"solver": selected, "mode": values.get("mode"), "case": values.get("case", {}), "world_scale": values.get("reconstruction", {}).get("world_scale"), "scale": values.get("scale")}
    quality=QualityReport(selected,inspected["scope"],status=status,metrics=tuple(item.to_dict() for item in metrics),threshold_results=tuple(item.to_dict() for item in results),findings=tuple(item.to_dict() for item in findings),failure_stage=next((item.stage for item in findings),None),eligibility={"best_candidate":status=="pass","reasons":["required evidence unavailable"] if required_missing else ["required threshold failed"] if required_failed else []},profile={"id":definition["id"],"version":definition["version"],"evaluator_version":EVALUATOR_VERSION}, provenance={"result_kind":"full","scientific_identity":{"policy":"neurodic.protected-scientific-identity/v1","digest":"sha256:" + __import__("hashlib").sha256(canonical_json(protected).encode()).hexdigest()}})
    return Envelope(status="ok",operation="evaluate.result",data={"quality":quality.to_dict(),"inspection": {"solver":selected,"readiness":inspected["readiness"]}})
