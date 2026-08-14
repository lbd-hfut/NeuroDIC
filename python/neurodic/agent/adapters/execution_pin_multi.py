"""Guarded, CPU-only PIN Multi pair-ROI execution adapter."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ...case_io import image_files
from ...pin_multi_roi import (_options_from_config, generate_pin_multi_pair_roi,
                               pair_id_for, select_pin_multi_pairs, camera_name_from_label)
from ..artifacts import require_path_within
from ..artifacts import content_identity
from ..execution import ProducedArtifact, TrustedAction


_PAIR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*__[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _run_pair_roi(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[str]:
    """Write one explicitly planned pair ROI below the supplied staging root."""
    pair_id = scope.get("pair_id")
    if not isinstance(pair_id, str) or not _PAIR.fullmatch(pair_id):
        raise ValueError("PIN Multi pair-ROI execution requires a validated scope.pair_id")
    case = values.get("case", {}); root = Path(case["root"]).resolve()
    calibration_value = Path(case["calibration"]); calibration = calibration_value if calibration_value.is_absolute() else root / calibration_value
    calibration = require_path_within(calibration, root, require_exists=True)
    selection, options = _options_from_config(values)
    selected = select_pin_multi_pairs(json.loads(calibration.read_text(encoding="utf-8")), selection)
    found = next(((left, right) for left, right, _details in selected if pair_id_for(left, right) == pair_id), None)
    if found is None: raise ValueError(f"Planned pair_id is not selected by the frozen configuration: {pair_id}")
    image_root = require_path_within(root / str(case["images"]), root, require_exists=True)
    left, right = found
    left_path = require_path_within(image_files(image_root / left)[0], root, require_exists=True)
    right_path = require_path_within(image_files(image_root / right)[0], root, require_exists=True)
    output = require_path_within(staging / pair_id, staging)
    result = generate_pin_multi_pair_roi(left_path, right_path, output, options=options)
    if result.get("status") != "ok": raise ValueError(f"Pair ROI did not meet artifact contract: {result.get('reason', 'unknown')}")
    return [str(path.relative_to(staging)) for path in sorted(output.iterdir()) if path.is_file()]


def _input_identities(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    """The pair-ROI contract consumes precisely its selected pair plus calibration."""
    pair_id = plan.get("scope", {}).get("pair_id")
    if not isinstance(pair_id, str) or not _PAIR.fullmatch(pair_id):
        raise ValueError("PIN Multi pair-ROI signature requires a validated scope.pair_id")
    case = values.get("case", {}); root = Path(case["root"]).resolve()
    calibration_value = Path(case["calibration"])
    calibration = calibration_value if calibration_value.is_absolute() else root / calibration_value
    calibration = require_path_within(calibration, root, require_exists=True)
    selection, _options = _options_from_config(values)
    selected = select_pin_multi_pairs(json.loads(calibration.read_text(encoding="utf-8")), selection)
    found = next(((left, right) for left, right, _details in selected if pair_id_for(left, right) == pair_id), None)
    if found is None: raise ValueError("Planned pair_id is not selected by the frozen configuration")
    image_root = require_path_within(root / str(case["images"]), root, require_exists=True)
    left, right = found
    left_path = require_path_within(image_files(image_root / left)[0], root, require_exists=True)
    right_path = require_path_within(image_files(image_root / right)[0], root, require_exists=True)
    return {"baseline_config": plan["baseline"]["effective_config_identity"],
            "calibration": content_identity(calibration).to_dict(),
            "reference_images": {str(left_path.relative_to(root)): content_identity(left_path).to_dict(),
                                 str(right_path.relative_to(root)): content_identity(right_path).to_dict()}}


def guarded_pair_roi_action() -> TrustedAction:
    """The sole real Loop 7 adapter currently approved for CPU smoke tests."""
    return TrustedAction("pin_multi.separate_pair_roi_call", _run_pair_roi,
                         "neurodic.pin_multi.pair_roi/v1", input_identities=_input_identities)


def _solve_scope(scope: Mapping[str, Any]) -> tuple[str, str, str, int]:
    pair_id = scope.get("pair_id"); frame = scope.get("selected_frame")
    if not isinstance(pair_id, str) or not _PAIR.fullmatch(pair_id) or not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("PIN Multi pair solve requires validated scope.pair_id and scope.selected_frame")
    reference, secondary = pair_id.split("__", 1)
    return pair_id, reference, secondary, frame


def _solve_inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    pair_id, reference, secondary, frame = _solve_scope(plan.get("scope", {}))
    case = values["case"]; root = Path(case["root"]).resolve()
    calibration_value = Path(case["calibration"]); calibration = calibration_value if calibration_value.is_absolute() else root / calibration_value
    calibration = require_path_within(calibration, root, require_exists=True)
    image_root = require_path_within(root / str(case["images"]), root, require_exists=True)
    from ...case_io import named_multiview_image_pairs
    cameras = sorted(json.loads(calibration.read_text(encoding="utf-8")).get("cameras", []), key=lambda item: str(item.get("label", "")))
    names = [camera_name_from_label(str(item.get("label", ""))) for item in cameras]
    references, frames = named_multiview_image_pairs(image_root, names)
    if frame >= len(frames): raise ValueError("PIN Multi selected_frame is outside the resolved case")
    ref, current = dict(zip(names, references)), dict(zip(names, frames[frame]))
    paths = {"reference_image": ref[reference], "secondary_reference_image": ref[secondary],
             "current_image": current[reference], "secondary_current_image": current[secondary]}
    return {"baseline_config": plan["baseline"]["effective_config_identity"], "pair_id": pair_id,
            "reference_camera": reference, "secondary_camera": secondary,
            "calibration": content_identity(calibration).to_dict(),
            "images": {role: content_identity(require_path_within(path, root, require_exists=True)).to_dict() for role, path in paths.items()}}


def _solve_outputs(pair_id: str) -> list[ProducedArtifact]:
    base = f"scientific/pairs/{pair_id}"
    outputs = [ProducedArtifact(f"{base}/disp/{name}.npz", f"pin_multi_field.{name}", "neurodic.pin_multi.field/v1")
               for name in ("reference_disparity", "left_temporal", "deformed_disparity")]
    outputs += [ProducedArtifact(f"{base}/reconstruct/{name}.npz", f"pin_multi_reconstruction.{name}", "neurodic.pin_multi.reconstruction/v1") for name in ("reference", "current")]
    outputs += [ProducedArtifact(f"{base}/deformation/initial_to_current.npz", "pin_multi_deformation", "neurodic.pin_multi.deformation/v1"),
                ProducedArtifact(f"{base}/deformation/initial_to_current_summary.json", "pin_multi_deformation_summary", "json/v1"),
                ProducedArtifact(f"{base}/quality/reason_codes.npy", "pin_multi_reason_codes", "npy/v1"),
                ProducedArtifact(f"{base}/quality/quality.json", "pin_multi_pair_quality", "json/v1"),
                ProducedArtifact(f"{base}/pair_metadata.json", "pin_multi_pair_metadata", "neurodic.pin_multi.pair_solve_quality/v1")]
    return outputs


def validate_pair_solve_quality_outputs(root: Path, pair_id: str) -> list[ProducedArtifact]:
    """Read-only structural validation for the published C1 output contract."""
    outputs = _solve_outputs(pair_id)
    for item in outputs:
        path = root / item.path
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"PIN Multi required output missing: {item.path}")
    import numpy as np
    base = root / f"scientific/pairs/{pair_id}"
    def npz(path: Path, keys: set[str], finite: set[str]) -> dict[str, Any]:
        try:
            with np.load(path, allow_pickle=False) as value:
                if keys - set(value.files): raise ValueError("missing required keys")
                for key in finite:
                    if not np.all(np.isfinite(np.asarray(value[key]))): raise ValueError("non-finite required numeric data")
                return {key: np.asarray(value[key]) for key in keys}
        except (OSError, ValueError) as error:
            raise ValueError(f"PIN Multi invalid NPZ {path.name}: {error}") from error
    for name in ("reference_disparity", "left_temporal", "deformed_disparity"):
        value = npz(base / f"disp/{name}.npz", {"coordinates", "displacement", "iterations", "final_loss"}, {"coordinates", "displacement", "iterations", "final_loss"})
        if value["coordinates"].ndim != 2 or value["coordinates"].shape[-1] != 2 or value["displacement"].shape != value["coordinates"].shape:
            raise ValueError(f"PIN Multi field {name} has incompatible coordinate/displacement shapes")
    for name in ("reference", "current"):
        value = npz(base / f"reconstruct/{name}.npz", {"left_coordinates", "right_coordinates", "points", "valid", "reprojection_error"}, {"left_coordinates", "right_coordinates", "points", "reprojection_error"})
        count = value["valid"].size
        if value["valid"].ndim != 1 or value["left_coordinates"].shape != (count, 2) or value["right_coordinates"].shape != (count, 2) or value["points"].shape != (count, 3) or value["reprojection_error"].shape != (count,):
            raise ValueError(f"PIN Multi reconstruction {name} has incompatible shapes")
    value = npz(base / "deformation/initial_to_current.npz", {"coordinates", "reference_points", "current_points", "displacement", "valid"}, {"coordinates", "reference_points", "current_points", "displacement"})
    count = value["valid"].size
    if value["valid"].ndim != 1 or value["coordinates"].shape != (count, 2) or any(value[key].shape != (count, 3) for key in ("reference_points", "current_points", "displacement")):
        raise ValueError("PIN Multi deformation has incompatible shapes")
    try: codes = np.load(base / "quality/reason_codes.npy", allow_pickle=False)
    except (OSError, ValueError) as error: raise ValueError("PIN Multi reason codes are invalid") from error
    if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer): raise ValueError("PIN Multi reason codes must be a one-dimensional integer array")
    try: quality = json.loads((base / "quality/quality.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise ValueError("PIN Multi quality JSON is invalid") from error
    if not isinstance(quality, Mapping) or not {"total_points", "valid_points", "valid_ratio", "reason_codes"}.issubset(quality): raise ValueError("PIN Multi quality JSON lacks required keys")
    if not isinstance(quality["total_points"], int) or not isinstance(quality["valid_points"], int) or quality["total_points"] < 0 or quality["valid_points"] < 0 or quality["valid_points"] > quality["total_points"] or not isinstance(quality["reason_codes"], Mapping) or not np.isfinite(quality["valid_ratio"]): raise ValueError("PIN Multi quality JSON has invalid counts")
    if codes.size != quality["total_points"]: raise ValueError("PIN Multi reason-code and quality counts differ")
    return outputs


def _run_pair_solve_quality(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    pair_id, reference, secondary, frame = _solve_scope(scope)
    dependency = scope.get("_managed_dependencies", {}).get("pair_roi")
    if not isinstance(dependency, Mapping) or dependency.get("scope", {}).get("pair_id") != pair_id:
        raise ValueError("PIN Multi pair solve requires an exact approved pair_roi dependency")
    roi = Path(dependency.get("files", {}).get("left_mask.npy", ""))
    if not roi.is_file(): raise ValueError("Approved pair_roi dependency lacks left_mask.npy")
    case = values["case"]; root = Path(case["root"]).resolve(); calibration = require_path_within(root / case["calibration"], root, require_exists=True)
    from ...api.pin_multi_slover_dic import solve_pin_multi_pair
    solve_pin_multi_pair(values, pair_id=pair_id, reference_camera=reference, secondary_camera=secondary,
                         selected_frame=frame, pair_roi_dir=roi.parent, calibration_path=calibration,
                         result_root=staging / "scientific", visualization_root=staging / "visualization")
    outputs = _solve_outputs(pair_id)
    metadata = json.loads((staging / outputs[-1].path).read_text(encoding="utf-8"))
    if metadata.get("pair_id") != pair_id or metadata.get("reference_camera") != reference or metadata.get("secondary_camera") != secondary or metadata.get("selected_frame") != frame:
        raise ValueError("PIN Multi pair metadata does not match frozen scope")
    return validate_pair_solve_quality_outputs(staging, pair_id)


def guarded_pair_solve_quality_action() -> TrustedAction:
    return TrustedAction("pin_multi.pair_solve_quality_call", _run_pair_solve_quality,
                         "neurodic.pin_multi.pair_solve_quality/v1",
                         output_contract="neurodic.pin_multi.pair-solve-quality-artifacts/v1", input_identities=_solve_inputs)


def _fusion_outputs() -> list[ProducedArtifact]:
    base = "scientific/fused"
    return [ProducedArtifact(f"{base}/{name}", artifact, schema) for name, artifact, schema in (
        ("reference_surface.npz", "pin_multi_fused_reference_surface", "neurodic.pin_multi.fused-surface/v1"),
        ("current_surface.npz", "pin_multi_fused_current_surface", "neurodic.pin_multi.fused-surface/v1"),
        ("deformation.npz", "pin_multi_fused_deformation", "neurodic.pin_multi.fused-deformation/v1"),
        ("strain.npz", "pin_multi_fused_strain", "neurodic.pin_multi.fused-strain/v1"),
        ("summary.json", "pin_multi_fusion_summary", "json/v1"),
    )]


def validate_fusion_postprocess_outputs(root: Path, planned_pair_ids: Sequence[str], values: Mapping[str, Any]) -> list[ProducedArtifact]:
    """Read-only validator for the mandatory C3 scientific output contract."""
    import numpy as np
    outputs = _fusion_outputs(); fused = root / "scientific/fused"
    for item in outputs:
        if not (root / item.path).is_file(): raise ValueError(f"PIN Multi fusion output missing: {item.path}")
    def load(name: str, required: set[str]):
        try:
            value = np.load(fused / name, allow_pickle=False)
            if required - set(value.files): raise ValueError("missing keys")
            return value
        except (OSError, ValueError) as error: raise ValueError(f"PIN Multi fusion {name} invalid: {error}") from error
    reference = load("reference_surface.npz", {"points", "valid", "reprojection_error", "source_pair", "pair_names", "voxel_size"})
    current = load("current_surface.npz", {"points", "valid", "reprojection_error", "source_pair", "pair_names", "voxel_size"})
    deformation = load("deformation.npz", {"coordinates", "reference_points", "current_points", "displacement", "valid", "source_pair", "pair_names", "voxel_size"})
    strain = load("strain.npz", {"coordinates", "strain", "valid", "source_pair", "pair_names", "voxel_size"})
    try: summary = json.loads((fused / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise ValueError("PIN Multi fusion summary invalid") from error
    n = reference["points"].shape[0]
    for value, fields in ((reference, ("points", "valid", "reprojection_error", "source_pair")), (current, ("points", "valid", "reprojection_error", "source_pair")), (deformation, ("coordinates", "reference_points", "current_points", "displacement", "valid", "source_pair")), (strain, ("coordinates", "strain", "valid", "source_pair"))):
        if any(np.asarray(value[key]).shape[0] != n for key in fields): raise ValueError("PIN Multi fusion output lengths disagree")
        if list(np.asarray(value["pair_names"]).astype(str)) != list(planned_pair_ids): raise ValueError("PIN Multi fusion pair_names disagree with frozen order")
        if np.any(np.asarray(value["source_pair"]) < 0) or np.any(np.asarray(value["source_pair"]) >= len(planned_pair_ids)): raise ValueError("PIN Multi fusion has invalid source_pair")
    if reference["points"].ndim != 2 or reference["points"].shape[1:] != (3,) or current["points"].shape != reference["points"].shape or deformation["displacement"].shape != reference["points"].shape or strain["strain"].shape != (n, 6): raise ValueError("PIN Multi fusion output shapes are invalid")
    finite_fields = ((reference, ("points", "reprojection_error")), (current, ("points", "reprojection_error")),
                     (deformation, ("coordinates", "reference_points", "current_points", "displacement")),
                     (strain, ("coordinates", "strain")))
    if any(not np.all(np.isfinite(np.asarray(value[key]))) for value, keys in finite_fields for key in keys):
        raise ValueError("PIN Multi fusion required numeric output is non-finite")
    expected = float(values.get("fusion", {}).get("voxel_size", 1.0))
    if any(float(np.asarray(value["voxel_size"])) != expected for value in (reference, current, deformation, strain)): raise ValueError("PIN Multi fusion voxel_size disagrees with configuration")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("selected_points"), int) or summary["selected_points"] != n:
        raise ValueError("PIN Multi fusion summary selected_points disagrees with outputs")
    return outputs


def _fusion_scope(scope: Mapping[str, Any]) -> tuple[list[str], int]:
    pairs, frame = scope.get("planned_pair_ids"), scope.get("selected_frame")
    if not isinstance(pairs, list) or not pairs or len(set(pairs)) != len(pairs) or not all(isinstance(item, str) and _PAIR.fullmatch(item) for item in pairs):
        raise ValueError("C3 requires unique ordered planned_pair_ids")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0: raise ValueError("C3 requires explicit selected_frame")
    if not isinstance(scope.get("planned_pair_set_identity"), str) or not isinstance(scope.get("fusion_input_identity"), str):
        raise ValueError("C3 requires C2 planned_pair_set_identity and fusion_input_identity")
    return list(pairs), frame


def _fusion_inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    pairs, _frame = _fusion_scope(plan.get("scope", {})); case = values["case"]; root = Path(case["root"]).resolve()
    calibration = require_path_within(root / case["calibration"], root, require_exists=True)
    return {"calibration": content_identity(calibration).to_dict(), "planned_pair_set_identity": plan["scope"]["planned_pair_set_identity"],
            "fusion_input_identity": plan["scope"]["fusion_input_identity"], "planned_pair_ids": pairs}


def _run_fusion_postprocess(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    pairs, frame = _fusion_scope(scope); dependencies = scope.get("_managed_dependencies", {})
    if not isinstance(dependencies, Mapping) or set(dependencies) != {f"pair/{pair}" for pair in pairs}:
        raise ValueError("C3 requires exactly one managed dependency for every planned pair")
    inputs = []
    for pair in pairs:
        dep = dependencies[f"pair/{pair}"]
        if dep.get("scope", {}).get("pair_id") != pair or dep.get("scope", {}).get("selected_frame") != frame:
            raise ValueError("C3 managed dependency scope mismatches frozen pair set")
        files = dep.get("files", {}); reference, current = files.get("reference.npz"), files.get("current.npz")
        if not reference or not current: raise ValueError("C3 dependency lacks fusion reconstruction artifacts")
        inputs.append({"pair_id": pair, "reference_reconstruction": reference, "current_reconstruction": current})
    from ...pin_multi_fusion import fuse_pin_multi_managed_pairs
    fuse_pin_multi_managed_pairs(values, ordered_pair_inputs=inputs, result_root=staging / "scientific", visualization_root=staging / "visualization")
    return validate_fusion_postprocess_outputs(staging, pairs, values)


def guarded_fusion_postprocess_action() -> TrustedAction:
    return TrustedAction("pin_multi.fusion_postprocess_call", _run_fusion_postprocess,
                         "neurodic.pin_multi.fusion_postprocess/v1",
                         output_contract="neurodic.pin_multi.fusion-postprocess-artifacts/v1", input_identities=_fusion_inputs)
