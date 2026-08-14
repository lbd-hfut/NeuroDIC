"""Trusted staging-only adapter for the atomic Stereo PIN public API."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...case_io import stereo_image_pairs
from ..artifacts import content_identity, require_path_within
from ..execution import ProducedArtifact, TrustedAction

_FIELDS = ("reference_disparity", "left_temporal", "deformed_disparity")

def _freeze(values: Mapping[str, Any], scope: Mapping[str, Any]) -> dict[str, Any]:
    frame = scope.get("selected_frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("Stereo execution requires validated scope.selected_frame")
    frozen = copy.deepcopy(dict(values)); frozen.setdefault("case", {})["frame"] = frame
    return frozen

def _paths(values: Mapping[str, Any]) -> tuple[Path, dict[str, Path]]:
    case = values["case"]; root = Path(case["root"]).resolve()
    (l0, r0), pairs = stereo_image_pairs(root / case["left_images"], root / case["right_images"])
    frame = int(case["frame"])
    try: lk, rk = pairs[frame]
    except IndexError as error: raise ValueError("Stereo selected_frame is outside matched deformed pairs") from error
    roi = require_path_within(root / case["roi"], root, require_exists=True)
    calibration = require_path_within(root / case["camera_pair"], root, require_exists=True)
    return root, {"reference_left": require_path_within(l0, root, require_exists=True), "reference_right": require_path_within(r0, root, require_exists=True),
                  "deformed_left": require_path_within(lk, root, require_exists=True), "deformed_right": require_path_within(rk, root, require_exists=True),
                  "roi_mask": roi, "calibration": calibration}

def _overlay(values: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    overlay = copy.deepcopy(dict(values)); root = require_path_within(staging, staging, require_exists=True)
    overlay["output"] = {"result": str(root / "scientific"), "visualization": str(root / "visualization")}
    return overlay

def _out(values: Mapping[str, Any]) -> list[ProducedArtifact]:
    out: list[ProducedArtifact] = []
    for field in _FIELDS:
        out.append(ProducedArtifact(f"scientific/disp/{field}.npz", f"stereo_field.{field}", "neurodic.stereo.field/v1"))
    out += [ProducedArtifact("scientific/reconstruct/initial.npz", "reference_reconstruction", "neurodic.stereo.reconstruction/v1"),
            ProducedArtifact("scientific/reconstruct/last.npz", "current_reconstruction", "neurodic.stereo.reconstruction/v1"),
            ProducedArtifact("scientific/deformation/initial_to_last.npz", "stereo_deformation", "neurodic.stereo.deformation/v1"),
            ProducedArtifact("scientific/deformation/initial_to_last_summary.json", "stereo_deformation_summary", "json/v1"),
            ProducedArtifact("scientific/diagnostics/stereo_geometry.npz", "stereo_geometry", "neurodic.stereo_geometry/v1"),
            ProducedArtifact("scientific/diagnostics/stereo_geometry.json", "stereo_geometry_summary", "neurodic.stereo_geometry/v1"),
            ProducedArtifact("scientific/diagnostics/field_provenance.json", "stereo_field_provenance", "neurodic.stereo.fields/v1")]
    if values.get("evaluation", {}).get("enabled", False):
        for field in _FIELDS:
            out += [ProducedArtifact(f"scientific/disp/{field}_evaluation.npz", f"stereo_evaluation.{field}", "neurodic.fixed_evaluation/v1"),
                    ProducedArtifact(f"scientific/disp/{field}_evaluation.json", f"stereo_evaluation_summary.{field}", "neurodic.fixed_evaluation/v1")]
    return out

def _npz(path: Path, keys: set[str], finite: set[str]) -> None:
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        if keys - set(data.files): raise ValueError(f"Stereo artifact {path.name} lacks required keys")
        for key in finite:
            value = np.asarray(data[key])
            if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)): raise ValueError(f"Stereo artifact {path.name} has non-finite {key}")

def _field_provenance(staging: Path) -> None:
    path = staging / "scientific/diagnostics/field_provenance.json"; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): return
    path.write_text(json.dumps({"schema_version":"neurodic.stereo.fields/v1", "fields": {
        "reference_disparity":{"reference":"reference_left","target":"reference_right"},
        "left_temporal":{"reference":"reference_left","target":"deformed_left"},
        "deformed_disparity":{"reference":"reference_left","target":"deformed_right"}}}, sort_keys=True), encoding="utf-8")

def _validate(values: Mapping[str, Any], staging: Path) -> Sequence[ProducedArtifact]:
    _field_provenance(staging); required = _out(values)
    for item in required:
        path = require_path_within(staging / item.path, staging, require_exists=True)
        if not path.is_file() or not path.stat().st_size: raise ValueError(f"Stereo required artifact missing: {item.path}")
    for field in _FIELDS: _npz(staging / f"scientific/disp/{field}.npz", {"coordinates","displacement","iterations","final_loss","training_history","training_history_columns","training_history_schema_version"}, {"coordinates","displacement","iterations","final_loss","training_history"})
    for name in ("initial", "last"): _npz(staging / f"scientific/reconstruct/{name}.npz", {"left_coordinates","right_coordinates","points","valid","reprojection_error"}, {"left_coordinates","right_coordinates","points","reprojection_error"})
    _npz(staging / "scientific/deformation/initial_to_last.npz", {"coordinates","reference_points","current_points","displacement","strain","strain_components","valid"}, {"coordinates","reference_points","current_points","displacement","strain"})
    _npz(staging / "scientific/diagnostics/stereo_geometry.npz", {"schema_version","reason_code","reason_names","valid","reference_reprojection_error","current_reprojection_error","reference_positive_depth","current_positive_depth"}, {"reason_code","reference_reprojection_error","current_reprojection_error"})
    provenance = json.loads((staging / "scientific/diagnostics/field_provenance.json").read_text())
    expected = {"reference_disparity":{"reference":"reference_left","target":"reference_right"}, "left_temporal":{"reference":"reference_left","target":"deformed_left"}, "deformed_disparity":{"reference":"reference_left","target":"deformed_right"}}
    if provenance.get("schema_version") != "neurodic.stereo.fields/v1" or provenance.get("fields") != expected: raise ValueError("Stereo field provenance is invalid")
    if values.get("evaluation", {}).get("enabled", False):
        for field in _FIELDS:
            _npz(staging / f"scientific/disp/{field}_evaluation.npz", {"schema_version","indices","residual"}, {"indices"})
            summary = json.loads((staging / f"scientific/disp/{field}_evaluation.json").read_text())
            if summary.get("schema_version") != "neurodic.fixed_evaluation/v1" or summary.get("scope", {}).get("field") != field: raise ValueError("Stereo evaluation field identity is invalid")
    return required

def _run(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    frozen = _freeze(values, scope); _root, _inputs = _paths(frozen); overlay = _overlay(frozen, staging)
    from ...api.pin_stereo_dic import pin_stereo_dic
    pin_stereo_dic(overlay)
    return _validate(frozen, staging)

def _inputs(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze(values, plan.get("scope", {})); root, paths = _paths(frozen)
    return {"baseline_config":plan["baseline"]["effective_config_identity"], **{role:{"path":str(path.relative_to(root)), **content_identity(path).to_dict()} for role,path in paths.items()}}

def guarded_stereo_action() -> TrustedAction:
    return TrustedAction("pin_stereo.combined_solver_call", _run, "neurodic.stereo.full_solve/v1", output_contract="neurodic.stereo.full-solve-artifacts/v1", input_identities=_inputs)
