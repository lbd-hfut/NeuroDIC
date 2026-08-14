"""Trusted staging-only adapter for the coarse planar PIN public API.

The native PIN entry point is a single solve call.  It does not expose safe
stage-selective initialization, training, inference, or evaluation calls, so
this adapter intentionally publishes only their combined artifact set.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...case_io import planar_image_series
from ..artifacts import content_identity, require_path_within
from ..execution import ProducedArtifact, TrustedAction


_RESULT = "scientific/pin_result.npz"
_TRAINING = "scientific/diagnostics_training.npz"
_EVALUATION_NPZ = "scientific/diagnostics_evaluation.npz"
_EVALUATION_JSON = "scientific/diagnostics_evaluation.json"
_VISUALIZATION = "visualization/pin_displacement.png"


def _case_paths(values: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    case = values.get("case", {})
    root = Path(case["root"]).resolve()
    image_root = require_path_within(root / str(case.get("images_dir", ".")), root, require_exists=True)
    reference, deformed, roi = planar_image_series(image_root)
    frame = int(case["frame"])
    try:
        current = deformed[frame]
    except IndexError as error:
        raise ValueError(f"Frozen PIN case.frame {frame} is outside the available deformed frames") from error
    return (root, require_path_within(reference, root, require_exists=True),
            require_path_within(current, root, require_exists=True), require_path_within(roi, root, require_exists=True))


def _freeze_scope(values: Mapping[str, Any], scope: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the already-approved input selection without an implicit default."""
    selected = scope.get("selected_frame")
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 0:
        raise ValueError("PIN execution requires a validated scope.selected_frame")
    frozen = copy.deepcopy(dict(values))
    frozen.setdefault("case", {})["frame"] = selected
    return frozen


def _execution_overlay(values: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    """Output-only overlay; it is intentionally excluded from the signature."""
    overlay = copy.deepcopy(dict(values))
    root = require_path_within(staging, staging, require_exists=True)
    overlay["output"] = {"result": str(root / "scientific"), "visualization": str(root / "visualization")}
    return overlay


def _required_outputs(values: Mapping[str, Any]) -> list[ProducedArtifact]:
    outputs = [ProducedArtifact(_RESULT, "pin_result", "neurodic.pin.result/v1"),
               ProducedArtifact(_TRAINING, "training_diagnostics", "neurodic.pin.training/v1")]
    if bool(values.get("evaluation", {}).get("enabled", False)):
        outputs += [ProducedArtifact(_EVALUATION_NPZ, "evaluation_diagnostics", "neurodic.fixed_evaluation/v1"),
                    ProducedArtifact(_EVALUATION_JSON, "evaluation_summary", "neurodic.fixed_evaluation/v1")]
    return outputs


def _validate_npz(path: Path, required: set[str], *, finite: set[str]) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"PIN artifact {path.name} lacks required keys: {sorted(missing)}")
        for key in finite:
            array = np.asarray(data[key])
            if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                raise ValueError(f"PIN artifact {path.name} has non-finite required field: {key}")


def _validate_outputs(values: Mapping[str, Any], staging: Path) -> Sequence[ProducedArtifact]:
    required = _required_outputs(values)
    for artifact in required:
        path = require_path_within(staging / artifact.path, staging, require_exists=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"PIN required artifact is absent or empty: {artifact.path}")
    _validate_npz(staging / _RESULT, {"coordinates", "displacement", "strain", "strain_components", "iterations", "final_loss"},
                  finite={"coordinates", "displacement", "strain", "iterations", "final_loss"})
    _validate_npz(staging / _TRAINING, {"schema_version", "history", "history_columns", "phase_names"}, finite={"history"})
    if bool(values.get("evaluation", {}).get("enabled", False)):
        _validate_npz(staging / _EVALUATION_NPZ, {"schema_version", "indices", "residual"}, finite={"indices"})
        summary = json.loads((staging / _EVALUATION_JSON).read_text(encoding="utf-8"))
        if summary.get("schema_version") != "neurodic.fixed_evaluation/v1" or summary.get("solver") != "pin":
            raise ValueError("PIN evaluation summary does not satisfy the fixed-evaluation contract")
        if not {"evaluation_set", "loss", "valid_count", "valid_ratio", "summary"}.issubset(summary):
            raise ValueError("PIN evaluation summary lacks required fixed-evaluation fields")
    visual = staging / _VISUALIZATION
    if visual.is_file() and visual.stat().st_size:
        return [*required, ProducedArtifact(_VISUALIZATION, "displacement_visualization", "image/png")]
    return required


def _run_pin(values: Mapping[str, Any], staging: Path, scope: Mapping[str, Any]) -> Sequence[ProducedArtifact]:
    """Call the fixed public API once, with every writable path under staging."""
    frozen = _freeze_scope(values, scope)
    root, _reference, _deformed, _roi = _case_paths(frozen)
    overlay = _execution_overlay(frozen, staging)
    # Import only inside the approved execution action: module import stays
    # native-free for planning, inspection, and test collection.
    from ...api.pin_dic import pin_dic
    pin_dic(root, config=overlay)
    return _validate_outputs(values, staging)


def _input_identities(plan: Mapping[str, Any], values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Full content identities for exactly the images consumed by this call."""
    frozen = _freeze_scope(values, plan.get("scope", {}))
    root, reference, deformed, roi = _case_paths(frozen)
    return {"baseline_config": plan["baseline"]["effective_config_identity"],
            "reference_image": {"path": str(reference.relative_to(root)), **content_identity(reference).to_dict()},
            "deformed_image": {"path": str(deformed.relative_to(root)), **content_identity(deformed).to_dict()},
            "roi_mask": {"path": str(roi.relative_to(root)), **content_identity(roi).to_dict()}}


def guarded_pin_action() -> TrustedAction:
    return TrustedAction("pin.combined_solver_call", _run_pin, "neurodic.pin.full_solve/v1",
                         output_contract="neurodic.pin.full-solve-artifacts/v1", input_identities=_input_identities)
