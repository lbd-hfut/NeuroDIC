"""Guarded, CPU-only PIN Multi pair-ROI execution adapter."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ...case_io import image_files
from ...pin_multi_roi import (_options_from_config, generate_pin_multi_pair_roi,
                               pair_id_for, select_pin_multi_pairs)
from ..artifacts import require_path_within
from ..artifacts import content_identity
from ..execution import TrustedAction


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
