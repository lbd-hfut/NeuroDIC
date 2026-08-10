"""Resolve isolated NDeF run directories from a solver mapping.

Calibration remains a case-shared product under ``result/calibration``. All
other NDeF products can live below an explicit namespace, independent of PIN.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def camera_name_from_label(label: str, fallback: str) -> str:
    """Return the camera-directory name from either a name or an image path."""
    path = Path(str(label))
    if path.suffix and path.parent.name:
        return path.parent.name
    return path.name or fallback


def ndef_run_roots(root: str | Path, values: Mapping[str, Any]) -> tuple[Path, Path]:
    """Return numerical and visualization roots for this NDeF run.

    ``output.ndef_subdir`` defaults to the historic ``ndef`` directory. A
    temporary mapping may set it to ``ndef_multi_slover`` without changing the
    public YAML configuration.
    """
    # Public configs commonly use a repository-relative case root. Return
    # absolute roots so callers can safely hand these paths to APIs that also
    # accept relative paths with respect to the case root.
    root = Path(root).resolve()
    output = values.get("output", {})
    result = _resolve(root, output.get("result", "result"))
    visualization = _resolve(root, output.get("visualization", "visualization"))
    subdir = output.get("ndef_subdir", "ndef")
    if subdir is None or str(subdir).strip() in {"", "."}:
        return result, visualization
    relative = Path(str(subdir))
    if relative.is_absolute():
        raise ValueError("output.ndef_subdir must be a relative path")
    return result / relative, visualization / relative


def make_ndef_run_mapping(config: Mapping[str, Any], case_root: str | Path,
                          *, namespace: str = "ndef_multi_slover") -> dict[str, Any]:
    """Copy a public NDeF config into an isolated case-local run mapping.

    This is intentionally an in-memory transformation: calling code can test a
    different case without modifying the shared example YAML. Calibration is
    left at the case-shared ``result/calibration`` location; all NDeF products
    are redirected to ``result/<namespace>`` and ``visualization/<namespace>``.
    """
    values = copy.deepcopy(dict(config))
    case = values.setdefault("case", {})
    case["root"] = str(case_root)
    case["calibration"] = "result/calibration/calibration_result_scaled.json"
    case["masks"] = f"result/{namespace}/roi/per_camera"
    case["reference_surface"] = f"result/{namespace}/surface/deformation_surface_dataset.npz"
    precalculation = values.setdefault("precalculation", {})
    precalculation["displacement"] = f"result/{namespace}/precalculation/sparse_tracks.npz"
    output = values.setdefault("output", {})
    output["result"] = "result"
    output["visualization"] = "visualization"
    output["ndef_subdir"] = namespace
    return values
