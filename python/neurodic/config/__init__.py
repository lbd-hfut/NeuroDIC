"""Configuration loading helpers for the thin Python API."""

from pathlib import Path
from typing import Any
import copy
import json


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one Traditional-DIC-compatible YAML mapping."""
    import yaml

    raw = Path(path).read_text(encoding="utf-8")
    # JSON is a supported YAML subset, but PyYAML's YAML 1.1 scalar resolver
    # reads a JSON exponent such as ``1e-06`` as a string.  Prefer the JSON
    # grammar whenever the document is JSON so numeric JSON values remain
    # numeric through configuration resolution and into native bindings.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively, with case values taking precedence."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_case_config(solver_config: str | Path, case_name: str,
                     case_paths: str | Path = "config/case_paths.yaml") -> dict[str, Any]:
    """Combine a path-free solver YAML with one named case-path mapping.

    ``case_paths`` owns all file-system locations and image ordering.  Keeping
    this composition explicit makes the same numerical configuration reusable
    for another dataset without editing its solver parameters.
    """
    paths = load_config(case_paths)
    if case_name not in paths:
        choices = ", ".join(sorted(paths))
        raise ValueError(f"Unknown case path mapping {case_name!r}; available: {choices}")
    selected = paths[case_name]
    if not isinstance(selected, dict):
        raise ValueError(f"case_paths entry must be a mapping: {case_name}")
    return _deep_merge(load_config(solver_config), selected)
