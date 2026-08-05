"""Configuration loading helpers for the thin Python API."""

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one Traditional-DIC-compatible YAML mapping."""
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data
