"""Reserved high-level entry point for pairwise multi-camera PIN-DIC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def pin_multi_slover_dic(
    config: str | Path | Mapping[str, Any] = "config/pin_multi_slover.yaml",
    *,
    write_case_artifacts: bool = True,
) -> None:
    """Reserve the multi-pair workflow without touching existing routes."""

    del config, write_case_artifacts
    raise NotImplementedError(
        "pin_multi_slover is not wired into the runtime yet; see PIN_MULTI_SLOVER_EXECUTION_PLAN.md"
    )
