"""Reserved SIFT-pair ROI contract for the ``pin_multi_slover`` route.

This module must remain independent of :mod:`neurodic.ndef_roi`.  Its future
input is the two reference images of one selected camera pair, not calibrated
multi-view tracks or an NDeF reference surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PINMultiPairROIOptions:
    """Configuration contract for reference-time pairwise SIFT ROI generation."""

    feature_method: Literal["sift"] = "sift"
    max_features: int = 12_000
    match_ratio: float = 0.75
    mutual_check: bool = True
    ransac_reprojection_threshold_px: float = 3.0
    min_matches: int = 20
    support: Literal["convex_hull", "alpha_shape"] = "convex_hull"
    alpha_radius_scale: float = 8.0
    erode_pixels: int = 0


def generate_pin_multi_pair_roi(
    left_reference: str | Path,
    right_reference: str | Path,
    output_dir: str | Path,
    *,
    options: PINMultiPairROIOptions | None = None,
) -> None:
    """Reserve the pair ROI generator; implementation is intentionally pending.

    The implementation must match SIFT only between ``left_reference(t0)`` and
    ``right_reference(t0)``, robustly retain geometric inliers, form the left
    image support mask, and save pair-local masks/diagnostics under
    ``output_dir``.  It must not consume or overwrite NDeF masks.
    """

    del left_reference, right_reference, output_dir, options
    raise NotImplementedError(
        "pin_multi_slover pairwise SIFT ROI generation is a placeholder; see PIN_MULTI_SLOVER_EXECUTION_PLAN.md"
    )
