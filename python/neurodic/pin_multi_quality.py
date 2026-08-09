"""Per-point quality reason codes for pairwise PIN-DIC products.

Each reconstructed pair keeps its own quality audit: per-point reason codes,
reprojection-error statistics, and per-field PIN diagnostics.  Reason codes
protect the later fusion stage from mixing low-quality pair points.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

REASON_CODES: dict[int, str] = {
    0: "valid",
    1: "invalid_field",
    2: "outside_roi",
    3: "out_of_bounds",
    4: "negative_depth",
    5: "reprojection_error",
}

REASON_INDEX = {name: code for code, name in REASON_CODES.items()}


def compute_pair_reason_codes(
    pair_result,
    roi_mask: np.ndarray | None = None,
    *,
    max_reprojection_error_px: float = 5.0,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point reason codes for one pair's stereo reconstruction.

    Returns ``(codes, reprojection_error)`` where ``codes`` is an int8 array
    over the reconstructed points.  Reason priority: invalid field, outside
    ROI, outside image bounds, non-positive depth, reprojection error.
    """
    result = pair_result.result if hasattr(pair_result, "result") else pair_result
    count = int(result.valid.numel())
    codes = np.zeros(count, dtype=np.int8)

    field_arrays = [
        result.reference_disparity.displacement.values.numpy(),
        result.left_temporal.displacement.values.numpy(),
        result.deformed_disparity.displacement.values.numpy(),
        result.reference_points.numpy(),
        result.current_points.numpy(),
        result.displacement_3d.numpy(),
    ]
    invalid = np.zeros(count, dtype=bool)
    for values in field_arrays:
        invalid |= ~np.isfinite(values).all(axis=1)
    codes[invalid] = REASON_INDEX["invalid_field"]

    coordinates = result.left_reference_coordinates.numpy()
    if roi_mask is not None:
        mask = np.asarray(roi_mask, dtype=bool)
        height, width = mask.shape
        xy = np.round(coordinates).astype(np.int64)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        inside[inside] = mask[xy[inside, 1], xy[inside, 0]] > 0
        codes[(codes == 0) & ~inside] = REASON_INDEX["outside_roi"]

    if image_size is not None:
        width, height = image_size
        coordinate_arrays = [
            result.left_reference_coordinates.numpy(),
            result.left_current_coordinates.numpy(),
            result.right_reference_coordinates.numpy(),
            result.right_current_coordinates.numpy(),
        ]
        in_bounds = np.ones(count, dtype=bool)
        for values in coordinate_arrays:
            in_bounds &= np.isfinite(values).all(axis=1)
            in_bounds &= (values[:, 0] >= 0.0) & (values[:, 0] <= width - 1.0)
            in_bounds &= (values[:, 1] >= 0.0) & (values[:, 1] <= height - 1.0)
        codes[(codes == 0) & ~in_bounds] = REASON_INDEX["out_of_bounds"]

    reference_points = result.reference_points.numpy()
    current_points = result.current_points.numpy()
    positive_depth = (reference_points[:, 2] > 0.0) & (current_points[:, 2] > 0.0)
    codes[(codes == 0) & ~positive_depth] = REASON_INDEX["negative_depth"]

    reprojection = np.maximum(result.reference_reprojection_error.numpy(),
                              result.current_reprojection_error.numpy())
    codes[(codes == 0) & (reprojection > max_reprojection_error_px)] = REASON_INDEX["reprojection_error"]
    return codes, reprojection


def pair_quality_summary(
    pair_result,
    roi_mask: np.ndarray | None = None,
    *,
    max_reprojection_error_px: float = 5.0,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Aggregated quality statistics for one pair, including reason counts."""
    result = pair_result.result if hasattr(pair_result, "result") else pair_result
    codes, reprojection = compute_pair_reason_codes(
        pair_result, roi_mask, max_reprojection_error_px=max_reprojection_error_px, image_size=image_size)
    counts = {name: int((codes == code).sum()) for code, name in REASON_CODES.items()}
    valid = codes == 0
    stats: dict[str, Any] = {
        "total_points": int(codes.size),
        "valid_points": int(counts["valid"]),
        "valid_ratio": float(counts["valid"] / codes.size) if codes.size else 0.0,
        "reason_codes": counts,
        "max_reprojection_error_px": float(max_reprojection_error_px),
    }
    if valid.any():
        stats["mean_reprojection_error_px"] = float(reprojection[valid].mean())
        stats["p95_reprojection_error_px"] = float(np.percentile(reprojection[valid], 95))
    else:
        stats["mean_reprojection_error_px"] = None
        stats["p95_reprojection_error_px"] = None
    stats["pin_diagnostics"] = {
        field_name: {
            "final_loss": float(getattr(result, field_name).diagnostics.final_loss),
            "iterations": int(getattr(result, field_name).diagnostics.iterations),
        }
        for field_name in ("reference_disparity", "left_temporal", "deformed_disparity")
    }
    return stats
