"""High-level dispatch for the Traditional-DIC calibration port."""

from .. import calibration as _calibration


def calibrate(mode: str, *args, **kwargs):
    """Dispatch mono, stereo, or COLMAP-like multiview calibration."""
    normalized = mode.lower()
    if normalized == "mono":
        return _calibration.calibrate_mono_zhang(*args, **kwargs)
    if normalized == "stereo":
        return _calibration.calibrate_stereo_zhang(*args, **kwargs)
    if normalized in {"colmap", "multiview", "self_calibration"}:
        return _calibration.calibrate_multiview_colmap_like(*args, **kwargs)
    raise ValueError(f"Unsupported calibration mode: {mode}")
