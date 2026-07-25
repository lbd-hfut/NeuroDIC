"""Calibration high-level Python entry point.

Responsibilities: future wrapper for C++ calibration adapters.
Inputs: calibration type and input paths/options.
Outputs: CalibrationResult from compiled backend.
Dependencies: neurodic._neurodic. TODO: add typed request objects.
"""


def calibrate(*args, **kwargs):
    """Run calibration through the compiled backend once implemented."""
    raise NotImplementedError("TODO(NeuroDIC): call C++ CalibrationManager through neurodic._neurodic")
