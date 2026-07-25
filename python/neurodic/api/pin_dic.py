"""PIN-DIC high-level Python entry point.

Responsibilities: future ergonomic wrapper for C++ PINSolver.
Inputs: images, ROI, calibration result, and config.
Outputs: PIN result from compiled backend.
Dependencies: neurodic._neurodic. TODO: wire after C++ solver is validated.
"""


def pin_dic(*args, **kwargs):
    """Run PIN-DIC through the compiled backend once implemented."""
    raise NotImplementedError("TODO(NeuroDIC): call C++ PINSolver through neurodic._neurodic")
