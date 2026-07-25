"""NDeF-DIC high-level Python entry point.

Responsibilities: future ergonomic wrapper for C++ NDeFSolver.
Inputs: multi-view observations, calibration result, ROI/surface config.
Outputs: NDeF result from compiled backend.
Dependencies: neurodic._neurodic. TODO: keep NDeF topology internal.
"""


def ndef_dic(*args, **kwargs):
    """Run NDeF-DIC through the compiled backend once implemented."""
    raise NotImplementedError("TODO(NeuroDIC): call C++ NDeFSolver through neurodic._neurodic")
