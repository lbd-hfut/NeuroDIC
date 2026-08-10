"""Thin Python package for NeuroDIC.

Responsibilities: expose high-level user API and compiled binding loader.
Inputs: user-facing Python arguments.
Outputs: calls into neurodic._neurodic when available.
Dependencies: compiled pybind11 extension. TODO: keep scientific kernels in C++.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    from . import _neurodic
except ImportError:
    _neurodic = None

from .api import (calibrate, ndef_dic, ndef_sparse_precalculation, pretrain_ndef_surface, pin_dic,
                  pin_multi_slover_dic, pin_stereo_dic)
from . import calibration
from .ndef_roi import NDeFROIOptions, generate_ndef_roi
from .ndef_preflight import inspect_ndef_preflight
from .ndef_paths import make_ndef_run_mapping
from .pin_multi_fusion import PINMultiFusionOptions, fuse_pin_multi_surfaces
from .pin_multi_roi import PINMultiPairROIOptions, pin_multi_pair_roi
from . import seeds
from . import models
from .runtime import configure_runtime


def native_available() -> bool:
    """Return whether the compiled C++ extension is importable."""
    return _neurodic is not None


__all__ = ["calibrate", "calibration", "models", "seeds", "ndef_dic", "ndef_sparse_precalculation", "pretrain_ndef_surface",
           "NDeFROIOptions", "generate_ndef_roi", "inspect_ndef_preflight", "make_ndef_run_mapping", "pin_dic", "pin_multi_slover_dic",
           "PINMultiPairROIOptions", "pin_multi_pair_roi", "PINMultiFusionOptions",
           "fuse_pin_multi_surfaces", "pin_stereo_dic", "configure_runtime", "native_available"]
