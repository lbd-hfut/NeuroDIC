"""Thin Python package for NeuroDIC.

Responsibilities: expose high-level user API and compiled binding loader.
Inputs: user-facing Python arguments.
Outputs: calls into neurodic._neurodic when available.
Dependencies: compiled pybind11 extension. TODO: keep scientific kernels in C++.
"""

from pkgutil import extend_path

from .api import calibrate, ndef_dic, pin_dic

__path__ = extend_path(__path__, __name__)

try:
    from . import _neurodic
except ImportError:
    _neurodic = None


def native_available() -> bool:
    """Return whether the compiled C++ extension is importable."""
    return _neurodic is not None


__all__ = ["calibrate", "ndef_dic", "pin_dic", "native_available"]
