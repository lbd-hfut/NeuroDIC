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

def native_available() -> bool:
    """Return whether the compiled C++ extension is importable."""
    return _neurodic is not None


_LAZY_EXPORTS = {
    "calibrate": (".api", "calibrate"), "ndef_dic": (".api", "ndef_dic"),
    "ndef_sparse_precalculation": (".api", "ndef_sparse_precalculation"),
    "pretrain_ndef_surface": (".api", "pretrain_ndef_surface"), "pin_dic": (".api", "pin_dic"),
    "run_planar_case": (".api", "run_planar_case"), "pin_multi_slover_dic": (".api", "pin_multi_slover_dic"),
    "pin_stereo_dic": (".api", "pin_stereo_dic"), "run_pin_multi_case": (".api", "run_pin_multi_case"),
    "run_stereo_case": (".api", "run_stereo_case"), "NDeFROIOptions": (".ndef_roi", "NDeFROIOptions"),
    "generate_ndef_roi": (".ndef_roi", "generate_ndef_roi"), "inspect_ndef_preflight": (".ndef_preflight", "inspect_ndef_preflight"),
    "make_ndef_run_mapping": (".ndef_paths", "make_ndef_run_mapping"), "PINMultiFusionOptions": (".pin_multi_fusion", "PINMultiFusionOptions"),
    "fuse_pin_multi_surfaces": (".pin_multi_fusion", "fuse_pin_multi_surfaces"), "PINMultiPairROIOptions": (".pin_multi_roi", "PINMultiPairROIOptions"),
    "pin_multi_pair_roi": (".pin_multi_roi", "pin_multi_pair_roi"), "configure_runtime": (".runtime", "configure_runtime"),
    "load_case_config": (".config", "load_case_config"),
}


def __getattr__(name: str):
    """Keep metadata-only imports usable without the optional native extension."""
    if name in {"calibration", "seeds", "models"}:
        from importlib import import_module
        value = import_module(f".{name}", __name__)
    elif name in _LAZY_EXPORTS:
        from importlib import import_module
        module, attribute = _LAZY_EXPORTS[name]
        if name == "calibrate":
            # ``calibration.py`` necessarily touches native bindings at import
            # time. Keep the public callable visible for metadata-only clients;
            # loading remains deferred until a calibration operation is invoked.
            def value(*args, **kwargs):
                return getattr(import_module(module, __name__), attribute)(*args, **kwargs)
        else:
            value = getattr(import_module(module, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = ["calibrate", "calibration", "models", "seeds", "ndef_dic", "ndef_sparse_precalculation", "pretrain_ndef_surface",
           "NDeFROIOptions", "generate_ndef_roi", "inspect_ndef_preflight", "make_ndef_run_mapping", "pin_dic", "run_planar_case", "pin_multi_slover_dic", "run_pin_multi_case", "run_stereo_case",
           "PINMultiPairROIOptions", "pin_multi_pair_roi", "PINMultiFusionOptions",
           "fuse_pin_multi_surfaces", "pin_stereo_dic", "configure_runtime", "load_case_config", "native_available"]
