"""High-level NeuroDIC Python API, loaded lazily to preserve metadata-only use."""

from importlib import import_module

_EXPORTS = {
    "calibrate": (".calibrate", "calibrate"), "ndef_dic": (".ndef_dic", "ndef_dic"),
    "ndef_sparse_precalculation": (".ndef_dic", "ndef_sparse_precalculation"),
    "pretrain_ndef_surface": (".ndef_surface", "pretrain_ndef_surface"),
    "pin_dic": (".pin_dic", "pin_dic"), "run_planar_case": (".pin_dic", "run_planar_case"),
    "pin_multi_slover_dic": (".pin_multi_slover_dic", "pin_multi_slover_dic"),
    "run_pin_multi_case": (".pin_multi_slover_dic", "run_pin_multi_case"),
    "pin_stereo_dic": (".pin_stereo_dic", "pin_stereo_dic"), "run_stereo_case": (".pin_stereo_dic", "run_stereo_case"),
}


def __getattr__(name: str):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

__all__ = ["calibrate", "ndef_dic", "ndef_sparse_precalculation", "pretrain_ndef_surface", "pin_dic", "run_planar_case",
           "pin_multi_slover_dic", "run_pin_multi_case", "pin_stereo_dic", "run_stereo_case"]
