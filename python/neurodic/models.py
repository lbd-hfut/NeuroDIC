"""Thin construction helpers for compiled NeuroDIC neural models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import load_config

try:
    from . import _neurodic
except ImportError:  # pragma: no cover - import-time guard
    _neurodic = None


def _require_backend():
    if _neurodic is None:
        raise ImportError("neurodic C++ model backend is not available")
    return _neurodic


def _config_data(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, (str, Path)):
        return load_config(config)
    return config


def make_pin_model(config: str | Path | Mapping[str, Any]):
    """Construct the compiled PIN MLP from a ``pin_2d.yaml`` mapping.

    Training remains owned by the future C++ PINSolver; this helper is for
    assembly and forward-contract validation only.
    """
    backend = _require_backend()
    model_config = dict(_config_data(config).get("model", {}))
    if model_config.get("type", "mlp") != "mlp":
        raise ValueError("PIN currently supports only model.type='mlp'")
    options = backend.PINModelOptions()
    options.hidden_dim = int(model_config.get("hidden_dim", options.hidden_dim))
    options.hidden_layers = int(model_config.get("hidden_layers", options.hidden_layers))
    encoding = model_config.get("fourier_encoding", {})
    for key in ("enabled", "num_frequencies", "include_input", "angular_scale"):
        if key in encoding:
            setattr(options.fourier_encoding, key, encoding[key])
    return backend.PINMLPModel(options)
