"""Reproducible process-wide random-state setup for NeuroDIC runs."""

from __future__ import annotations

import os
import random
from typing import Any, Mapping

import numpy as np
import torch

from .models import _require_backend


def configure_runtime(config: Mapping[str, Any]) -> int | None:
    """Apply ``runtime.random_seed`` once before preprocessing or model creation.

    A missing seed leaves the caller's random state untouched. With a seed, Python,
    NumPy, Torch CPU/CUDA, OpenCV (through the C++ core), and deterministic Torch
    operators are configured consistently.
    """
    runtime = config.get("runtime", {})
    value = runtime.get("random_seed")
    if value is None:
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError("runtime.random_seed must be nonnegative")
    # Must be set before CUDA handles are initialized; do not replace a user choice.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(runtime.get("deterministic", True)), warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(runtime.get("deterministic", True))
    _require_backend().set_random_seed(seed)
    return seed
