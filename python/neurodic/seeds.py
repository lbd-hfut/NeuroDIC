"""PIN 2D seed initialization wrappers and case runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_config
from .runtime import configure_runtime

try:
    from . import _neurodic
except ImportError:  # pragma: no cover - import-time guard
    _neurodic = None


def _require_backend():
    if _neurodic is None:
        raise ImportError("neurodic C++ initialization backend is not available")
    return _neurodic


def _set_options(options: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(options, key):
            setattr(options, key, value)


def _common_options(initialization: Mapping[str, Any], options: Any) -> None:
    options.target_seed_count = int(initialization.get("target_seed_count", options.target_seed_count))
    cleanup = initialization.get("outlier_rejection", {})
    if hasattr(options, "cleanup"):
        options.cleanup.mad_threshold = float(cleanup.get("threshold", options.cleanup.mad_threshold))
        options.cleanup.min_seed_count = int(cleanup.get("min_seed_count", options.cleanup.min_seed_count))
    else:
        options.mad_threshold = float(cleanup.get("threshold", options.mad_threshold))
        options.min_seeds_per_roi = int(cleanup.get("min_seed_count", options.min_seeds_per_roi))


def initialize_seeds(reference: np.ndarray, deformed: np.ndarray, roi_mask: np.ndarray,
                     config: Mapping[str, Any]) -> dict[str, np.ndarray | str]:
    """Run one configured seed strategy and return original-image seed tensors."""
    configure_runtime(config)
    backend = _require_backend()
    import torch

    initialization = config.get("initialization", config)
    strategy = str(initialization.get("strategy", "integer_search"))
    ref = torch.as_tensor(np.ascontiguousarray(reference), dtype=torch.float32)
    deformed_tensor = torch.as_tensor(np.ascontiguousarray(deformed), dtype=torch.float32)
    mask = torch.as_tensor(np.ascontiguousarray(roi_mask.astype(bool)), dtype=torch.bool)
    if strategy == "integer_search":
        options = backend.TraditionalSeedOptions()
        _common_options(initialization, options)
        branch = initialization.get("integer_search", {})
        _set_options(options, {k: v for k, v in branch.items() if k in {"kmeans_iterations", "kmeans_sample_limit"}})
        search = branch.get("search", {})
        _set_options(options, search)
        prior = branch.get("sift_prior", {})
        _set_options(options, {f"sift_{k}": v for k, v in prior.items() if k != "enabled"})
        options.sift_prior_enabled = bool(prior.get("enabled", options.sift_prior_enabled))
        subpixel = branch.get("subpixel", {})
        _set_options(options, {f"subpixel_{k}": v for k, v in subpixel.items()})
        seeds = backend.TraditionalSeedInitializer(options).initialize(ref, deformed_tensor, mask)
    elif strategy == "sift_search":
        options = backend.SiftGridSeedOptions()
        _common_options(initialization, options)
        _set_options(options, initialization.get("sift_search", {}))
        seeds = backend.SiftGridSeedInitializer(options).initialize(ref, deformed_tensor, mask)
    else:
        raise ValueError(f"Unsupported initialization.strategy: {strategy}")
    return {
        "strategy": strategy,
        "seed_pos": seeds.seed_pos.detach().cpu().numpy(),
        "seed_uv": seeds.seed_uv.detach().cpu().numpy(),
        "scale_uv": seeds.scale_uv.detach().cpu().numpy(),
    }


def run_seed_case(case_root: str | Path, strategy: str, config_path: str | Path) -> dict[str, Any]:
    """Generate seeds, data files, and a correspondence visualization for one 2D case."""
    import cv2

    root = Path(case_root).resolve()
    config = load_config(config_path)
    config = dict(config)
    config["initialization"] = dict(config.get("initialization", {}), strategy=strategy)
    reference = cv2.imread(str(root / "001.bmp"), cv2.IMREAD_GRAYSCALE)
    deformed = cv2.imread(str(root / "002.bmp"), cv2.IMREAD_GRAYSCALE)
    roi_image = cv2.imread(str(root / "003.bmp"), cv2.IMREAD_GRAYSCALE)
    if reference is None or deformed is None or roi_image is None:
        raise FileNotFoundError(f"Expected 001.bmp, 002.bmp, and ROI 003.bmp in {root}")
    if roi_image.shape != reference.shape or deformed.shape != reference.shape:
        raise ValueError("Reference, deformed, and ROI images must have the same shape")
    result = initialize_seeds(reference, deformed, roi_image != 0, config)
    result_dir = root / "result" / "seeds"
    visualization_dir = root / "visualization" / "seeds"
    result_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{strategy}_seeds"
    np.savez_compressed(result_dir / f"{stem}.npz", seed_pos=result["seed_pos"], seed_uv=result["seed_uv"], scale_uv=result["scale_uv"])
    summary = {
        "strategy": strategy,
        "reference_image": "001.bmp",
        "deformed_image": "002.bmp",
        "roi_image": "003.bmp",
        "seed_count": int(len(result["seed_pos"])),
        "uvmean": [float(v) for v in result["scale_uv"][:2]],
        "uvscale": [float(v) for v in result["scale_uv"][2:]],
    }
    (result_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from .visualization.seeds import visualize_seed_matches

    image_path = visualize_seed_matches(reference, deformed, result["seed_pos"], result["seed_uv"],
                                        visualization_dir / f"{stem}.png", strategy)
    summary["result_file"] = str(result_dir / f"{stem}.npz")
    summary["visualization_file"] = str(image_path)
    return summary
