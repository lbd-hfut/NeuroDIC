"""Thin YAML/image assembly for the compiled planar PIN-DIC solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..config import load_config
from ..models import _require_backend
from ..seeds import initialize_seeds


def _mapping(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else config


def _model_options(backend, config: Mapping[str, Any]):
    values = config.get("model", {})
    if values.get("type", "mlp") != "mlp":
        raise ValueError("PIN currently supports only model.type='mlp'")
    options = backend.PINModelOptions()
    options.hidden_dim = int(values.get("hidden_dim", options.hidden_dim))
    options.hidden_layers = int(values.get("hidden_layers", options.hidden_layers))
    for key, value in values.get("fourier_encoding", {}).items():
        if key in {"enabled", "num_frequencies", "include_input", "angular_scale"}:
            setattr(options.fourier_encoding, key, value)
    return options


def _precompute_options(backend, config: Mapping[str, Any]):
    options = backend.ImagePrecomputeOptions()
    interpolation = config.get("interpolation", {})
    options.bspline_degree = int(interpolation.get("degree", options.bspline_degree))
    seed = config.get("initialization", {}).get("integer_search", {})
    search = seed.get("search", {})
    options.integer_search_radius = int(search.get("search_radius", 0))
    options.subset_radius = int(seed.get("subpixel", {}).get("subset_radius", 0))
    return options


def _apply_training(problem, config: Mapping[str, Any]) -> None:
    for key in ("seed_iterations", "photometric_iterations", "photometric_sample_count",
                "photometric_sampling_enabled", "znssd_kernel_size", "seed_learning_rate",
                "photometric_learning_rate"):
        if key in config.get("training", {}):
            setattr(problem, key, config["training"][key])
    loss_name = str(config.get("training", {}).get("photometric_loss", "znssd")).lower()
    if loss_name == "ssd":
        problem.photometric_loss = _require_backend().PhotometricLossType.SSD
    elif loss_name == "znssd":
        problem.photometric_loss = _require_backend().PhotometricLossType.ZNSSD
    else:
        raise ValueError("training.photometric_loss must be 'ssd' or 'znssd'")
    device = config.get("training", {}).get("device", "cpu")
    problem.set_device(str(device))


def _write_case_artifacts(case_root: Path, result, output_subdir: str | None) -> None:
    output = case_root / "result" / "pin"
    visual = case_root / "visualization" / "pin"
    if output_subdir is not None:
        name = Path(output_subdir)
        if name.is_absolute() or len(name.parts) != 1 or name.name in {"", ".", ".."}:
            raise ValueError("output_subdir must be one relative directory name")
        output /= name
        visual /= name
    output.mkdir(parents=True, exist_ok=True)
    visual.mkdir(parents=True, exist_ok=True)
    xy = result.displacement.coordinates.numpy()
    uv = result.displacement.values.numpy()
    np.savez(output / "pin_result.npz", coordinates=xy, displacement=uv,
             iterations=result.diagnostics.iterations, final_loss=result.diagnostics.final_loss)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, index, label in zip(axes, (0, 1), ("u displacement", "v displacement")):
        image = np.full((1280, 1280), np.nan, dtype=np.float32)
        image[xy[:, 1].astype(int), xy[:, 0].astype(int)] = uv[:, index]
        rendered = axis.imshow(image, cmap="turbo")
        axis.set_title(label)
        figure.colorbar(rendered, ax=axis)
    figure.savefig(visual / "pin_displacement.png", dpi=160)
    plt.close(figure)


def pin_dic(reference, deformed=None, roi_mask=None, config: str | Path | Mapping[str, Any] = "config/pin_2d.yaml",
            *, seeds=None, write_case_artifacts: bool = True, output_subdir: str | None = None):
    """Run the C++ planar PIN-DIC pipeline.

    ``reference`` may be a case directory containing ``001.bmp``, ``002.bmp``, and
    ``003.bmp``. Otherwise all three observations are NumPy arrays (or tensors).
    """
    backend = _require_backend()
    values = _mapping(config)
    case_root = Path(reference) if isinstance(reference, (str, Path)) and deformed is None else None
    if case_root is not None:
        import cv2

        reference = cv2.imread(str(case_root / "001.bmp"), cv2.IMREAD_GRAYSCALE)
        deformed = cv2.imread(str(case_root / "002.bmp"), cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.imread(str(case_root / "003.bmp"), cv2.IMREAD_GRAYSCALE) != 0
    if deformed is None or roi_mask is None:
        raise ValueError("deformed image and roi_mask are required unless reference is a case directory")
    reference_array = np.asarray(reference, dtype=np.float32)
    deformed_array = np.asarray(deformed, dtype=np.float32)
    mask_array = np.asarray(roi_mask, dtype=bool)
    if reference_array.shape != deformed_array.shape or reference_array.shape != mask_array.shape:
        raise ValueError("reference, deformed, and ROI mask must have matching 2D shapes")
    if seeds is None:
        seeds = initialize_seeds(reference_array, deformed_array, mask_array, values)
    if isinstance(seeds, Mapping):
        seeds = backend.make_seed_set(torch.as_tensor(seeds["seed_pos"], dtype=torch.float32),
                                      torch.as_tensor(seeds["seed_uv"], dtype=torch.float32))
    problem = backend.PINProblem(torch.from_numpy(reference_array), torch.from_numpy(deformed_array),
                                 torch.from_numpy(mask_array), seeds, _model_options(backend, values),
                                 _precompute_options(backend, values))
    _apply_training(problem, values)
    result = backend.PINSolver().solve(problem)
    if case_root is not None and write_case_artifacts:
        _write_case_artifacts(case_root, result, output_subdir)
    return result
