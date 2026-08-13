"""Thin YAML/image assembly for the compiled planar PIN-DIC solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..config import load_config
from ..case_io import planar_image_series
from ..models import _require_backend
from ..runtime import configure_runtime
from ..seeds import initialize_seeds
from ..visualization import Field2DPanel, render_2d_field_grid


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
    for key in ("seed_iterations", "seed_pretrain_uv_scale_threshold", "photometric_iterations", "photometric_sample_count",
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


def _write_case_artifacts(case_root: Path, result, output_subdir: str | None, reference_image: np.ndarray,
                          output_config: Mapping[str, Any] | None = None) -> None:
    output_config = output_config or {}
    output = Path(output_config.get("result", "result/pin"))
    visual = Path(output_config.get("visualization", "visualization/pin"))
    output = output if output.is_absolute() else case_root / output
    visual = visual if visual.is_absolute() else case_root / visual
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
    strain = result.strain.values.numpy()
    np.savez(output / "pin_result.npz", coordinates=xy, displacement=uv, strain=strain,
             strain_components=np.asarray(["E_xx", "E_yy", "E_xy"]),
             iterations=result.diagnostics.iterations, final_loss=result.diagnostics.final_loss)
    panels = [Field2DPanel(reference_image, xy, uv[:, index], label, label, symmetric=True)
              for index, label in enumerate(("u displacement", "v displacement"))]
    render_2d_field_grid(panels, visual / "pin_displacement.png", rows=1, columns=2, alpha=.9)


def _build_problem(reference_array, deformed_array, mask_array, values, seeds=None):
    """Assemble one planar problem; stereo orchestration reuses this thin helper."""
    backend = _require_backend()
    if seeds is None:
        seeds = initialize_seeds(reference_array, deformed_array, mask_array, values)
    if isinstance(seeds, Mapping):
        seeds = backend.make_seed_set(torch.as_tensor(seeds["seed_pos"], dtype=torch.float32),
                                      torch.as_tensor(seeds["seed_uv"], dtype=torch.float32))
    problem = backend.PINProblem(torch.from_numpy(np.asarray(reference_array, dtype=np.float32)),
                                 torch.from_numpy(np.asarray(deformed_array, dtype=np.float32)),
                                 torch.from_numpy(np.asarray(mask_array, dtype=bool)), seeds,
                                 _model_options(backend, values), _precompute_options(backend, values))
    _apply_training(problem, values)
    return problem


def pin_dic(reference, deformed=None, roi_mask=None, config: str | Path | Mapping[str, Any] = "config/pin_2d.yaml",
            *, seeds=None, write_case_artifacts: bool = True, output_subdir: str | None = None):
    """Run the C++ planar PIN-DIC pipeline.

    ``reference`` may be a case directory.  Its first sorted image is the
    reference, its final sorted image is the ROI, and ``case.frame`` selects
    one of the intervening deformed frames.  Otherwise all three observations
    are NumPy arrays (or tensors).
    """
    backend = _require_backend()
    values = _mapping(config)
    configure_runtime(values)
    case_root = Path(reference) if isinstance(reference, (str, Path)) and deformed is None else None
    if case_root is not None:
        import cv2

        case = values.get("case", {})
        reference_path, deformed_paths, roi_path = planar_image_series(case_root, case.get("images_dir", "."))
        frame = int(case.get("frame", 0))
        try:
            deformed_path = deformed_paths[frame]
        except IndexError as error:
            raise ValueError(f"case.frame {frame} is outside the {len(deformed_paths)} planar deformed frames") from error
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        deformed = cv2.imread(str(deformed_path), cv2.IMREAD_GRAYSCALE)
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if reference is None or deformed is None or roi is None:
            raise ValueError("Unable to read planar image sequence or ROI")
        roi_mask = roi != 0
    if deformed is None or roi_mask is None:
        raise ValueError("deformed image and roi_mask are required unless reference is a case directory")
    reference_array = np.asarray(reference, dtype=np.float32)
    deformed_array = np.asarray(deformed, dtype=np.float32)
    mask_array = np.asarray(roi_mask, dtype=bool)
    if reference_array.shape != deformed_array.shape or reference_array.shape != mask_array.shape:
        raise ValueError("reference, deformed, and ROI mask must have matching 2D shapes")
    problem = _build_problem(reference_array, deformed_array, mask_array, values, seeds)
    result = backend.PINSolver().solve(problem)
    if case_root is not None and write_case_artifacts:
        _write_case_artifacts(case_root, result, output_subdir, reference_array, values.get("output", {}))
    return result


def run_planar_case(config: str | Path | Mapping[str, Any] = "config/pin_2d.yaml") -> list[Any]:
    """Solve every deformed image discovered between a planar reference and ROI."""
    import copy

    values = _mapping(config)
    case = values.get("case", {})
    root = Path(case.get("root", "."))
    _, deformed, _ = planar_image_series(root, case.get("images_dir", "."))
    results = []
    for index, path in enumerate(deformed):
        current = copy.deepcopy(dict(values))
        current.setdefault("case", {})["frame"] = index
        results.append(pin_dic(root, config=current, output_subdir=path.stem))
    return results
