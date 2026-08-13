"""Thin YAML/image assembly for the compiled planar PIN-DIC solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import hashlib

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
    evaluation = config.get("evaluation", {})
    allowed = {"enabled", "sample_count", "seed", "patch_radius"}
    unknown = set(evaluation) - allowed
    if unknown:
        raise ValueError(f"Unknown PIN evaluation settings: {sorted(unknown)}")
    if evaluation:
        problem.evaluation_enabled = bool(evaluation.get("enabled", False))
        problem.evaluation_sample_count = int(evaluation.get("sample_count", problem.evaluation_sample_count))
        problem.evaluation_seed = int(evaluation.get("seed", problem.evaluation_seed))
        problem.evaluation_patch_radius = int(evaluation.get("patch_radius", problem.evaluation_patch_radius))


def _write_case_artifacts(case_root: Path, result, output_subdir: str | None, reference_image: np.ndarray,
                          output_config: Mapping[str, Any] | None = None, *, deformed_image: np.ndarray | None = None,
                          roi_mask: np.ndarray | None = None) -> None:
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
    history = result.training_history.numpy()
    np.savez_compressed(output / "diagnostics_training.npz", schema_version=np.asarray("neurodic.pin.training/v1"),
                        history=history, history_columns=np.asarray(["phase", "phase_step", "loss"]),
                        phase_names=np.asarray(["seed_mse", "photometric"]))
    if result.evaluation_requested_count:
        residuals = result.evaluation_residuals.numpy()
        finite = residuals[np.isfinite(residuals)]
        digest = hashlib.sha256()
        for item in (reference_image, deformed_image, roi_mask):
            if item is not None:
                array = np.ascontiguousarray(item); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        summary = {"schema_version": "neurodic.fixed_evaluation/v1", "solver": "pin",
                   "evaluation_set": {"identity": f"pin-v1:{digest.hexdigest()}:{result.evaluation_seed}:{result.evaluation_patch_radius}:{result.evaluation_loss_type}",
                                      "seed": result.evaluation_seed, "sampling": "stable_hash_ranked_roi_indices",
                                      "eligible_count": result.evaluation_eligible_count, "requested_count": result.evaluation_requested_count},
                   "loss": {"type": result.evaluation_loss_type, "patch_radius": result.evaluation_patch_radius,
                            "aggregation": "mean_per_valid_window", "unit": "photometric_objective"},
                   "valid_count": result.evaluation_valid_count,
                   "valid_ratio": result.evaluation_valid_count / result.evaluation_requested_count,
                   "summary": {"mean": float(finite.mean()) if len(finite) else None,
                               "median": float(np.median(finite)) if len(finite) else None,
                               "p95": float(np.percentile(finite, 95)) if len(finite) else None}}
        np.savez_compressed(output / "diagnostics_evaluation.npz", schema_version=np.asarray(summary["schema_version"]),
                            indices=result.evaluation_indices.numpy(), residual=residuals)
        (output / "diagnostics_evaluation.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
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
        _write_case_artifacts(case_root, result, output_subdir, reference_array, values.get("output", {}),
                              deformed_image=deformed_array, roi_mask=mask_array)
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
