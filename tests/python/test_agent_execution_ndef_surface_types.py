"""NDeF combined-surface scalar type contract tests."""

from __future__ import annotations

import copy
import json
import os
import types
from pathlib import Path

import pytest

from fixtures.prepare_ndef_d2a_fixture import prepare
from neurodic.agent.adapters.execution_ndef import validate_ndef_surface_config
from neurodic.agent.errors import ControlPlaneError
from neurodic.agent.inspect import resolve_config
from neurodic.config import load_config


def _bounded_values(tmp_path: Path) -> tuple[Path, dict]:
    config = prepare(tmp_path / "d2a")
    values = resolve_config(config, case_key="ndef_d2a", case_paths=config.parent / "case_paths.yaml")["effective_config"]
    return config, values


def _replace(values: dict, path: str, value) -> dict:
    result = copy.deepcopy(values)
    target = result
    parts = path.split(".")
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    return result


def test_d2a_fixture_json_keeps_every_consumed_scalar_typed(tmp_path: Path) -> None:
    config, values = _bounded_values(tmp_path)

    assert load_config(config)["surface_training"]["weight_decay"] == 1e-6
    assert isinstance(values["surface_training"]["weight_decay"], float)
    assert isinstance(values["surface_training"]["pretrain_iterations"], int)
    assert isinstance(values["surface_training"]["pretrain_learning_rate"], float)
    assert isinstance(values["surface_dense_training"]["enabled"], bool)
    assert isinstance(values["surface_dense_training"]["epochs"], int)
    assert isinstance(values["surface_dense_training"]["learning_rate"], float)
    assert isinstance(values["surface_dense_training"]["seed"], int)
    assert isinstance(values["surface"]["fusion_relative_sample_spacing"], float)
    assert isinstance(values["surface"]["fusion_max_candidate_points"], int)
    validate_ndef_surface_config(values)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("surface_training.weight_decay", "1e-06"),
        ("surface_training.weight_decay", "abc"),
        ("surface_training.weight_decay", "1.2.3"),
        ("surface_training.weight_decay", None),
        ("surface_training.weight_decay", float("nan")),
        ("surface_training.weight_decay", float("inf")),
        ("surface_training.pretrain_iterations", True),
        ("surface_dense_training.samples_per_camera", True),
    ],
)
def test_invalid_surface_scalar_fails_closed_before_science(tmp_path: Path, path: str, value) -> None:
    _config, values = _bounded_values(tmp_path)

    with pytest.raises(ControlPlaneError) as raised:
        validate_ndef_surface_config(_replace(values, path, value))

    assert raised.value.record.code == "NDEF.CONFIG_TYPE_INVALID"
    assert raised.value.record.path == path


@pytest.mark.skipif(os.environ.get("NEURODIC_D2B0_NATIVE_CONSTRUCTION") != "1",
                    reason="requires the explicitly requested native construction-only probe")
def test_bounded_public_pretrain_reaches_solver_sentinel_without_solving() -> None:
    """Exercise real preparation and pybind assignments, then stop at solve."""
    import neurodic.api.ndef_surface as surface_api

    fixture = Path("/tmp/neurodic-d2a-ndef-smoke")
    roi = Path("/tmp/neurodic-d2a-managed-final/trials/d2a-roi-real-final/artifacts/ndef.roi.generate_call/attempt_2c1d67dd478b922c/roi/per_camera")
    values = resolve_config(fixture / "ndef_d2a.yaml", case_key="ndef_d2a", case_paths=fixture / "case_paths.yaml")["effective_config"]
    values["case"] = dict(values["case"])
    values["case"]["masks"] = str(roi)
    values["output"] = {"result": "/tmp/d2b0-construction-probe-results", "visualization": "/tmp/d2b0-construction-probe-visualization", "ndef_subdir": None}
    validate_ndef_surface_config(values)

    backend = surface_api._require_backend()

    class SentinelSolver:
        def solve(self, problem):
            assert isinstance(problem.weight_decay, float)
            assert problem.weight_decay == 1e-6
            assert problem.pretrain_iterations == 1
            assert problem.dense_epochs == 1
            assert problem.dense_samples_per_camera == 4
            raise AssertionError("D2-B0 solver sentinel: construction complete")

    proxy = types.SimpleNamespace(NDeFSurfaceProblem=backend.NDeFSurfaceProblem,
                                  NDeFDepthModelOptions=backend.NDeFDepthModelOptions,
                                  NDeFSurfaceSolver=SentinelSolver)
    original = surface_api._require_backend
    surface_api._require_backend = lambda: proxy
    try:
        with pytest.raises(AssertionError, match="D2-B0 solver sentinel: construction complete"):
            surface_api.pretrain_ndef_surface(values)
    finally:
        surface_api._require_backend = original
