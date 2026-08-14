"""pybind11 smoke test.

The extension is optional in environments without pybind11. When it is built,
this test verifies the conceptual import name required by the architecture.
"""

import importlib

import neurodic


def test_binding_import_smoke() -> None:
    if not neurodic.native_available():
        return
    backend = importlib.import_module("neurodic._neurodic")
    assert backend is not None
    assert hasattr(backend, "PINStereoProblem")
    assert hasattr(backend, "PINStereoSolver")
    assert hasattr(backend, "PINStereoResult")


def test_ndef_deformation_binding_capability() -> None:
    if not neurodic.native_available():
        return
    from neurodic.agent.adapters.execution_ndef_deformation import (
        ndef_deformation_backend_capability,
    )

    capability = ndef_deformation_backend_capability()
    assert capability["available"] is True
    assert capability["exception"] is None
    assert capability["missing_symbols"] == []
    assert all(record["present"] for record in capability["symbols"].values())
