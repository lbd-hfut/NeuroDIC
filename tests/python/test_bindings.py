"""pybind11 smoke test.

The extension is optional in environments without pybind11. When it is built,
this test verifies the conceptual import name required by the architecture.
"""

import importlib

import neurodic


def test_binding_import_smoke() -> None:
    if not neurodic.native_available():
        return
    assert importlib.import_module("neurodic._neurodic") is not None
