# NeuroDIC

Status: Architecture Reconstruction / C++-First Skeleton

NeuroDIC is being reconstructed around a C++ scientific core, a LibTorch
differentiable path, pybind11 bindings, and a thin Python API.

```text
NeuroDIC
├── PINSolver
│   ├── PIN-DIC 2D
│   └── PIN-DIC Stereo
└── NDeFSolver
    └── NDeF Multi-view DIC
```

The repository currently contains interfaces, TODOs, binding stubs, CMake
targets, documentation, tests, and examples. It does not contain validated
PIN-DIC or NDeF-DIC numerical algorithms yet.

## Architecture Rules

1. C++ is the primary scientific implementation language.
2. LibTorch owns the differentiable core.
3. The model-to-loss path uses `torch::Tensor` throughout.
4. Python access is provided through pybind11 as `neurodic._neurodic`.
5. The Python package is intentionally thin and lives under `python/neurodic`.
6. The first version solves one ROI as one continuous neural field.
7. One `PINSolver` handles both planar 2D and stereo PIN-DIC.
8. `NDeFSolver` owns an internally controlled architecture.
9. B-spline interpolation supports degrees 1, 3, and 5 only.
10. Calibration is performed before problem construction, not inside solvers.
11. `Representation` describes the physical field; `Model` describes the neural network.
12. MSPINN/FBPINN multi-region domain decomposition is out of scope.

## Differentiability Rule

Any operation between neural-field output and loss evaluation must preserve the
PyTorch autograd graph. No NumPy/Eigen/OpenCV round-trip is allowed inside the
differentiable path.

## Layout

```text
include/neurodic/       Public C++ interfaces
src/                    C++ skeleton implementations
bindings/python/        pybind11 binding skeleton for neurodic._neurodic
python/neurodic/        Thin Python API
tests/cpp/              C++ architecture and invariant tests
tests/python/           Python import and binding smoke tests
docs/                   Architecture and differentiability notes
```

## Build Notes

LibTorch is required for the differentiable core. If CMake cannot find Torch,
pass the PyTorch CMake prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The pybind11 extension is built only when pybind11 is discoverable by CMake.
