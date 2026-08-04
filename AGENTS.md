# NeuroDIC Codex Startup Context

This file is the project-level startup note for Codex. Read it before modifying
or building NeuroDIC.

## Authoritative Environment

The canonical development environment is documented in:

- [docs/development_environment.md](docs/development_environment.md)

Use that document as the source of truth for compilers, CMake flags, conda
environment paths, Python ABI selection, and known dependency limitations.

## Required Defaults

- Repository root: `/home/a306/01project/NeuroDIC`
- Conda environment path: `/home/a306/miniconda3/envs/neurodic`
- Do not use base Python for NeuroDIC builds.
- Do not use system `g++` for CUDA/LibTorch builds.
- GPU/driver access may be hidden by the normal Codex sandbox.
- For GPU runtime checks, CUDA/LibTorch GPU debugging, or local GPU compilation
  validation, request an escalated shell command instead of concluding that the
  GPU is absent.
- Always set `CUDAHOSTCXX` to the conda `x86_64-conda-linux-gnu-g++`.
- Always pass `-DPython_EXECUTABLE=/home/a306/miniconda3/envs/neurodic/bin/python`
  when configuring CMake with Python bindings.

## GPU / Sandbox Rule

Normal sandboxed commands may fail to contact the NVIDIA driver:

```bash
nvidia-smi
```

If GPU access is needed, use an escalated command request. Escalated
`nvidia-smi` was validated on 2026-08-04 and can see:

```text
NVIDIA GeForce RTX 4070 SUPER
Driver 595.71.05
CUDA 13.2
12282 MiB VRAM
```

Use escalated commands for:

- `nvidia-smi`
- GPU runtime smoke tests such as `torch.cuda.is_available()`
- CUDA/LibTorch configure/build/debug commands that need the local driver
- GPU memory, kernel, or runtime diagnostics

Do not request broad unsandboxed shell access. Ask for the specific command
needed for the GPU build/debug step.

## Canonical Configure Command

```bash
export ND_ENV=/home/a306/miniconda3/envs/neurodic
export CXX_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-g++
export C_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-gcc
export CUDAHOSTCXX=$CXX_COMPILER

$ND_ENV/bin/cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH=$ND_ENV \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DPython_EXECUTABLE=$ND_ENV/bin/python \
    -DNEURODIC_ENABLE_TORCH=ON \
    -DNEURODIC_BUILD_PYTHON=ON \
    -DNEURODIC_BUILD_TESTS=ON \
    -DNEURODIC_USE_EIGEN=OFF \
    -DNEURODIC_USE_OPENCV=OFF
```

Then build and test with:

```bash
CUDAHOSTCXX=$CXX_COMPILER $ND_ENV/bin/cmake --build build -j
CUDAHOSTCXX=$CXX_COMPILER $ND_ENV/bin/ctest --test-dir build --output-on-failure
```

For build-tree Python imports:

```bash
export PYTHONPATH=$PWD/python:$PWD/build/python
$ND_ENV/bin/python -c "import neurodic; import neurodic._neurodic"
```

## Current Audit Notes

- LibTorch, CUDA toolkit, pybind11, Eigen, yaml-cpp, CMake, Ninja, and conda GCC
  are installed in the `neurodic` environment.
- `pytest` is currently missing from the `neurodic` environment.
- GPU runtime access is hidden in the normal sandbox, but available through
  approved escalated commands.
- Set `MPLCONFIGDIR=/tmp/neurodic-matplotlib` before importing matplotlib in
  sandboxed runs.

## Architecture Reminder

NeuroDIC is C++ first: scientific core in `include/` and `src/`, LibTorch for
the differentiable path, pybind11 for `neurodic._neurodic`, and thin Python API
under `python/neurodic`.
