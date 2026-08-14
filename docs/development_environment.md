# NeuroDIC Local Development Environment

Status: Established.

This document defines the canonical local development environment for the
NeuroDIC project, the toolchain versions, where development happens, and the
exact commands to configure and build the project.

## 1. Environment Overview

Development happens in a dedicated conda environment named `neurodic`.
It contains the GPU LibTorch C++ libraries, the CUDA toolchain, Eigen, and all
project dependencies, pre-installed and validated.

| Item | Value |
|---|---|
| Conda environment | `neurodic` |
| Environment path | `/home/a306/miniconda3/envs/neurodic` |
| Python | 3.12.13 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GiB (sm_89) |
| Driver | 595.71.05 |
| CMake prefix | `-DCMAKE_PREFIX_PATH=/home/a306/miniconda3/envs/neurodic` |
| Python executable for bindings | `-DPython_EXECUTABLE=/home/a306/miniconda3/envs/neurodic/bin/python` |

## 2. Toolchain

### 2.1 C++ scientific core (LibTorch)

| Tool | Version | Notes |
|---|---|---|
| LibTorch | **2.12.1** `cuda130` (conda-forge) | `lib/libtorch_cuda.so`, `lib/libtorch_cpu.so`, `lib/libc10.so` |
| Torch CMake config | `share/cmake/Torch/TorchConfig.cmake` | found by `find_package(Torch)` |
| CUDA toolkit | 13.0 (nvcc **13.0.88**, ptxas) | `bin/nvcc` |
| C++ compiler | conda-forge gcc/g++ **14.3.0** | `x86_64-conda-linux-gnu-g++` (NOT the system g++ 15.2) |
| CMake | 4.4.2 | in the environment |
| Ninja | 1.13.2 | default generator |

### 2.2 Libraries

| Library | Version | Located |
|---|---|---|
| Eigen | 3.x (conda-forge) | `include/eigen3` |
| pybind11 | conda-forge | `share/cmake/pybind11/pybind11Config.cmake` |
| protobuf | 6.x (conda-forge) | required by Torch CMake config |
| yaml-cpp | conda-forge | for config parsing |
| OpenCV (Python) | opencv-python (pip) | Python layer only |

### 2.3 Python packages (in `neurodic` env)

| Package | Version |
|---|---|
| torch | **2.12.1** (pip, aligned with LibTorch 2.12.1) |
| numpy, scipy, opencv-python, pyyaml, matplotlib, imageio | latest |

Current dependency audit notes:

- `pytest` is not installed in the `neurodic` conda environment as of
  2026-08-04. C++ tests run through `ctest`; Python tests require installing
  pytest or running them from another environment intentionally.
- Normal sandboxed Codex commands cannot communicate with the NVIDIA driver in
  this session, and sandboxed `torch.cuda.is_available()` returns `False`.
  Escalated local commands can access the GPU; escalated `nvidia-smi` was
  validated on 2026-08-04 and reports the RTX 4070 SUPER.
- `matplotlib` imports, but its default config path is not writable in this
  sandbox. Set `MPLCONFIGDIR=/tmp/neurodic-matplotlib` when scripts import
  matplotlib.

Python `torch` version is intentionally aligned with the C++ LibTorch
version (both 2.12.1) to avoid ABI mismatches when the pybind11 extension is
loaded alongside `torch` in the same Python process.

## 3. Critical Build Rules

### 3.1 Use the conda compiler, not the system g++

The system g++ is **15.2.0** on this machine. nvcc 13.0.88 cannot parse
`/usr/include/x86_64-linux-gnu/bits/mathcalls.h` with g++ 15
(`--gnu_version=150200` error), so CUDA language detection fails.
The conda-forge gcc 14.3.0 must be used, and `CUDAHOSTCXX` must point at it.

### 3.2 pybind11 needs an explicit find_package(Python) before it

NeuroDIC calls `find_package(Torch)` first; Torch's Caffe2 config sets
`Python_FOUND`, which makes pybind11 (NewTools) skip its own
`find_package(Python)`. `python_add_library` is then never defined and
`pybind11_add_module` fails with "Unknown CMake command". This is fixed in
`cmake/NeuroDICDependencies.cmake` by an explicit
`find_package(Python COMPONENTS Interpreter Development.Module)` before
`find_package(pybind11)`.

### 3.3 The static core must be built with -fPIC

`neurodic_core` is a static library linked into the pybind11 shared module
`_neurodic`. Without position-independent code the shared-object link fails
(`relocation R_X86_64_PC32 ... can not be used when making a shared object`).
This is fixed in `CMakeLists.txt` with
`set_target_properties(neurodic_core PROPERTIES POSITION_INDEPENDENT_CODE ON)`.

### 3.4 Canonical configure command

```bash
export ND_ENV=/home/a306/miniconda3/envs/neurodic
export CXX_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-g++
export C_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-gcc
export CUDAHOSTCXX=$CXX_COMPILER

cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH=$ND_ENV \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DPython_EXECUTABLE=$ND_ENV/bin/python \
    -DNEURODIC_ENABLE_TORCH=ON \
    -DNEURODIC_BUILD_PYTHON=ON \
    -DNEURODIC_BUILD_TESTS=ON \
    -DNEURODIC_USE_EIGEN=OFF \
    -DNEURODIC_USE_OPENCV=OFF

cmake --build build -j
ctest --test-dir build --output-on-failure
```

`CUDAHOSTCXX` must be set for both configure and build invocations.

`Python_EXECUTABLE` must point to `$ND_ENV/bin/python`. Without it, CMake may
discover the base Python interpreter first and build `_neurodic` with the wrong
CPython ABI suffix.

### 3.5 GPU access from Codex

The normal Codex sandbox may hide the local NVIDIA driver. Do not treat a
sandboxed `nvidia-smi` failure as proof that the GPU or CUDA environment is
missing.

When a task needs local GPU runtime access, CUDA/LibTorch GPU debugging, or
driver-visible diagnostics, request an escalated command for the specific step.
Validated escalated GPU state:

```text
NVIDIA GeForce RTX 4070 SUPER
Driver 595.71.05
CUDA 13.2
12282 MiB VRAM
```

Examples of commands that may require escalation:

```bash
nvidia-smi
CUDAHOSTCXX=$CXX_COMPILER cmake --build build -j
$ND_ENV/bin/python -c "import torch; print(torch.cuda.is_available())"
```

### 3.6 Python bindings

The pybind11 extension (`_neurodic`) is built when pybind11 is discoverable;
it links against the same LibTorch. Building with `-DNEURODIC_BUILD_PYTHON=ON`
and the prefix above produces `neurodic/_neurodic*.so` in the build tree.

For build-tree imports during development:

```bash
export PYTHONPATH=$PWD/python:$PWD/build/python
$ND_ENV/bin/python -c "import neurodic; import neurodic._neurodic"
```

Native-capable control-plane actions must retain that exact source-before-build
ordering. `PYTHONPATH=$PWD/python` is suitable only for native-free planning and
tests; it cannot resolve the build-tree extension. Before an NDeF deformation
execution, the control plane additionally requires these extension attributes:
`CameraModel`, `NDeFProblem`, `NDeFModelOptions`,
`estimate_ndef_displacement_scale`, `PhotometricLossType`, and `NDeFSolver`.
The import path and runtime capability report are environment evidence, not
scientific producer-signature determinants.

## 4. Where Development Happens

| Task | Environment | Command |
|---|---|---|
| C++ core build/test | `neurodic` | `cmake --build build` + `ctest` (see 3.2) |
| Python bindings | `neurodic` | via CMake with `-DNEURODIC_BUILD_PYTHON=ON` |
| Python thin API dev | `neurodic` | `python -m neurodic ...` |
| Cross-check vs MSPINN-DIC | `neurodic` (torch) | scripts under `tests/python` |

The older `pinndic2d` environment is superseded for NeuroDIC work; it is kept
only as a historical reference (libtorch 2.10.0). The base conda environment
is not used for NeuroDIC builds.

## 5. Validation Status

- [x] `find_package(Torch)` succeeds with `-DCMAKE_PREFIX_PATH=$ND_ENV`.
- [x] Full build of `neurodic_core` + `neurodic_cpp_tests` succeeds (Ninja, gcc 14.3).
- [x] `ctest` passes: 1/1 test (neurodic_cpp_tests).
- [x] CUDA arch auto-detected: `compute_89,code=sm_89` (RTX 4070 SUPER).
