# NeuroDIC C++-First Architecture Reconstruction Instructions for Codex

> Version: v0.2  
> Project: **NeuroDIC**  
> Status: **Architecture Reconstruction / C++-First Skeleton / Interfaces and TODOs Only**  
> Primary goal: remove the previous Python-first skeleton and rebuild the project around a **C++ scientific core + LibTorch differentiable core + pybind11 Python bindings + thin Python high-level API**.

---

# 1. Executive Instruction

You are working inside the **existing NeuroDIC repository root**.

The repository may already contain files and directories created according to an older Python-first architecture. That old architecture is now superseded.

Your task is to:

1. inspect the current repository;
2. identify files/directories that belong to the old Python-first architecture;
3. remove or replace those obsolete architecture-skeleton files;
4. preserve unrelated user data, experiments, datasets, Git metadata, documentation, licenses, and any existing scientific implementation that is not clearly part of the obsolete skeleton;
5. rebuild the repository according to the new architecture specified in this document;
6. create the requested C++ header/source files, pybind11 binding files, Python package shell, tests, benchmarks, examples, CMake files, and documentation placeholders;
7. place **clear TODOs, responsibilities, abstract interfaces, function signatures, and design notes** inside all new code files;
8. keep complex scientific algorithms explicitly unimplemented unless a trivial safe placeholder is necessary for compilation;
9. make the C++ project compile as far as reasonably possible with placeholder implementations;
10. create a minimal pybind11 smoke-test extension when dependencies are available.

This is an **architecture reconstruction task**, not a full scientific implementation task.

---

# 2. Repository Migration Rules

The current repository may contain a previous architecture similar to:

```text
pinndic/
neurodic/
  core/
  data/
  interpolation/
  calibration/
  initialization/
  problem/
  solver/
  representation/
  model/
  geometry/
  loss/
  optimizer/
  postprocess/
  io/
  api/
  utils/
```

where the scientific implementation was planned primarily in Python.

That architecture must no longer be the primary implementation layout.

## 2.1 What to remove or replace

Remove obsolete skeleton files when they are clearly architecture-only placeholders from the previous Python-first design, especially:

```text
pinndic/
```

and Python modules that duplicate scientific-core responsibilities now owned by C++.

Examples of modules that should NOT remain as separate Python scientific implementations:

```text
python/neurodic/interpolation/bspline.py
python/neurodic/geometry/triangulation.py
python/neurodic/solver/pin_solver.py
python/neurodic/solver/ndef_solver.py
python/neurodic/loss/znssd.py
```

when these files would duplicate C++ implementations.

## 2.2 What must NOT be deleted automatically

Do not blindly delete:

```text
.git/
.gitignore
LICENSE
README content written by the user
datasets/
data/
results/
experiments/
assets/
papers/
notebooks/
third_party/
external/
existing validated scientific code
calibration data
COLMAP outputs
images
MAT files
configuration files containing real experiment settings
```

If an existing file has nontrivial scientific implementation, preserve it and report the architecture conflict instead of deleting it.

## 2.3 Migration report

Before destructive changes, inspect and classify existing content into:

```text
A. obsolete skeleton
B. reusable implementation
C. user data / experiments
D. unknown
```

Only category A may be removed automatically.

At the end, report everything removed, retained, moved, or replaced.

---

# 3. Core Architectural Principle

NeuroDIC must be organized as:

```text
C++ Scientific Core
        ↓
Differentiable LibTorch Core
        ↓
libneurodic
        ↓
pybind11
        ↓
_neurodic Python extension
        ↓
Thin Python package
        ↓
User-facing Python API
```

The main development direction is:

```text
C++ first
    ↓
C++ unit tests
    ↓
C++ numerical kernels
    ↓
LibTorch differentiable pipeline
    ↓
pybind11
    ↓
Python API
```

Do NOT recreate a Python-first scientific implementation.

---

# 4. Hard Architectural Constraints

The following decisions are fixed unless explicitly changed later.

## 4.1 Single ROI only

The first NeuroDIC version solves:

```text
one ROI → one continuous neural field
```

Do NOT introduce:

- MSPINN region partitioning;
- FBPINN domain decomposition;
- multiple subdomain activation schedules;
- partition-of-unity controllers;
- multiple ROI neural solvers;
- multiple independent subnetworks for one ROI.

## 4.2 Unified PIN solver

There are only two top-level solver families:

```text
Solver
├── PINSolver
└── NDeFSolver
```

`PINSolver` handles both:

```text
PIN-DIC 2D
PIN-DIC Stereo
```

Do NOT create:

```text
PIN2DSolver
PINStereoSolver
pin_2d_solver.*
pin_stereo_solver.*
```

The difference between 2D and stereo belongs to dataset, calibration, geometry, problem configuration, and result construction.

## 4.3 NDeF network topology is internal

`NDeFSolver` uses an internally controlled neural architecture.

Normal user-facing Python/YAML APIs must not expose arbitrary NDeF topology parameters such as hidden layer count, arbitrary layer widths, arbitrary skip topology, or arbitrary internal branches.

Internal/developer configuration may exist separately.

## 4.4 B-spline only

Image interpolation must use one unified B-spline implementation.

Supported degrees:

```text
1
3
5
```

Do NOT add separate `bilinear`, `bicubic`, or `quintic-interpolator` modules.

## 4.5 Calibration is independent of the solver

Calibration is performed before problem construction.

```text
Calibration
    ↓
CalibrationResult
    ↓
ProblemBuilder
    ↓
Problem
    ↓
Solver
```

Solver classes consume `CalibrationResult` and must not calibrate cameras themselves.

## 4.6 Representation != Model

Keep separate abstractions:

```text
Representation = what physical field is represented
Model          = what neural network approximates that field
Solver         = how optimization/training is performed
```

Examples:

```text
PINDisplacementField != MLP
NDeFDeformationField != NDeFInternalModel
```

---

# 5. Critical Differentiable-Core Rule

This is a hard design requirement.

## 5.1 Use `torch::Tensor` throughout the differentiable path

Any operation between the neural-field output and the final loss must preserve the PyTorch autograd graph.

The differentiable core must therefore use:

```cpp
torch::Tensor
```

as the primary tensor representation.

The differentiable path includes, where applicable:

- neural model forward;
- displacement/correspondence decoding;
- coordinate warping;
- B-spline image sampling;
- B-spline spatial derivatives;
- differentiable projection;
- differentiable coordinate transforms;
- differentiable geometry required by the loss;
- differentiable photometric residual;
- differentiable ZNSSD/MSE loss;
- regularization terms;
- neural optimization.

Required chain:

```text
network parameters θ
        ↓
network output
        ↓
displacement / deformation
        ↓
warped coordinates
        ↓
B-spline sampling
        ↓
photometric residual
        ↓
loss
        ↓
backward()
```

## 5.2 No graph-breaking conversion inside the differentiable path

Inside the differentiable path, do NOT perform round trips such as:

```text
torch::Tensor
    ↓
NumPy
    ↓
Eigen
    ↓
OpenCV
    ↓
torch::Tensor
```

when gradients must propagate through the operation.

The following rule must be documented in the code:

> Any operation between neural-field output and loss evaluation must preserve the PyTorch autograd graph. No NumPy/Eigen/OpenCV round-trip is allowed inside the differentiable path.

## 5.3 Non-differentiable preprocessing may use other libraries

The following components may use conventional C++ libraries where appropriate:

```text
OpenCV
Eigen
COLMAP adapters
standard C++
```

Examples:

- image loading;
- file I/O;
- calibration board detection;
- SIFT;
- RANSAC;
- COLMAP parsing;
- ROI preprocessing;
- sparse initialization;
- metadata handling;
- B-spline coefficient precomputation when coefficients are treated as constants.

## 5.4 B-spline-specific autograd requirement

The B-spline sampler is a core differentiable operation.

For PIN-DIC:

```text
(x, y)
   ↓
uθ(x, y), vθ(x, y)
   ↓
(x + uθ, y + vθ)
   ↓
B-spline sampler
   ↓
I(x + uθ, y + vθ)
   ↓
loss
```

Gradient propagation through sampling coordinates must be preserved.

Image coefficients normally represent fixed observations and do not necessarily require gradients.

Therefore separate coefficient preprocessing from differentiable sampling.

## 5.5 Preferred initial implementation strategy

First implementation:

```text
LibTorch tensor operations
```

Later optimization may introduce custom C++ CPU kernels, custom CUDA kernels, or `torch::autograd::Function`, but only after the LibTorch reference implementation is validated.

Do NOT prematurely write custom CUDA backward kernels during the skeleton phase.

---

# 6. Target Repository Structure

Reconstruct the repository toward:

```text
NeuroDIC/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── cmake/
│   ├── NeuroDICOptions.cmake
│   ├── NeuroDICDependencies.cmake
│   └── NeuroDICCompilerWarnings.cmake
├── config/
│   ├── pin_2d.yaml
│   ├── pin_stereo.yaml
│   ├── ndef_multiview.yaml
│   ├── calibration_mono.yaml
│   ├── calibration_stereo.yaml
│   └── calibration_colmap.yaml
├── include/neurodic/
│   ├── core/
│   │   ├── types.hpp
│   │   ├── result.hpp
│   │   ├── context.hpp
│   │   ├── config.hpp
│   │   └── exceptions.hpp
│   ├── data/
│   │   ├── image.hpp
│   │   ├── roi.hpp
│   │   ├── dataset.hpp
│   │   ├── stereo_dataset.hpp
│   │   └── multiview_dataset.hpp
│   ├── interpolation/
│   │   ├── bspline.hpp
│   │   ├── bspline_coefficients.hpp
│   │   └── torch_bspline.hpp
│   ├── initialization/
│   │   ├── initializer.hpp
│   │   ├── initialization_result.hpp
│   │   ├── sampling.hpp
│   │   ├── integer_search.hpp
│   │   ├── subpixel_search.hpp
│   │   ├── sift_initializer.hpp
│   │   ├── sparse_prior.hpp
│   │   └── output_normalization.hpp
│   ├── calibration/
│   │   ├── camera_model.hpp
│   │   ├── calibration_result.hpp
│   │   ├── calibration_manager.hpp
│   │   ├── mono_calibration.hpp
│   │   ├── stereo_calibration.hpp
│   │   └── colmap_calibration.hpp
│   ├── problem/
│   │   ├── problem.hpp
│   │   ├── pin_problem.hpp
│   │   ├── ndef_problem.hpp
│   │   └── problem_builder.hpp
│   ├── representation/
│   │   ├── representation.hpp
│   │   ├── pin_displacement_field.hpp
│   │   ├── pin_disparity_field.hpp
│   │   ├── ndef_surface_field.hpp
│   │   └── ndef_deformation_field.hpp
│   ├── model/
│   │   ├── neural_model.hpp
│   │   ├── model_factory.hpp
│   │   ├── mlp.hpp
│   │   ├── siren.hpp
│   │   ├── fourier.hpp
│   │   ├── resnet.hpp
│   │   └── ndef_internal_model.hpp
│   ├── geometry/
│   │   ├── geometry.hpp
│   │   ├── projection.hpp
│   │   ├── triangulation.hpp
│   │   ├── stereo_geometry.hpp
│   │   ├── ndef_geometry.hpp
│   │   ├── visibility.hpp
│   │   └── coordinate_transform.hpp
│   ├── loss/
│   │   ├── loss.hpp
│   │   ├── mse.hpp
│   │   ├── znssd.hpp
│   │   ├── photometric.hpp
│   │   └── regularization.hpp
│   ├── optimizer/
│   │   ├── optimizer.hpp
│   │   ├── adam.hpp
│   │   ├── lbfgs.hpp
│   │   ├── scheduler.hpp
│   │   └── convergence.hpp
│   ├── solver/
│   │   ├── solver.hpp
│   │   ├── pin_solver.hpp
│   │   └── ndef_solver.hpp
│   └── postprocess/
│       ├── displacement.hpp
│       ├── strain.hpp
│       ├── filtering.hpp
│       └── physical_scale.hpp
├── src/
│   ├── core/
│   ├── data/
│   ├── interpolation/
│   ├── initialization/
│   ├── calibration/
│   ├── problem/
│   ├── representation/
│   ├── model/
│   ├── geometry/
│   ├── loss/
│   ├── optimizer/
│   ├── solver/
│   └── postprocess/
├── bindings/python/
│   ├── module.cpp
│   ├── bind_core.cpp
│   ├── bind_data.cpp
│   ├── bind_interpolation.cpp
│   ├── bind_initialization.cpp
│   ├── bind_calibration.cpp
│   ├── bind_problem.cpp
│   ├── bind_geometry.cpp
│   ├── bind_solver.cpp
│   └── bind_result.cpp
├── python/neurodic/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── pin_dic.py
│   │   ├── ndef_dic.py
│   │   └── calibrate.py
│   ├── config/__init__.py
│   ├── io/__init__.py
│   └── visualization/__init__.py
├── tests/
│   ├── cpp/
│   │   ├── test_core.cpp
│   │   ├── test_roi.cpp
│   │   ├── test_bspline.cpp
│   │   ├── test_autograd_bspline.cpp
│   │   ├── test_initialization.cpp
│   │   ├── test_geometry.cpp
│   │   └── test_solver_interfaces.cpp
│   └── python/
│       ├── test_import.py
│       └── test_bindings.py
├── benchmarks/
│   ├── cpp/benchmark_bspline.cpp
│   └── python/benchmark_bindings.py
├── examples/
│   ├── cpp/
│   │   ├── pin_2d.cpp
│   │   └── bspline_sampling.cpp
│   └── python/
│       ├── pin_2d.py
│       ├── pin_stereo.py
│       └── ndef_multiview.py
└── docs/
    ├── architecture.md
    ├── differentiable_core.md
    ├── cpp_core.md
    ├── python_bindings.md
    ├── pin_solver.md
    ├── ndef_solver.md
    ├── calibration.md
    ├── initialization.md
    └── geometry.md
```

Do not create another nested project root.

---

# 7. C++ Skeleton Coding Standard

Every C++ header must contain:

1. file-level documentation;
2. namespace declaration;
3. responsibility description;
4. inputs;
5. outputs;
6. ownership/lifetime notes where relevant;
7. differentiability notes where relevant;
8. explicit TODO items;
9. public abstract interfaces;
10. no fake scientific output.

Use:

```cpp
namespace neurodic {
...
}
```

Prefer `#pragma once` for headers.

For unfinished scientific methods, either:

```cpp
throw std::logic_error("TODO: implement ...");
```

or leave them pure virtual when appropriate.

Do not return fake zeros that could be mistaken for valid scientific results.

---

# 8. Core Module Skeleton

## `include/neurodic/core/types.hpp`

Create enums similar to:

```cpp
#pragma once

namespace neurodic {

enum class SolverType { PIN, NDEF };
enum class CalibrationType { NONE, MONO, STEREO, COLMAP };
enum class GeometryType { PLANAR_2D, STEREO, NDEF_MULTIVIEW };
enum class InterpolationDegree : int { LINEAR = 1, CUBIC = 3, QUINTIC = 5 };
enum class SolverStatus { NOT_STARTED, RUNNING, CONVERGED, NOT_CONVERGED, FAILED };

}  // namespace neurodic
```

## `context.hpp`

Create a runtime context shell with fields such as:

```cpp
std::string device = "auto";
torch::Dtype dtype = torch::kFloat32;
std::optional<std::uint64_t> random_seed;
bool debug = false;
```

Do not over-engineer backend dispatch.

## `result.hpp`

Create lightweight result/diagnostic structs. Use `torch::Tensor` for differentiable numerical field results where appropriate.

---

# 9. Data Module

Data containers should be C++ first.

Suggested conceptual interfaces:

```cpp
class Image {
public:
    Image() = default;
    int64_t width() const;
    int64_t height() const;
    int64_t channels() const;
    void validate() const;

    // TODO:
    // - decide owned storage vs view semantics;
    // - decide CPU image container vs torch::Tensor backing.
};
```

ROI:

```cpp
class ROI {
public:
    bool contains(double x, double y) const;
    void validate() const;

    // TODO:
    // - mask representation;
    // - polygon representation;
    // - bounds;
    // - conversion to sampling coordinates.
};
```

Single ROI only.

---

# 10. B-Spline Module

This module requires special care.

Create:

```text
bspline.hpp
bspline_coefficients.hpp
torch_bspline.hpp
```

## 10.1 Reference / preprocessing interface

Example:

```cpp
torch::Tensor compute_bspline_coefficients(
    const torch::Tensor& image,
    int degree
);
```

Supported degrees: `1, 3, 5`.

Document whether preprocessing is expected to run under `torch::NoGradGuard`.

## 10.2 Differentiable sampler

`torch_bspline.hpp` must define a differentiable sampler API.

```cpp
class TorchBSplineInterpolator {
public:
    explicit TorchBSplineInterpolator(int degree = 5);

    torch::Tensor evaluate(
        const torch::Tensor& coefficients,
        const torch::Tensor& coordinates
    ) const;

    torch::Tensor gradient(
        const torch::Tensor& coefficients,
        const torch::Tensor& coordinates
    ) const;

private:
    int degree_;

    // TODO:
    // 1. Implement tensorized B-spline basis evaluation.
    // 2. Support CPU and CUDA through LibTorch tensor ops.
    // 3. Preserve autograd through coordinates.
    // 4. Verify gradients using finite differences / gradcheck.
    // 5. Optimize memory layout only after correctness.
};
```

Document coordinate shape, preferably `[N, 2]`.

Do not convert coordinates to Eigen/NumPy/OpenCV inside `evaluate()`.

## 10.3 Autograd test placeholder

Create `tests/cpp/test_autograd_bspline.cpp` with TODOs for:

```text
loss = sampler(coefficients, coordinates).sum()
loss.backward()
coordinates.grad() must exist
finite-difference comparison
torch::autograd::gradcheck where applicable
CPU/GPU consistency
```

Do not fabricate a passing numerical test before the implementation exists.

---

# 11. Initialization Module

Create:

```cpp
struct InitializationResult {
    torch::Tensor coordinates;
    torch::Tensor displacement;
    torch::Tensor confidence;
    torch::Tensor displacement_mean;
    torch::Tensor displacement_scale;
};
```

Conceptual pipeline:

```text
ROI
 ↓
uniform sampling / SIFT
 ↓
integer or subpixel displacement
 ↓
SparsePrior
 ↓
mean / scale estimation
 ↓
neural-field warm start
```

Initialization itself does not need to be differentiable. OpenCV/Eigen may be used here where appropriate.

---

# 12. Calibration Module

Create interfaces for:

```text
MonoCalibration
StereoCalibration
COLMAPCalibration
CalibrationManager
CalibrationResult
CameraModel
```

`CalibrationResult` is the standardized output consumed by problem builders and solvers.

Do not perform calibration inside `PINSolver` or `NDeFSolver`.

---

# 13. Problem Module

Create:

```text
DICProblem
PINProblem
NDeFProblem
ProblemBuilder
```

Conceptual flow:

```text
Data
+ ROI
+ Calibration
+ B-spline coefficients
+ Initialization
        ↓
ProblemBuilder
        ↓
PINProblem / NDeFProblem
```

`PINProblem` serves both planar 2D and stereo PIN-DIC.

---

# 14. Representation Module

Create base:

```cpp
class FieldRepresentation {
public:
    virtual ~FieldRepresentation() = default;

    virtual torch::Tensor decode(
        const torch::Tensor& coordinates,
        const torch::Tensor& model_output
    ) const = 0;
};
```

Create shells:

```text
PINDisplacementField
PINDisparityField
NDeFSurfaceField
NDeFDeformationField
```

Do not finalize the exact stereo correspondence parameterization yet. Document that decision as TODO.

---

# 15. Model Module

Use LibTorch-compatible interfaces.

A practical skeleton may wrap `torch::nn::Module` or define a thin project-level abstraction around it.

PIN user-selectable future models:

```text
MLP / FCN
SIREN
Fourier-feature network
ResNet-like network
```

NDeF uses `NDeFInternalModel`, which must remain internally controlled.

Do not expose arbitrary NDeF topology through the public Python API.

---

# 16. Geometry Module

Geometry is one top-level concept but has different engines.

## 16.1 Stereo geometry

Create interfaces for projection, triangulation, reference 3D reconstruction, current 3D reconstruction, and 3D displacement.

## 16.2 NDeF geometry

Create interfaces for multi-view projection, reference surface projection, deformed surface projection, visibility, and photometric sampling geometry.

Do not force NDeF into stereo triangulation architecture.

## 16.3 Differentiability

If geometry appears between model output and loss, implement it with `torch::Tensor` operations.

Offline calibration helpers may use Eigen/OpenCV.

---

# 17. Loss Module

Base interface:

```cpp
class Loss {
public:
    virtual ~Loss() = default;
    virtual torch::Tensor compute(/* TODO arguments */) = 0;
};
```

Create shells:

```text
MSELoss
ZNSSDLoss
PhotometricLoss
Regularization
```

Loss must return differentiable scalar tensors.

Do not detach tensors inside the loss implementation.

---

# 18. Optimizer Module

Create wrappers/project-level interfaces for Adam, LBFGS, learning-rate scheduler, and convergence monitor.

The optimizer must not know camera calibration details.

Do not add MSPINN domain scheduling.

---

# 19. Solver Module

## 19.1 Base solver

```cpp
class Solver {
public:
    virtual ~Solver() = default;
};
```

## 19.2 PINSolver

Create one `PINSolver`.

```text
PINProblem
   ↓
Representation
   ↓
User-selectable Model
   ↓
Geometry
   ↓
B-spline sampling
   ↓
Loss
   ↓
Optimizer
   ↓
PINResult
```

For planar 2D:

```text
(x, y) → (u, v)
```

For stereo, `PINSolver` still handles the PIN continuous image-space field, while `StereoGeometry` handles 3D reconstruction.

## 19.3 NDeFSolver

```text
NDeFProblem
   ↓
Reference Surface Representation
   ↓
NDeF Deformation Representation
   ↓
Internal Model
   ↓
NDeFGeometry
   ↓
Photometric Loss
   ↓
Optimizer
   ↓
NDeFResult
```

---

# 20. Postprocess Module

Create interfaces for displacement magnitude, strain, filtering, and physical scaling.

Postprocessing does not belong inside solver optimization logic.

---

# 21. pybind11 Layer

The Python binding layer must wrap C++ functionality rather than reimplement it.

Create:

```text
bindings/python/module.cpp
bindings/python/bind_core.cpp
bindings/python/bind_data.cpp
bindings/python/bind_interpolation.cpp
bindings/python/bind_initialization.cpp
bindings/python/bind_calibration.cpp
bindings/python/bind_problem.cpp
bindings/python/bind_geometry.cpp
bindings/python/bind_solver.cpp
bindings/python/bind_result.cpp
```

The compiled extension should conceptually be:

```text
neurodic._neurodic
```

Do not expose every internal class automatically.

---

# 22. Python Package

The Python layer is intentionally thin.

Create only high-level modules such as:

```text
python/neurodic/api/
python/neurodic/config/
python/neurodic/io/
python/neurodic/visualization/
```

Do not duplicate C++ scientific kernels in Python.

Example future API:

```python
import neurodic
result = neurodic.pin_dic(...)
```

Internally:

```text
Python API
   ↓
_neurodic
   ↓
C++ / LibTorch
```

---

# 23. CMake Requirements

Create a clean CMake project.

Initial target structure should allow:

```text
neurodic_core
neurodic_python
```

or equivalent.

Plan dependencies for:

```text
LibTorch
pybind11
optional OpenCV
optional Eigen
testing framework
```

LibTorch is a required architectural dependency for the differentiable core.

CMake must document how `Torch_DIR` or the installed PyTorch CMake prefix should be supplied.

Useful future approach:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

Do not hardcode a machine-specific path.

---

# 24. Python Packaging

Use project name `NeuroDIC` and Python import name `neurodic`.

Do not use the old package name `pinndic`.

`pyproject.toml` should be prepared for a compiled extension build.

Do not finalize a complicated wheel/release system during this task.

---

# 25. Config Templates

Create placeholder YAML templates:

```text
pin_2d.yaml
pin_stereo.yaml
ndef_multiview.yaml
calibration_mono.yaml
calibration_stereo.yaml
calibration_colmap.yaml
```

PIN configurations may expose model type and width/depth.

NDeF public config must NOT expose arbitrary network topology.

---

# 26. Testing Skeleton

Create tests for architectural invariants.

Examples:

```text
core enum definitions
single ROI behavior
B-spline degree validation
torch tensor API shape validation
PINSolver existence
NDeFSolver existence
no separate PIN2DSolver/PINStereoSolver
CalibrationResult interface
Representation != Model separation
pybind11 import smoke test
```

Future numerical tests must include:

```text
B-spline interpolation accuracy
B-spline gradient correctness
autograd coordinate gradient
finite-difference gradient comparison
CPU/GPU consistency
stereo geometry correctness
PIN-DIC synthetic displacement recovery
```

Do not write fake scientific correctness assertions before algorithms exist.

---

# 27. Build and Verification Targets

The first architecture reconstruction should aim for:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If LibTorch is available, verify compilation of minimal Torch code.

If pybind11 is available, attempt:

```bash
python -c "import neurodic"
python -c "import neurodic._neurodic"
```

If a dependency is unavailable, do not install large packages automatically unless already permitted. Report the missing dependency, attempted command, and expected configuration.

---

# 28. Implementation Phases

## Phase 0 — Repository reconstruction

```text
inspect old architecture
classify files
remove obsolete Python-first skeleton
create C++-first directory tree
create CMake skeleton
create pyproject skeleton
create documentation skeleton
```

## Phase 1 — Core infrastructure

```text
core
data
ROI
result
context
C++ unit test infrastructure
```

## Phase 2 — B-spline reference and differentiable sampler

```text
degree 1 / 3 / 5
coefficient preprocessing
LibTorch sampler
coordinate gradients
autograd tests
benchmark skeleton
```

## Phase 3 — Initialization

```text
ROI sampling
integer search interface
subpixel interface
SIFT adapter
SparsePrior
mean / scale normalization
```

## Phase 4 — Calibration + common geometry

```text
Mono
Stereo
COLMAP adapter
projection
triangulation
coordinate transforms
```

## Phase 5 — Problem + representation + model abstractions

```text
PINProblem
NDeFProblem
ProblemBuilder
FieldRepresentation
NeuralModel
```

## Phase 6 — Loss + optimizer abstractions

```text
MSE
ZNSSD
Photometric
Regularization
Adam
LBFGS
Convergence
```

## Phase 7 — PINSolver skeleton

One solver only:

```text
PIN 2D
PIN Stereo
```

## Phase 8 — PIN-DIC 2D implementation

First complete scientific pipeline.

## Phase 9 — Stereo PIN-DIC

Reuse `PINSolver`; add `StereoGeometry`.

## Phase 10 — NDeF

```text
COLMAP
surface
visibility
NDeFGeometry
NDeFDeformationField
NDeFSolver
```

## Phase 11 — complete pybind11 bindings

Expose validated core interfaces.

## Phase 12 — Python user API

Thin wrappers only.

---

# 29. First Codex Task Scope

For THIS task, do only:

```text
Phase 0
+
selected interface skeletons from Phases 1–7
```

Specifically:

1. reconstruct the directory tree;
2. create header/source files;
3. write documentation blocks;
4. write TODOs;
5. write abstract interfaces;
6. write safe minimal constructors/validation where obvious;
7. create minimal CMake targets;
8. create pybind11 binding skeleton;
9. create thin Python package skeleton;
10. create test skeleton;
11. do not implement full PIN-DIC;
12. do not implement full NDeF;
13. do not implement custom CUDA;
14. do not invent unvalidated scientific equations.

---

# 30. Required TODO Style

Every incomplete module should explain what remains to be implemented.

Example:

```cpp
// TODO(NeuroDIC):
// 1. Implement tensorized cubic B-spline basis evaluation.
// 2. Preserve autograd with respect to sampling coordinates.
// 3. Validate gradients using finite differences.
// 4. Add CUDA performance benchmark after correctness is established.
```

Avoid useless TODOs such as:

```cpp
// TODO: implement
```

Prefer concrete scientific/software tasks.

---

# 31. Required Differentiability Comments

Every module involved in the differentiable path must explicitly state one of:

```text
Differentiable: YES
Differentiable: NO
Differentiable: PARTIAL
```

Example:

```cpp
/**
 * Differentiability
 * -----------------
 * YES with respect to `coordinates`.
 *
 * `coefficients` are treated as fixed image observations in the standard
 * DIC workflow and do not require gradients by default.
 *
 * Do not convert `coordinates` to Eigen/OpenCV/NumPy inside this method.
 */
```

This requirement applies especially to:

```text
torch_bspline
representation
model
projection
differentiable geometry
loss
solver optimization path
```

---

# 32. README Architecture Summary

Update README to state:

```text
Status: Architecture Reconstruction / C++-First Skeleton
```

Include:

```text
NeuroDIC
├── PINSolver
│   ├── PIN-DIC 2D
│   └── PIN-DIC Stereo
└── NDeFSolver
    └── NDeF Multi-view DIC
```

Also document:

1. C++ first;
2. LibTorch differentiable core;
3. `torch::Tensor` throughout model-to-loss path;
4. pybind11 Python bindings;
5. thin Python API;
6. single ROI;
7. unified PINSolver;
8. internally controlled NDeF architecture;
9. B-spline 1/3/5 only;
10. calibration separated from solver;
11. Representation != Model;
12. no MSPINN multi-region architecture.

---

# 33. `docs/differentiable_core.md`

Create this file and document:

```text
Differentiable path
Non-differentiable preprocessing
Tensor ownership
Autograd constraints
B-spline differentiation
Geometry differentiation
Loss construction
CPU/GPU strategy
Future custom CUDA strategy
```

Include this rule prominently:

> Any operation between neural-field output and loss evaluation must preserve the PyTorch autograd graph. No NumPy/Eigen/OpenCV round-trip is allowed inside the differentiable path.

---

# 34. What Codex Must NOT Do

Do NOT:

1. retain the old Python-first architecture as the primary scientific implementation;
2. create a second nested NeuroDIC project root;
3. duplicate C++ scientific kernels in Python;
4. create PIN2DSolver and PINStereoSolver;
5. add MSPINN multi-region logic;
6. add bicubic/bilinear interpolation modules;
7. expose arbitrary NDeF topology publicly;
8. put calibration inside solver classes;
9. merge Representation and Model abstractions;
10. break autograd with NumPy/Eigen/OpenCV conversions;
11. implement fake numerical outputs;
12. implement unvalidated stereo correspondence equations;
13. invent NDeF geometry;
14. write custom CUDA kernels before the LibTorch reference path is validated;
15. optimize before correctness;
16. silently delete user data or existing nontrivial scientific implementation;
17. rename the Python package back to `pinndic`;
18. finalize unstable public APIs unnecessarily.

---

# 35. Final Verification

Run as many of the following as the current environment supports:

```bash
find . -type f | sort

cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If Python packaging is wired:

```bash
python -m compileall python/neurodic
python -c "import neurodic"
```

If the extension is built:

```bash
python -c "import neurodic._neurodic"
```

Check that obsolete old-package imports do not remain:

```bash
grep -R "pinndic" -n . \
  --exclude-dir=.git \
  --exclude-dir=build
```

Check that forbidden separate solver names do not exist:

```bash
find . -type f | grep -Ei 'pin_2d_solver|pin_stereo_solver|pin2dsolver|pinstereosolver'
```

Check for graph-breaking TODO risks:

```bash
grep -R "numpy\|Eigen\|cv::Mat" -n src include \
  | grep -Ei "loss|solver|representation|torch_bspline|photometric|geometry"
```

Do not treat this grep as proof of an error; inspect each occurrence.

---

# 36. Final Codex Report

After execution, report:

1. old architecture files/directories identified;
2. files removed;
3. files preserved;
4. files moved/replaced;
5. new C++ directory tree;
6. new headers;
7. new source files;
8. new pybind11 files;
9. new Python wrapper files;
10. tests created;
11. CMake targets created;
12. LibTorch detection status;
13. pybind11 detection status;
14. compilation result;
15. test result;
16. Python import result;
17. TODO-heavy modules;
18. differentiable-core modules;
19. any current code that conflicts with the new architecture;
20. recommended next implementation task.

The recommended next implementation task after the skeleton is:

```text
Core + Data + ROI
        ↓
B-spline coefficient preprocessing
        ↓
LibTorch differentiable B-spline sampler
        ↓
autograd gradient validation
```

---

# 37. Final Execution Instruction

Proceed to modify the current NeuroDIC repository directly.

First inspect and classify the existing repository.

Then remove only obsolete Python-first skeleton components that are clearly superseded by this document.

Rebuild the architecture according to this specification.

Create all requested pre-code files.

Inside each file:

- write responsibilities;
- write TODOs;
- define abstract interfaces;
- define function/class signatures;
- document differentiability;
- avoid fake numerical behavior.

Do not only describe what should be done.

Perform the reconstruction and report the result.
