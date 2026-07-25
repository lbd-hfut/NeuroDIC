# Architecture

Status: C++-first architecture skeleton.

```text
C++ Scientific Core
        ↓
Differentiable LibTorch Core
        ↓
libneurodic / neurodic_core
        ↓
pybind11
        ↓
neurodic._neurodic
        ↓
Thin Python API
```

The primary implementation lives in `include/neurodic` and `src`. Python modules
under `python/neurodic` should assemble inputs and call compiled bindings; they
must not duplicate scientific kernels.

## Solver Families

```text
Solver
├── PINSolver
└── NDeFSolver
```

`PINSolver` handles both PIN-DIC 2D and PIN-DIC Stereo. Stereo-specific behavior
belongs to data, calibration, geometry, problem configuration, and result
construction. `NDeFSolver` owns an internally controlled model topology.

## Problem Flow

```text
Data + ROI + Calibration + B-spline coefficients + Initialization
        ↓
ProblemBuilder
        ↓
PINProblem / NDeFProblem
        ↓
PINSolver / NDeFSolver
        ↓
Result
```

Calibration is independent of solvers. One ROI maps to one continuous neural
field. MSPINN/FBPINN multi-region domain decomposition is intentionally absent.
