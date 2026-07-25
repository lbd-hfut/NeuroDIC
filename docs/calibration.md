# Calibration

Calibration is independent from solvers.

```text
Mono / Stereo / COLMAP calibration
        ↓
CalibrationResult
        ↓
ProblemBuilder
        ↓
Problem
        ↓
Solver
```

Solvers consume `CalibrationResult`; they do not detect boards, parse COLMAP, or
estimate camera parameters.
