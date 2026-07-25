# PINSolver

`PINSolver` is the single top-level solver for both PIN-DIC 2D and PIN-DIC
Stereo.

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

Do not add `PIN2DSolver`, `PINStereoSolver`, `pin_2d_solver.*`, or
`pin_stereo_solver.*`.
