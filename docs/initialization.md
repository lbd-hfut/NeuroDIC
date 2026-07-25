# Initialization

Initialization is non-differentiable preprocessing.

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

OpenCV/Eigen-style helpers may be used here later because this stage is outside
the model-to-loss autograd path.
