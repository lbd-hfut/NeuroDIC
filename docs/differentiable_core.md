# Differentiable Core

## Rule

> Any operation between neural-field output and loss evaluation must preserve the
> PyTorch autograd graph. No NumPy/Eigen/OpenCV round-trip is allowed inside the
> differentiable path.

## Differentiable Path

```text
network parameters theta
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

All operations in this path use `torch::Tensor`.

## Non-Differentiable Preprocessing

Image loading, COLMAP parsing, calibration board detection, SIFT, RANSAC, ROI
preprocessing, metadata handling, and fixed B-spline coefficient preprocessing
may use standard C++, Eigen, OpenCV, or file-format libraries.

## Tensor Ownership

Observed images and coefficients are normally fixed observations. Coordinates,
model outputs, decoded fields, residuals, and losses must preserve autograd when
they participate in optimization.

## B-Spline Differentiation

The first implementation target is a LibTorch tensorized sampler. Coordinates
are expected to have shape `[N, 2]`. Gradients with respect to coordinates must
be validated with finite differences and `gradcheck` where applicable.

## Geometry Differentiation

Projection, coordinate transforms, and geometry operations that occur between a
model output and a loss must be implemented with `torch::Tensor` operations.
Offline calibration helpers may use non-differentiable libraries.

## Loss Construction

Loss functions must return scalar tensors and must not detach inputs.

## CPU/GPU Strategy

The initial reference path should run on CPU and CUDA through LibTorch tensor
operations. Custom CUDA kernels are a future optimization only after correctness
and gradients are validated.
