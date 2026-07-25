/**
 * Differentiable LibTorch B-spline sampler.
 *
 * Responsibilities: sample fixed B-spline coefficients at differentiable coordinates.
 * Inputs: coefficients tensor and coordinates with shape [N, 2].
 * Outputs: sampled intensities or spatial gradients.
 * Ownership: tensors use PyTorch reference-counted ownership.
 *
 * Differentiability
 * -----------------
 * YES with respect to coordinates. Coefficients are fixed observations by default.
 * Any operation between neural-field output and loss evaluation must preserve the
 * PyTorch autograd graph. No NumPy/Eigen/OpenCV round-trip is allowed inside the
 * differentiable path.
 *
 * TODO(NeuroDIC):
 * 1. Implement tensorized B-spline basis evaluation.
 * 2. Support CPU and CUDA through LibTorch tensor ops.
 * 3. Preserve autograd through coordinates.
 * 4. Verify gradients using finite differences / gradcheck.
 * 5. Optimize memory layout only after correctness.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

class TorchBSplineInterpolator {
public:
    explicit TorchBSplineInterpolator(int degree = 5);

    [[nodiscard]] int degree() const noexcept { return degree_; }

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
};

}  // namespace neurodic
