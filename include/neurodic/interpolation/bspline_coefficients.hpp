/**
 * B-spline coefficient preprocessing.
 *
 * Responsibilities: compute fixed image coefficients for later differentiable sampling.
 * Inputs: observed image tensor and degree 1/3/5.
 * Outputs: coefficient tensor.
 * Ownership: returned tensor follows PyTorch reference-counted ownership.
 * Differentiable: PARTIAL. Coefficient preprocessing is normally run under
 * torch::NoGradGuard because coefficients are fixed observations.
 * TODO(NeuroDIC):
 * 1. Implement validated recursive filtering for degrees 3 and 5.
 * 2. Define boundary conditions.
 * 3. Validate coefficient layout against sampler expectations.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct BSplineCoefficientBlock {
    int height{0};
    int width{0};
    int degree{5};
    int pad_offset{0};
    torch::Tensor coeff_cpu;
    mutable torch::Tensor coeff_gpu;

    void validate() const;
    const torch::Tensor& cpu() const;
    const torch::Tensor& on(const torch::Device& device) const;
};

torch::Tensor compute_bspline_coefficients(const torch::Tensor& image, int degree);

BSplineCoefficientBlock make_bspline_coefficient_block(
    const torch::Tensor& mirror_padded_image,
    int degree,
    int pad_offset
);

}  // namespace neurodic
