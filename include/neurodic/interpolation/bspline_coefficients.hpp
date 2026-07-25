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

torch::Tensor compute_bspline_coefficients(const torch::Tensor& image, int degree);

}  // namespace neurodic
