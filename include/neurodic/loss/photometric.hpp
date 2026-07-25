/**
 * Photometric loss shell.
 *
 * Responsibilities: connect warped sampling residuals to loss values.
 * Inputs: sampled reference/current tensors.
 * Outputs: scalar tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): compose B-spline sampling and robust residual choices.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic { class PhotometricLoss : public Loss { public: torch::Tensor compute(const torch::Tensor& residual) override; }; }
