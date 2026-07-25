/**
 * MSE loss shell.
 *
 * Responsibilities: future differentiable MSE photometric term.
 * Inputs: residual tensor.
 * Outputs: scalar loss tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): implement without detaching residual tensors.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic { class MSELoss : public Loss { public: torch::Tensor compute(const torch::Tensor& residual) override; }; }
