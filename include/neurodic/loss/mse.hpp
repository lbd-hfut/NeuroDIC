/**
 * Mean squared error loss used for seed supervision.
 *
 * Responsibilities: future differentiable MSE photometric term.
 * Inputs: residual tensor.
 * Outputs: scalar loss tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic {
class MSELoss : public Loss {
public:
    torch::Tensor compute(const torch::Tensor& residual) override;
};
}  // namespace neurodic
