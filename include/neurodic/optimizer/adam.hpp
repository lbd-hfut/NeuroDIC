/**
 * LibTorch Adam optimizer and closure runner.
 *
 * Responsibilities: future project-level Adam integration.
 * Inputs: neural parameters and closure.
 * Outputs: updated parameters.
 * Ownership: future torch::optim state.
 * Differentiable: PARTIAL.
 */
#pragma once

#include "neurodic/optimizer/optimizer.hpp"

namespace neurodic {

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(std::vector<torch::Tensor> parameters, double learning_rate);
    OptimizationResult minimize(int iterations, const LossClosure& closure) override;

private:
    torch::optim::Adam optimizer_;
};

}  // namespace neurodic
