/**
 * Optimizer interface.
 *
 * Responsibilities: wrap project-level neural optimization loops.
 * Inputs: model parameters and differentiable loss closures.
 * Outputs: optimization step status.
 * Ownership: implementations own optimizer state.
 * Differentiable: PARTIAL. Optimizer consumes differentiable losses but is control logic.
 */
#pragma once

#include <functional>
#include <vector>

#include <torch/torch.h>

namespace neurodic {

using LossClosure = std::function<torch::Tensor()>;

struct OptimizationResult {
    int iterations{0};
    double final_loss{0.0};
    std::vector<double> losses;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual OptimizationResult minimize(int iterations, const LossClosure& closure) = 0;
};

}  // namespace neurodic
