/**
 * Optimizer interface.
 *
 * Responsibilities: wrap project-level neural optimization loops.
 * Inputs: model parameters and differentiable loss closures.
 * Outputs: optimization step status.
 * Ownership: implementations own optimizer state.
 * Differentiable: PARTIAL. Optimizer consumes differentiable losses but is control logic.
 * TODO(NeuroDIC): define closure API compatible with LibTorch Adam/LBFGS.
 */
#pragma once

namespace neurodic {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
};

}  // namespace neurodic
