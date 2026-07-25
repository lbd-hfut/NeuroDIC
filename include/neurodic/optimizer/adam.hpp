/**
 * Adam optimizer wrapper shell.
 *
 * Responsibilities: future project-level Adam integration.
 * Inputs: neural parameters and closure.
 * Outputs: updated parameters.
 * Ownership: future torch::optim state.
 * Differentiable: PARTIAL.
 * TODO(NeuroDIC): implement after solver closure contract is fixed.
 */
#pragma once

#include "neurodic/optimizer/optimizer.hpp"

namespace neurodic { class AdamOptimizer : public Optimizer { public: void step() override; }; }
