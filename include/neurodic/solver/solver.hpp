/**
 * Base solver interface.
 *
 * Responsibilities: mark top-level optimization families.
 * Inputs: prepared problem objects only.
 * Outputs: typed result objects.
 * Ownership: solvers own optimization/model state during solve.
 * Differentiable: PARTIAL. Solver orchestration preserves the differentiable path.
 * TODO(NeuroDIC): add shared diagnostics and cancellation hooks.
 */
#pragma once

namespace neurodic {

class Solver {
public:
    virtual ~Solver() = default;
};

}  // namespace neurodic
