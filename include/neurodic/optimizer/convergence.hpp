/**
 * Convergence monitor shell.
 *
 * Responsibilities: decide when solver optimization should stop.
 * Inputs: iteration, loss, gradient norms, update norms.
 * Outputs: convergence decision.
 * Ownership: value shell.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement robust stopping criteria and diagnostics.
 */
#pragma once

namespace neurodic {

struct ConvergenceMonitor {
    int max_iterations = 0;
    double tolerance = 0.0;
    bool converged(double /*loss*/) const { return false; }
};

}  // namespace neurodic
