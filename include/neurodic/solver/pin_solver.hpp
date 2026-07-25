/**
 * Unified PIN solver.
 *
 * Responsibilities: solve both PIN-DIC 2D and PIN-DIC Stereo through one class.
 * Inputs: PINProblem with prepared data/calibration/coefficients/initialization.
 * Outputs: PINResult.
 * Ownership: owns model/optimizer during solve.
 * Differentiable: PARTIAL. Model -> representation -> geometry -> B-spline -> loss
 * must preserve the PyTorch autograd graph.
 * TODO(NeuroDIC): implement the first validated PIN-DIC 2D pipeline.
 */
#pragma once

#include "neurodic/core/result.hpp"
#include "neurodic/problem/pin_problem.hpp"
#include "neurodic/solver/solver.hpp"

namespace neurodic {

class PINSolver : public Solver {
public:
    PINResult solve(const PINProblem& problem) const;
};

}  // namespace neurodic
