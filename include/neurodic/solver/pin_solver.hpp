/**
 * Unified PIN solver.
 *
 * Responsibilities: solve planar 2D PIN-DIC through one C++/LibTorch path.
 * Inputs: PINProblem with prepared data/calibration/coefficients/initialization.
 * Outputs: PINResult.
 * Ownership: owns model/optimizer during solve.
 * Differentiable: PARTIAL. Model -> representation -> geometry -> B-spline -> loss
 * must preserve the PyTorch autograd graph.
 * The current implementation performs seed MSE pretraining followed by ZNSSD
 * photometric Adam optimization. Stereo and L-BFGS refinement remain pending.
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
