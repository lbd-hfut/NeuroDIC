/**
 * NDeF solver.
 *
 * Responsibilities: solve NDeF multi-view DIC with internally controlled topology.
 * Inputs: NDeFProblem.
 * Outputs: NDeFResult.
 * Ownership: owns internal model/optimizer during solve.
 * Differentiable: PARTIAL. NDeF model -> geometry -> photometric loss preserves autograd.
 * TODO(NeuroDIC): implement only after PIN-DIC and geometry foundations are validated.
 */
#pragma once

#include "neurodic/core/result.hpp"
#include "neurodic/problem/ndef_problem.hpp"
#include "neurodic/solver/solver.hpp"

namespace neurodic {

class NDeFSolver : public Solver {
public:
    NDeFResult solve(const NDeFProblem& problem) const;
};

}  // namespace neurodic
