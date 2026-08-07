#include "neurodic/solver/pin_multi_slover.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void PINMultiSolver::solve(const PINMultiProblem& problem) const {
    problem.validate();
    throw ValidationError(
        "pin_multi_slover is a reserved route placeholder; pairwise PIN multi-view solving is not implemented yet");
}

}  // namespace neurodic
