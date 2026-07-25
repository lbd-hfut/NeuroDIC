#include "neurodic/solver/pin_solver.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

PINResult PINSolver::solve(const PINProblem& problem) const {
    problem.validate();
    throw NotImplementedScientificError(
        "TODO(NeuroDIC): implement unified PIN-DIC 2D/stereo solver after B-spline autograd validation"
    );
}

}  // namespace neurodic
