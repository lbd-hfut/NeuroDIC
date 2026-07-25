#include "neurodic/solver/ndef_solver.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

NDeFResult NDeFSolver::solve(const NDeFProblem& problem) const {
    problem.validate();
    throw NotImplementedScientificError(
        "TODO(NeuroDIC): implement NDeF solver with internally controlled topology after geometry validation"
    );
}

}  // namespace neurodic
