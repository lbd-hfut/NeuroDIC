#include "neurodic/problem/pin_multi_problem.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void PINMultiProblem::validate() const {
    // Placeholder only: validation and CMake integration are intentionally
    // deferred until the pairwise execution implementation is introduced.
    if (route_id != "pin_multi_slover")
        throw ValidationError("PIN multi route_id must be pin_multi_slover");
}

}  // namespace neurodic
