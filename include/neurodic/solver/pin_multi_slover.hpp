/** Placeholder orchestration interface for the pin_multi_slover route. */
#pragma once

#include "neurodic/problem/pin_multi_problem.hpp"

namespace neurodic {

// The implementation will run pairwise PIN fields, reconstruct X0/Xk for
// each pair, then fuse pair surfaces and their 3D displacement fields.
class PINMultiSolver {
public:
    void solve(const PINMultiProblem& problem) const;
};

}  // namespace neurodic
