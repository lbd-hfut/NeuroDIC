/** Pairwise orchestration for the independent multi-camera PIN-DIC route.

 *  Every selected camera pair is solved independently: the three planar PIN
 *  fields (A0->B0, A0->Ak, A0->Bk) feed the validated stereo reconstruction
 *  path, and each pair keeps its own X0/Xk/dX products.  Fusion of the
 *  pairwise surfaces is a later, separately enabled stage.
 */
#pragma once

#include "neurodic/core/result.hpp"
#include "neurodic/problem/pin_multi_problem.hpp"

namespace neurodic {

class PINMultiSolver {
public:
    PINMultiResult solve(const PINMultiProblem& problem) const;
};

}  // namespace neurodic
