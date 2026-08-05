/** C++ orchestration of three existing 2D PIN solves and CPU stereo reconstruction. */
#pragma once

#include "neurodic/core/result.hpp"
#include "neurodic/problem/pin_stereo_problem.hpp"

namespace neurodic {

class PINStereoSolver {
public:
    PINStereoResult solve(const PINStereoProblem& problem) const;
    PINStereoResult reconstruct(const PINResult& reference_disparity, const PINResult& left_temporal,
                                const PINResult& deformed_disparity, const PINStereoProblem& problem) const;
};

}  // namespace neurodic
