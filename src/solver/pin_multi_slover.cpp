#include "neurodic/solver/pin_multi_slover.hpp"

#include <utility>

#include "neurodic/problem/pin_stereo_problem.hpp"
#include "neurodic/solver/pin_stereo_solver.hpp"

namespace neurodic {

PINMultiResult PINMultiSolver::solve(const PINMultiProblem& problem) const {
    problem.validate();
    PINMultiResult result;
    result.pairs.reserve(problem.pairs.size());
    PINStereoSolver stereo_solver;
    for (const auto& pair : problem.pairs) {
        PINStereoProblem stereo(
            pair.reference_stereo, pair.left_temporal, pair.deformed_stereo,
            pair.left_camera, pair.right_camera);
        stereo.world_scale = problem.world_scale;
        stereo.require_image_bounds = problem.require_image_bounds;
        stereo.reconstruction = problem.reconstruction;
        // Pair fields are an intermediate reconstruction product.  Estimate
        // traditional strain only after the pair surfaces have been fused.
        stereo.compute_traditional_strain = false;
        result.pairs.push_back({pair.pair_id, stereo_solver.solve(stereo)});
    }
    return result;
}

}  // namespace neurodic
