#include "neurodic/solver/pin_stereo_solver.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/geometry/stereo_geometry.hpp"
#include "neurodic/solver/pin_solver.hpp"

namespace neurodic {
namespace {

torch::Tensor inside_image(const torch::Tensor& xy, std::int64_t width, std::int64_t height) {
    return torch::isfinite(xy).all(1) & (xy.select(1, 0) >= 0.0) & (xy.select(1, 0) <= width - 1.0) &
           (xy.select(1, 1) >= 0.0) & (xy.select(1, 1) <= height - 1.0);
}

void validate_field_pair(const PINResult& first, const PINResult& other) {
    if (!first.displacement.coordinates.defined() || !first.displacement.values.defined() ||
        first.displacement.coordinates.dim() != 2 || first.displacement.coordinates.size(1) != 2 ||
        first.displacement.values.sizes() != first.displacement.coordinates.sizes() ||
        other.displacement.coordinates.sizes() != first.displacement.coordinates.sizes() ||
        other.displacement.values.sizes() != first.displacement.values.sizes() ||
        !torch::equal(first.displacement.coordinates, other.displacement.coordinates))
        throw ValidationError("Stereo PIN results must be matching [N,2] fields on the same L0 coordinates");
}

}  // namespace

PINStereoResult PINStereoSolver::solve(const PINStereoProblem& problem) const {
    problem.validate();
    PINSolver planar_solver;
    auto reference_disparity = planar_solver.solve(problem.reference_disparity);
    auto left_temporal = planar_solver.solve(problem.left_temporal);
    auto deformed_disparity = planar_solver.solve(problem.deformed_disparity);
    return reconstruct(reference_disparity, left_temporal, deformed_disparity, problem);
}

PINStereoResult PINStereoSolver::reconstruct(const PINResult& reference_disparity,
                                             const PINResult& left_temporal,
                                             const PINResult& deformed_disparity,
                                             const PINStereoProblem& problem) const {
    problem.validate();
    validate_field_pair(reference_disparity, left_temporal);
    validate_field_pair(reference_disparity, deformed_disparity);
    auto l0 = reference_disparity.displacement.coordinates.detach().to(torch::kCPU).to(torch::kFloat64);
    auto r0 = l0 + reference_disparity.displacement.values.detach().to(torch::kCPU).to(torch::kFloat64);
    auto l1 = l0 + left_temporal.displacement.values.detach().to(torch::kCPU).to(torch::kFloat64);
    auto r1 = l0 + deformed_disparity.displacement.values.detach().to(torch::kCPU).to(torch::kFloat64);
    StereoGeometry geometry(problem.left_camera, problem.right_camera, problem.reconstruction);
    auto reference = geometry.reconstruct_reference(l0, r0);
    auto current = geometry.reconstruct_current(l1, r1);
    auto valid = reference.valid & current.valid & torch::isfinite(l0).all(1) & torch::isfinite(r0).all(1) &
                 torch::isfinite(l1).all(1) & torch::isfinite(r1).all(1);
    if (problem.require_image_bounds) {
        const auto left_width = problem.left_camera.image_width;
        const auto left_height = problem.left_camera.image_height;
        const auto right_width = problem.right_camera.image_width;
        const auto right_height = problem.right_camera.image_height;
        if (left_width <= 0 || left_height <= 0 || right_width <= 0 || right_height <= 0)
            throw ValidationError("Image-bound filtering requires positive camera image dimensions");
        valid &= inside_image(l0, left_width, left_height) & inside_image(l1, left_width, left_height) &
                 inside_image(r0, right_width, right_height) & inside_image(r1, right_width, right_height);
    }
    auto reference_points = reference.points * problem.world_scale;
    auto current_points = current.points * problem.world_scale;
    return {reference_disparity, left_temporal, deformed_disparity,
            l0, r0, l1, r1, reference_points, current_points,
            geometry.displacement_3d(reference_points, current_points), valid,
            reference.max_reprojection_error, current.max_reprojection_error};
}

}  // namespace neurodic
