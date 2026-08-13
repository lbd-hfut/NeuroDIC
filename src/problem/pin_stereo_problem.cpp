#include "neurodic/problem/pin_stereo_problem.hpp"

#include <cmath>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

PINStereoProblem::PINStereoProblem(PINProblem reference_disparity_problem,
                                   PINProblem left_temporal_problem,
                                   PINProblem deformed_disparity_problem,
                                   CameraModel left, CameraModel right)
    : reference_disparity(std::move(reference_disparity_problem)),
      left_temporal(std::move(left_temporal_problem)),
      deformed_disparity(std::move(deformed_disparity_problem)),
      left_camera(std::move(left)), right_camera(std::move(right)) {}

void PINStereoProblem::validate() const {
    reference_disparity.validate();
    left_temporal.validate();
    deformed_disparity.validate();
    left_camera.validate();
    right_camera.validate();
    const auto shape = reference_disparity.reference_image.sizes();
    if (left_temporal.reference_image.sizes() != shape || deformed_disparity.reference_image.sizes() != shape ||
        reference_disparity.roi_mask.sizes() != left_temporal.roi_mask.sizes() ||
        reference_disparity.roi_mask.sizes() != deformed_disparity.roi_mask.sizes() ||
        !torch::equal(reference_disparity.roi_mask, left_temporal.roi_mask) ||
        !torch::equal(reference_disparity.roi_mask, deformed_disparity.roi_mask))
        throw ValidationError("Stereo PIN fields must share the L0 image shape and ROI");
    if (!std::isfinite(world_scale) || world_scale <= 0.0)
        throw ValidationError("Stereo PIN world_scale must be finite and positive");
    if (traditional_strain_neighbors < 3)
        throw ValidationError("Stereo PIN traditional_strain_neighbors must be at least 3");
}

}  // namespace neurodic
