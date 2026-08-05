#include "neurodic/geometry/stereo_geometry.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

StereoGeometry::StereoGeometry(CameraModel left_camera, CameraModel right_camera, ReconstructionOptions options)
    : left_camera_(std::move(left_camera)), right_camera_(std::move(right_camera)), options_(options) {
    left_camera_.validate();
    right_camera_.validate();
}

ReconstructionResult StereoGeometry::reconstruct_reference(const torch::Tensor& left_coordinates,
                                                           const torch::Tensor& right_coordinates) const {
    return triangulate_stereo(left_coordinates, right_coordinates, left_camera_, right_camera_, options_);
}

ReconstructionResult StereoGeometry::reconstruct_current(const torch::Tensor& left_coordinates,
                                                         const torch::Tensor& right_coordinates) const {
    return triangulate_stereo(left_coordinates, right_coordinates, left_camera_, right_camera_, options_);
}

torch::Tensor StereoGeometry::displacement_3d(const torch::Tensor& reference, const torch::Tensor& current) const {
    if (reference.sizes() != current.sizes() || reference.dim() != 2 || reference.size(1) != 3)
        throw ValidationError("Stereo 3D displacement expects matching [N,3] fields");
    return current - reference;
}

}  // namespace neurodic
