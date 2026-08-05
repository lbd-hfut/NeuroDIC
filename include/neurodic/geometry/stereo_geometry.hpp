/** CPU stereo reconstruction wrapper for post-solve 3D fields. */
#pragma once

#include "neurodic/geometry/geometry.hpp"
#include "neurodic/geometry/triangulation.hpp"

namespace neurodic {

class StereoGeometry : public Geometry {
public:
    StereoGeometry(CameraModel left_camera, CameraModel right_camera, ReconstructionOptions options = {});
    ReconstructionResult reconstruct_reference(const torch::Tensor& left_coordinates,
                                               const torch::Tensor& right_coordinates) const;
    ReconstructionResult reconstruct_current(const torch::Tensor& left_coordinates,
                                             const torch::Tensor& right_coordinates) const;
    torch::Tensor displacement_3d(const torch::Tensor& reference, const torch::Tensor& current) const;

private:
    CameraModel left_camera_;
    CameraModel right_camera_;
    ReconstructionOptions options_;
};

}  // namespace neurodic
