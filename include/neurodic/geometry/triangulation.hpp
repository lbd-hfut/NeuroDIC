/** CPU float64 reconstruction through undistorted DLT triangulation. */
#pragma once

#include <vector>

#include <torch/torch.h>

#include "neurodic/calibration/camera_model.hpp"

namespace neurodic {

struct ReconstructionOptions {
    double max_reprojection_error{2.0};
    bool require_positive_depth{true};
    int undistort_iterations{12};
};

struct ReconstructionResult {
    torch::Tensor points;                    // CPU float64 [N,3]
    torch::Tensor valid;                     // CPU bool [N]
    torch::Tensor mean_reprojection_error;   // CPU float64 [N]
    torch::Tensor max_reprojection_error;    // CPU float64 [N]
    torch::Tensor observations_used;         // CPU int64 [N]
};

ReconstructionResult triangulate_multiview(const torch::Tensor& observations,
                                           const std::vector<CameraModel>& cameras,
                                           const torch::Tensor& observation_valid = {},
                                           ReconstructionOptions options = {});
ReconstructionResult triangulate_stereo(const torch::Tensor& left_coordinates,
                                        const torch::Tensor& right_coordinates,
                                        const CameraModel& left_camera, const CameraModel& right_camera,
                                        ReconstructionOptions options = {});

}  // namespace neurodic
