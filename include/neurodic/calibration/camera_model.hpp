/** Fixed pinhole camera calibration parameters used before/after DIC solving. */
#pragma once

#include <cstdint>
#include <string>
#include <torch/torch.h>

namespace neurodic {

struct CameraModel {
    torch::Tensor intrinsics;  // CPU float64 [3,3], K
    torch::Tensor rotation;    // CPU float64 [3,3], world -> camera R
    torch::Tensor translation; // CPU float64 [3], world -> camera t
    torch::Tensor distortion;  // CPU float64 [N], OpenCV radial/tangential order
    std::int64_t image_width{0};
    std::int64_t image_height{0};
    double rms_error{0.0};
    std::string label;

    void validate() const;
    torch::Tensor projection_matrix() const;  // K [R|t], [3,4]
    torch::Tensor camera_center() const;      // -R^T t, [3]
};

}  // namespace neurodic
