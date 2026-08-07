/** Fixed pinhole camera calibration parameters used before/after DIC solving. */
#pragma once

#include <cstdint>
#include <string>
#include <torch/torch.h>

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
#include <Eigen/Dense>
#include <vector>
#endif

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

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
namespace neurodic::calibration {

// Eigen representation used by feature matching, PnP, triangulation, and BA.
// The Torch representation above remains the core DIC geometry data model.
struct CameraModel {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    std::vector<double> distortion;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int image_width = 0;
    int image_height = 0;
    double rms_error = 0.0;
    std::string label;

    Eigen::Matrix<double, 3, 4> projection_matrix() const;
    Eigen::Vector3d camera_center() const;
};

}  // namespace neurodic::calibration
#endif
