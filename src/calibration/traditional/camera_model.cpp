/**
 * @file camera_model.cpp
 * @brief Minimal implementation placeholder for camera model.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <neurodic/calibration/traditional/camera_model.hpp>

namespace neurodic::calibration {

Eigen::Matrix<double, 3, 4> CameraModel::projection_matrix() const
{
    Eigen::Matrix<double, 3, 4> extrinsic;
    extrinsic.block<3, 3>(0, 0) = R;
    extrinsic.col(3) = t;
    return K * extrinsic;
}

Eigen::Vector3d CameraModel::camera_center() const { return -R.transpose() * t; }

}  // namespace neurodic::calibration
