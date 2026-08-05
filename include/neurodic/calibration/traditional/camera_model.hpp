/**
 * @file camera_model.hpp
 * @brief Pinhole camera model with projection matrix.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef NEURODIC_CALIBRATION_TRADITIONAL_CAMERA_MODEL_HPP
#define NEURODIC_CALIBRATION_TRADITIONAL_CAMERA_MODEL_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace neurodic::calibration {

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

#endif // NEURODIC_CALIBRATION_TRADITIONAL_CAMERA_MODEL_HPP
