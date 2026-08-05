/**
 * @file stereo_calibration.hpp
 * @brief Stereo calibration result and entry point.
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

#ifndef NEURODIC_CALIBRATION_TRADITIONAL_STEREO_CALIBRATION_HPP
#define NEURODIC_CALIBRATION_TRADITIONAL_STEREO_CALIBRATION_HPP

#include <neurodic/calibration/traditional/camera_model.hpp>
#include <neurodic/calibration/traditional/mono_calibration.hpp>

#include <string>

namespace neurodic::calibration {

struct StereoCalibrationOptions {
    BoardDetectionOptions detection;
    bool fix_intrinsics = false;
    bool estimate_tangential_distortion = true;
    bool estimate_k3 = true;
    bool reject_outlier_pairs = false;
    double outlier_mad_factor = 3.5;
    double left_right_error_ratio_threshold = 3.0;
    double left_right_error_abs_threshold = 0.25;
    int min_pairs_after_rejection = 6;
    int max_iterations = 100;
    double epsilon = 1e-9;
};

struct StereoCalibrationResult {
    CameraModel left;
    CameraModel right;
    Eigen::Matrix3d R_lr = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_lr = Eigen::Vector3d::Zero();
    Eigen::Matrix3d essential = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
    std::vector<double> per_pair_errors;
    std::vector<double> per_pair_left_errors;
    std::vector<double> per_pair_right_errors;
    std::vector<double> initial_per_pair_errors;
    std::vector<double> initial_per_pair_left_errors;
    std::vector<double> initial_per_pair_right_errors;
    std::vector<int> kept_pair_indices;
    std::vector<int> rejected_pair_indices;
    std::vector<std::string> rejection_reasons;
    std::vector<CalibrationDetection> left_detections;
    std::vector<CalibrationDetection> right_detections;
    double rms_error = 0.0;
    double initial_rms_error = 0.0;
    bool outlier_rejection_applied = false;
};

StereoCalibrationResult calibrate_stereo_zhang(const std::vector<std::string>& left_image_paths,
                                               const std::vector<std::string>& right_image_paths,
                                               const CalibrationBoard& board,
                                               const StereoCalibrationOptions& options = {});

StereoCalibrationResult calibrate_stereo_from_points(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& left_image_points,
    const std::vector<std::vector<Eigen::Vector2d>>& right_image_points,
    int image_width,
    int image_height,
    const StereoCalibrationOptions& options = {});

StereoCalibrationResult calibrate_stereo(int image_pair_count);

}  // namespace neurodic::calibration

#endif // NEURODIC_CALIBRATION_TRADITIONAL_STEREO_CALIBRATION_HPP
