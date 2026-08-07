/**
 * @file mono_calibration.hpp
 * @brief Mono calibration placeholder.
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
 * - Internal OpenCV calibration modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef NEURODIC_CALIBRATION_MONO_CALIBRATION_HPP
#define NEURODIC_CALIBRATION_MONO_CALIBRATION_HPP

#include <neurodic/calibration/camera_model.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace neurodic::calibration {

enum class CalibrationBoardType {
    Chessboard,
    SymmetricCircles,
    AsymmetricCircles
};

struct CalibrationBoard {
    CalibrationBoardType type = CalibrationBoardType::Chessboard;
    int rows = 0;
    int cols = 0;
    double spacing = 1.0;

    int point_count() const { return rows * cols; }
    std::vector<Eigen::Vector3d> object_points() const;
};

struct BoardDetectionOptions {
    bool refine_corners = true;
    bool normalize_image = true;
    int max_iterations = 30;
    double epsilon = 1e-3;
};

struct CalibrationDetection {
    bool found = false;
    std::string image_path;
    int image_width = 0;
    int image_height = 0;
    std::vector<Eigen::Vector2d> image_points;
};

struct MonoCalibrationOptions {
    BoardDetectionOptions detection;
    bool estimate_tangential_distortion = true;
    bool estimate_k3 = true;
    int max_iterations = 100;
    double epsilon = 1e-9;
};

struct MonoCalibrationResult {
    CameraModel camera;
    std::vector<Eigen::Matrix3d> board_rotations;
    std::vector<Eigen::Vector3d> board_translations;
    std::vector<double> per_view_errors;
    std::vector<CalibrationDetection> detections;
    double rms_error = 0.0;
};

CalibrationDetection detect_calibration_board(const std::string& image_path,
                                              const CalibrationBoard& board,
                                              const BoardDetectionOptions& options = {});

MonoCalibrationResult calibrate_mono_zhang(const std::vector<std::string>& image_paths,
                                           const CalibrationBoard& board,
                                           const MonoCalibrationOptions& options = {});

MonoCalibrationResult calibrate_mono_from_points(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& image_points,
    int image_width,
    int image_height,
    const MonoCalibrationOptions& options = {});

CameraModel calibrate_mono(int image_count);

}  // namespace neurodic::calibration

#endif // NEURODIC_CALIBRATION_MONO_CALIBRATION_HPP
