/**
 * @file multiview_calibration.hpp
 * @brief Multiview calibration placeholder.
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

#ifndef NEURODIC_CALIBRATION_TRADITIONAL_MULTIVIEW_CALIBRATION_HPP
#define NEURODIC_CALIBRATION_TRADITIONAL_MULTIVIEW_CALIBRATION_HPP

#include <neurodic/calibration/traditional/camera_model.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace neurodic::calibration {

struct FeatureTrackObservation {
    int image_index = -1;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
};

struct SparsePoint3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    std::vector<FeatureTrackObservation> observations;
    double reprojection_error = 0.0;
};

struct MultiviewCalibrationOptions {
    // CPU SIFT / matching defaults aligned with COLMAP.
    int max_features = 8192;
    double match_ratio = 0.8;
    double ransac_reprojection_threshold = 4.0;
    // COLMAP Mapper.filter_min_tri_angle.
    double min_triangulation_angle_degrees = 1.5;
    // COLMAP Mapper.init_min_num_inliers.
    int min_inlier_matches = 100;
    int matching_window = 2;
    bool wrap_matching = true;
    // -1/-1 selects the strongest valid pair automatically.
    int initial_image1 = -1;
    int initial_image2 = -1;
    double initial_focal_length_factor = 1.2;
    double abs_pose_max_error = 12.0;
    int abs_pose_min_num_inliers = 30;
    double abs_pose_min_inlier_ratio = 0.25;
    double filter_max_reproj_error = 4.0;
    int ba_local_num_images = 6;
    // COLMAP defaults to true, but the current windowed graph needs these
    // tracks until full global retriangulation is migrated.
    bool ignore_two_view_tracks = false;
    bool refine_bundle = true;
    bool refine_focal_length = false;
    bool refine_principal_point = false;
    bool refine_extra_params = false;
    bool share_intrinsics = false;
    std::vector<CameraModel> initial_cameras;
};

struct MultiviewCalibrationResult {
    std::vector<CameraModel> cameras;
    std::vector<SparsePoint3D> sparse_points;
    std::vector<std::vector<int>> inlier_match_counts;
    double mean_reprojection_error = 0.0;
};

struct MultiviewScaleObservation {
    int camera_index = -1;
    std::vector<Eigen::Vector2d> image_points;
};

struct MultiviewScaleOptions {
    int board_rows = 0;
    int board_cols = 0;
    double square_size = 1.0;
    double max_reprojection_error = 3.0;
    double trim_fraction = 0.20;
    int min_common_corners = 12;
};

struct MultiviewScaleResult {
    double sfm_to_world_scale = 1.0;
    double world_to_sfm_scale = 1.0;
    double sfm_square_size_mean = 0.0;
    double sfm_square_size_median = 0.0;
    double sfm_square_size_std = 0.0;
    double edge_cv = 0.0;
    int triangulated_corners = 0;
    int valid_edges = 0;
    std::vector<Eigen::Vector3d> triangulated_board_points_sfm;
    std::vector<double> edge_lengths_sfm;
    std::vector<CameraModel> scaled_cameras;
    std::vector<SparsePoint3D> scaled_sparse_points;
};

MultiviewCalibrationResult calibrate_multiview_colmap_like(
    const std::vector<std::string>& image_paths,
    const MultiviewCalibrationOptions& options = {});

MultiviewScaleResult estimate_multiview_chessboard_scale(
    const std::vector<CameraModel>& cameras,
    const std::vector<SparsePoint3D>& sparse_points,
    const std::vector<MultiviewScaleObservation>& observations,
    const MultiviewScaleOptions& options);

std::vector<CameraModel> calibrate_multiview(int image_count);

}  // namespace neurodic::calibration

#endif // NEURODIC_CALIBRATION_TRADITIONAL_MULTIVIEW_CALIBRATION_HPP
