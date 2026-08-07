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

#ifndef NEURODIC_CALIBRATION_MULTIVIEW_CALIBRATION_HPP
#define NEURODIC_CALIBRATION_MULTIVIEW_CALIBRATION_HPP

#include <neurodic/calibration/camera_model.hpp>
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

// Per-point traceable diagnostics for the final sparse model: creation source,
// track composition, per-observation reprojection errors, triangulation angles,
// cheirality, and bundle-adjusted state before/after the final BA plus the
// outcome of the final general geometric filter.
struct SparsePointDiagnostic {
    int point_id = -1;
    int track_length = 0;
    std::vector<int> images;
    std::vector<double> per_observation_errors;
    double max_triangulation_angle_degrees = 0.0;
    double median_triangulation_angle_degrees = 0.0;
    bool all_positive_depth = true;
    std::string creation_source = "unknown";  // create | merge | retriangulate
    double max_depth_ratio = 0.0;  // max/min observation depth (0 if < 3 views)
    Eigen::Vector3d xyz_before_final_ba = Eigen::Vector3d::Zero();
    double rms_before_final_ba = 0.0;
    Eigen::Vector3d xyz_after_final_ba = Eigen::Vector3d::Zero();
    double rms_after_final_ba = 0.0;
    bool kept_by_final_filter = true;
};

struct MultiviewCalibrationOptions {
    // CPU SIFT / matching defaults aligned with COLMAP.
    int max_features = 8192;
    double match_ratio = 0.8;
    double ransac_reprojection_threshold = 4.0;
    // COLMAP SIFT contrast threshold (OpenCV default 0.04; lower keeps more
    // weak features, higher keeps fewer but stronger ones).
    double sift_contrast_threshold = 0.04;
    // RootSIFT descriptor normalization (L1 then element-wise sqrt) to
    // approximate COLMAP's descriptor normalization.
    bool root_sift = false;
    // Mutually-consistent (bidirectional) ratio-test matching: keep only
    // one-to-one matches that survive the ratio test in both directions.
    bool bidirectional_matching = false;
    // COLMAP Mapper.filter_min_tri_angle.
    double min_triangulation_angle_degrees = 1.5;
    // COLMAP Mapper.init_min_num_inliers.
    int min_inlier_matches = 100;
    // "window": ring/sequential matching limited by matching_window and
    //            wrap_matching (legacy NeuroDIC behavior).
    // "exhaustive": match every image pair (COLMAP match_exhaustive), the
    //            graph then only filters pairs by geometric verification and
    //            the inlier-count threshold.
    std::string matching_mode = "window";
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
    // COLMAP Mapper.init_max_forward_motion: reject initial pairs whose
    // relative translation is dominated by the forward component |t_z|/|t|.
    double init_max_forward_motion = 0.95;
    // COLMAP Mapper.init_min_tri_angle (degrees): minimum median triangulation
    // angle of an initial pair's inlier matches. A stronger threshold makes the
    // initial pair selection prefer wide-baseline seeds (weak adjacent-pair
    // seeds drift together during incremental BA on ring datasets).
    double init_min_tri_angle_degrees = 16.0;
    // COLMAP Mapper.max_reg_trials: cap on registration attempts per image.
    int max_reg_trials = 3;
    // Number of outer initialization trials, mirroring COLMAP's
    // init_num_trials loop with threshold relaxation.
    int init_num_trials = 10;
    // Structure-less (2D-2D epipolar) registration fallback. Kept off by
    // default: the official estimator relies on an epipolar minimal solver
    // that is not part of this codebase; when enabled, images that fail PnP
    // are skipped (never aborting the loop) and retried after global
    // refinement, which covers the common failure modes of this estimator.
    bool structure_less_registration_fallback = false;
    // IncrementalTriangulator::Options equivalents.
    double create_max_angle_error_degrees = 2.0;
    double continue_max_angle_error_degrees = 2.0;
    double re_max_angle_error_degrees = 5.0;
    double re_min_ratio = 0.2;
    int re_max_trials = 1;
    // COLMAP pipeline refinement loop parameters.
    int ba_local_max_refinements = 2;
    double ba_local_max_refinement_change = 0.001;
    int ba_global_max_refinements = 5;
    double ba_global_max_refinement_change = 0.0005;
    // COLMAP reconstruction->Normalize() after the initial pair and during
    // global refinement (unit-extent bbox around the point centroid).
    bool normalize_reconstruction = true;
    // After the final global refinement, run one additional global bundle
    // adjustment that releases only the shared SIMPLE_PINHOLE focal length
    // (principal point and distortion stay fixed). Mirrors the staged
    // PyCOLMAP reference run used for the CylinderDIC case.
    bool final_refine_focal_length = false;
    // Final general geometric filter applied once after the last bundle
    // adjustment (never radius/ground-truth based):
    // - minimum track length (multi-view points are verifiable; 2-view points
    //   cannot be cross-checked and are the dominant extreme-outlier source);
    // - per-observation reprojection error <= filter_max_reproj_error;
    // - positive depth in every observation;
    // - min triangulation angle >= min_triangulation_angle_degrees;
    // - depth consistency for multi-view tracks (max/min observation depth).
    int final_min_track_length = 3;
    double final_max_depth_ratio = 4.0;
    std::vector<CameraModel> initial_cameras;
};

struct MultiviewStageStat {
    std::string stage;
    int num_registered_cameras = 0;
    int num_points3d = 0;
    int num_observations = 0;
    double mean_reprojection_error = 0.0;
    double focal_length = 0.0;
    double principal_point_x = 0.0;
    double principal_point_y = 0.0;
    double distortion_k1 = 0.0;
};

struct MultiviewRegistrationAttempt {
    int image_index = -1;
    bool success = false;
    std::string reason;
    int num_visible_points = 0;
    int num_pnp_correspondences = 0;
    int num_pnp_inliers = 0;
};

struct MultiviewCalibrationResult {
    std::vector<CameraModel> cameras;
    std::vector<SparsePoint3D> sparse_points;
    std::vector<std::vector<int>> inlier_match_counts;
    double mean_reprojection_error = 0.0;
    // Per-stage diagnostics aligned with COLMAP's incremental pipeline logs.
    std::vector<MultiviewStageStat> stage_stats;
    // Every image registration attempt (successes and failures with reason).
    std::vector<MultiviewRegistrationAttempt> registration_attempts;
    // Free-form pipeline log lines (initialization trials, relaxations, ...).
    std::vector<std::string> pipeline_log;
    // Traceable per-point diagnostics (creation source, per-obs errors, angles,
    // BA-before/after state, final-filter outcome).
    std::vector<SparsePointDiagnostic> point_diagnostics;
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

#endif // NEURODIC_CALIBRATION_MULTIVIEW_CALIBRATION_HPP
