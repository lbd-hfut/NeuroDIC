#include <neurodic/calibration/mono_calibration.hpp>
#include <neurodic/calibration/multiview_calibration.hpp>
#include <neurodic/calibration/calibration_result.hpp>
#include <neurodic/calibration/stereo_calibration.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_opencv_calibration(py::module_& m)
{
    auto sub = m.def_submodule("calibration");

    py::enum_<neurodic::calibration::CalibrationBoardType>(sub, "CalibrationBoardType")
        .value("Chessboard", neurodic::calibration::CalibrationBoardType::Chessboard)
        .value("SymmetricCircles", neurodic::calibration::CalibrationBoardType::SymmetricCircles)
        .value("AsymmetricCircles", neurodic::calibration::CalibrationBoardType::AsymmetricCircles);

    py::class_<neurodic::calibration::CameraModel>(sub, "CameraModel")
        .def(py::init<>())
        .def_readwrite("K", &neurodic::calibration::CameraModel::K)
        .def_readwrite("distortion", &neurodic::calibration::CameraModel::distortion)
        .def_readwrite("R", &neurodic::calibration::CameraModel::R)
        .def_readwrite("t", &neurodic::calibration::CameraModel::t)
        .def_readwrite("image_width", &neurodic::calibration::CameraModel::image_width)
        .def_readwrite("image_height", &neurodic::calibration::CameraModel::image_height)
        .def_readwrite("rms_error", &neurodic::calibration::CameraModel::rms_error)
        .def_readwrite("label", &neurodic::calibration::CameraModel::label)
        .def("projection_matrix", &neurodic::calibration::CameraModel::projection_matrix)
        .def("camera_center", &neurodic::calibration::CameraModel::camera_center);

    py::class_<neurodic::calibration::CalibrationBoard>(sub, "CalibrationBoard")
        .def(py::init<>())
        .def_readwrite("type", &neurodic::calibration::CalibrationBoard::type)
        .def_readwrite("rows", &neurodic::calibration::CalibrationBoard::rows)
        .def_readwrite("cols", &neurodic::calibration::CalibrationBoard::cols)
        .def_readwrite("spacing", &neurodic::calibration::CalibrationBoard::spacing)
        .def("point_count", &neurodic::calibration::CalibrationBoard::point_count)
        .def("object_points", &neurodic::calibration::CalibrationBoard::object_points);

    py::class_<neurodic::calibration::BoardDetectionOptions>(sub, "BoardDetectionOptions")
        .def(py::init<>())
        .def_readwrite("refine_corners", &neurodic::calibration::BoardDetectionOptions::refine_corners)
        .def_readwrite("normalize_image", &neurodic::calibration::BoardDetectionOptions::normalize_image)
        .def_readwrite("max_iterations", &neurodic::calibration::BoardDetectionOptions::max_iterations)
        .def_readwrite("epsilon", &neurodic::calibration::BoardDetectionOptions::epsilon);

    py::class_<neurodic::calibration::CalibrationDetection>(sub, "CalibrationDetection")
        .def(py::init<>())
        .def_readwrite("found", &neurodic::calibration::CalibrationDetection::found)
        .def_readwrite("image_path", &neurodic::calibration::CalibrationDetection::image_path)
        .def_readwrite("image_width", &neurodic::calibration::CalibrationDetection::image_width)
        .def_readwrite("image_height", &neurodic::calibration::CalibrationDetection::image_height)
        .def_readwrite("image_points", &neurodic::calibration::CalibrationDetection::image_points);

    py::class_<neurodic::calibration::MonoCalibrationOptions>(sub, "MonoCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("detection", &neurodic::calibration::MonoCalibrationOptions::detection)
        .def_readwrite("estimate_tangential_distortion", &neurodic::calibration::MonoCalibrationOptions::estimate_tangential_distortion)
        .def_readwrite("estimate_k3", &neurodic::calibration::MonoCalibrationOptions::estimate_k3)
        .def_readwrite("max_iterations", &neurodic::calibration::MonoCalibrationOptions::max_iterations)
        .def_readwrite("epsilon", &neurodic::calibration::MonoCalibrationOptions::epsilon);

    py::class_<neurodic::calibration::MonoCalibrationResult>(sub, "MonoCalibrationResult")
        .def(py::init<>())
        .def_readwrite("camera", &neurodic::calibration::MonoCalibrationResult::camera)
        .def_readwrite("board_rotations", &neurodic::calibration::MonoCalibrationResult::board_rotations)
        .def_readwrite("board_translations", &neurodic::calibration::MonoCalibrationResult::board_translations)
        .def_readwrite("per_view_errors", &neurodic::calibration::MonoCalibrationResult::per_view_errors)
        .def_readwrite("detections", &neurodic::calibration::MonoCalibrationResult::detections)
        .def_readwrite("rms_error", &neurodic::calibration::MonoCalibrationResult::rms_error);

    py::class_<neurodic::calibration::StereoCalibrationOptions>(sub, "StereoCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("detection", &neurodic::calibration::StereoCalibrationOptions::detection)
        .def_readwrite("fix_intrinsics", &neurodic::calibration::StereoCalibrationOptions::fix_intrinsics)
        .def_readwrite("estimate_tangential_distortion", &neurodic::calibration::StereoCalibrationOptions::estimate_tangential_distortion)
        .def_readwrite("estimate_k3", &neurodic::calibration::StereoCalibrationOptions::estimate_k3)
        .def_readwrite("reject_outlier_pairs", &neurodic::calibration::StereoCalibrationOptions::reject_outlier_pairs)
        .def_readwrite("outlier_mad_factor", &neurodic::calibration::StereoCalibrationOptions::outlier_mad_factor)
        .def_readwrite("left_right_error_ratio_threshold", &neurodic::calibration::StereoCalibrationOptions::left_right_error_ratio_threshold)
        .def_readwrite("left_right_error_abs_threshold", &neurodic::calibration::StereoCalibrationOptions::left_right_error_abs_threshold)
        .def_readwrite("min_pairs_after_rejection", &neurodic::calibration::StereoCalibrationOptions::min_pairs_after_rejection)
        .def_readwrite("max_iterations", &neurodic::calibration::StereoCalibrationOptions::max_iterations)
        .def_readwrite("epsilon", &neurodic::calibration::StereoCalibrationOptions::epsilon);

    py::class_<neurodic::calibration::StereoCalibrationResult>(sub, "StereoCalibrationResult")
        .def(py::init<>())
        .def_readwrite("left", &neurodic::calibration::StereoCalibrationResult::left)
        .def_readwrite("right", &neurodic::calibration::StereoCalibrationResult::right)
        .def_readwrite("R_lr", &neurodic::calibration::StereoCalibrationResult::R_lr)
        .def_readwrite("t_lr", &neurodic::calibration::StereoCalibrationResult::t_lr)
        .def_readwrite("essential", &neurodic::calibration::StereoCalibrationResult::essential)
        .def_readwrite("fundamental", &neurodic::calibration::StereoCalibrationResult::fundamental)
        .def_readwrite("per_pair_errors", &neurodic::calibration::StereoCalibrationResult::per_pair_errors)
        .def_readwrite("per_pair_left_errors", &neurodic::calibration::StereoCalibrationResult::per_pair_left_errors)
        .def_readwrite("per_pair_right_errors", &neurodic::calibration::StereoCalibrationResult::per_pair_right_errors)
        .def_readwrite("initial_per_pair_errors", &neurodic::calibration::StereoCalibrationResult::initial_per_pair_errors)
        .def_readwrite("initial_per_pair_left_errors", &neurodic::calibration::StereoCalibrationResult::initial_per_pair_left_errors)
        .def_readwrite("initial_per_pair_right_errors", &neurodic::calibration::StereoCalibrationResult::initial_per_pair_right_errors)
        .def_readwrite("kept_pair_indices", &neurodic::calibration::StereoCalibrationResult::kept_pair_indices)
        .def_readwrite("rejected_pair_indices", &neurodic::calibration::StereoCalibrationResult::rejected_pair_indices)
        .def_readwrite("rejection_reasons", &neurodic::calibration::StereoCalibrationResult::rejection_reasons)
        .def_readwrite("left_detections", &neurodic::calibration::StereoCalibrationResult::left_detections)
        .def_readwrite("right_detections", &neurodic::calibration::StereoCalibrationResult::right_detections)
        .def_readwrite("rms_error", &neurodic::calibration::StereoCalibrationResult::rms_error)
        .def_readwrite("initial_rms_error", &neurodic::calibration::StereoCalibrationResult::initial_rms_error)
        .def_readwrite("outlier_rejection_applied", &neurodic::calibration::StereoCalibrationResult::outlier_rejection_applied);

    py::class_<neurodic::calibration::FeatureTrackObservation>(sub, "FeatureTrackObservation")
        .def(py::init<>())
        .def_readwrite("image_index", &neurodic::calibration::FeatureTrackObservation::image_index)
        .def_readwrite("point", &neurodic::calibration::FeatureTrackObservation::point);

    py::class_<neurodic::calibration::SparsePoint3D>(sub, "SparsePoint3D")
        .def(py::init<>())
        .def_readwrite("point", &neurodic::calibration::SparsePoint3D::point)
        .def_readwrite("observations", &neurodic::calibration::SparsePoint3D::observations)
        .def_readwrite("reprojection_error", &neurodic::calibration::SparsePoint3D::reprojection_error);

    py::class_<neurodic::calibration::SparsePointDiagnostic>(sub, "SparsePointDiagnostic")
        .def(py::init<>())
        .def_readwrite("point_id", &neurodic::calibration::SparsePointDiagnostic::point_id)
        .def_readwrite("track_length", &neurodic::calibration::SparsePointDiagnostic::track_length)
        .def_readwrite("images", &neurodic::calibration::SparsePointDiagnostic::images)
        .def_readwrite("per_observation_errors", &neurodic::calibration::SparsePointDiagnostic::per_observation_errors)
        .def_readwrite("max_triangulation_angle_degrees", &neurodic::calibration::SparsePointDiagnostic::max_triangulation_angle_degrees)
        .def_readwrite("median_triangulation_angle_degrees", &neurodic::calibration::SparsePointDiagnostic::median_triangulation_angle_degrees)
        .def_readwrite("all_positive_depth", &neurodic::calibration::SparsePointDiagnostic::all_positive_depth)
        .def_readwrite("creation_source", &neurodic::calibration::SparsePointDiagnostic::creation_source)
        .def_readwrite("max_depth_ratio", &neurodic::calibration::SparsePointDiagnostic::max_depth_ratio)
        .def_readwrite("xyz_before_final_ba", &neurodic::calibration::SparsePointDiagnostic::xyz_before_final_ba)
        .def_readwrite("rms_before_final_ba", &neurodic::calibration::SparsePointDiagnostic::rms_before_final_ba)
        .def_readwrite("xyz_after_final_ba", &neurodic::calibration::SparsePointDiagnostic::xyz_after_final_ba)
        .def_readwrite("rms_after_final_ba", &neurodic::calibration::SparsePointDiagnostic::rms_after_final_ba)
        .def_readwrite("kept_by_final_filter", &neurodic::calibration::SparsePointDiagnostic::kept_by_final_filter);

    py::class_<neurodic::calibration::MultiviewCalibrationOptions>(sub, "MultiviewCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("max_features", &neurodic::calibration::MultiviewCalibrationOptions::max_features)
        .def_readwrite("match_ratio", &neurodic::calibration::MultiviewCalibrationOptions::match_ratio)
        .def_readwrite("sift_contrast_threshold", &neurodic::calibration::MultiviewCalibrationOptions::sift_contrast_threshold)
        .def_readwrite("root_sift", &neurodic::calibration::MultiviewCalibrationOptions::root_sift)
        .def_readwrite("bidirectional_matching", &neurodic::calibration::MultiviewCalibrationOptions::bidirectional_matching)
        .def_readwrite("ransac_reprojection_threshold", &neurodic::calibration::MultiviewCalibrationOptions::ransac_reprojection_threshold)
        .def_readwrite("min_triangulation_angle_degrees", &neurodic::calibration::MultiviewCalibrationOptions::min_triangulation_angle_degrees)
        .def_readwrite("min_inlier_matches", &neurodic::calibration::MultiviewCalibrationOptions::min_inlier_matches)
        .def_readwrite("matching_mode", &neurodic::calibration::MultiviewCalibrationOptions::matching_mode)
        .def_readwrite("matching_window", &neurodic::calibration::MultiviewCalibrationOptions::matching_window)
        .def_readwrite("wrap_matching", &neurodic::calibration::MultiviewCalibrationOptions::wrap_matching)
        .def_readwrite("initial_image1", &neurodic::calibration::MultiviewCalibrationOptions::initial_image1)
        .def_readwrite("initial_image2", &neurodic::calibration::MultiviewCalibrationOptions::initial_image2)
        .def_readwrite("initial_focal_length_factor", &neurodic::calibration::MultiviewCalibrationOptions::initial_focal_length_factor)
        .def_readwrite("abs_pose_max_error", &neurodic::calibration::MultiviewCalibrationOptions::abs_pose_max_error)
        .def_readwrite("abs_pose_min_num_inliers", &neurodic::calibration::MultiviewCalibrationOptions::abs_pose_min_num_inliers)
        .def_readwrite("abs_pose_min_inlier_ratio", &neurodic::calibration::MultiviewCalibrationOptions::abs_pose_min_inlier_ratio)
        .def_readwrite("filter_max_reproj_error", &neurodic::calibration::MultiviewCalibrationOptions::filter_max_reproj_error)
        .def_readwrite("ba_local_num_images", &neurodic::calibration::MultiviewCalibrationOptions::ba_local_num_images)
        .def_readwrite("ignore_two_view_tracks", &neurodic::calibration::MultiviewCalibrationOptions::ignore_two_view_tracks)
        .def_readwrite("refine_bundle", &neurodic::calibration::MultiviewCalibrationOptions::refine_bundle)
        .def_readwrite("refine_focal_length", &neurodic::calibration::MultiviewCalibrationOptions::refine_focal_length)
        .def_readwrite("refine_principal_point", &neurodic::calibration::MultiviewCalibrationOptions::refine_principal_point)
        .def_readwrite("refine_extra_params", &neurodic::calibration::MultiviewCalibrationOptions::refine_extra_params)
        .def_readwrite("share_intrinsics", &neurodic::calibration::MultiviewCalibrationOptions::share_intrinsics)
        .def_readwrite("init_max_forward_motion", &neurodic::calibration::MultiviewCalibrationOptions::init_max_forward_motion)
        .def_readwrite("init_min_tri_angle_degrees", &neurodic::calibration::MultiviewCalibrationOptions::init_min_tri_angle_degrees)
        .def_readwrite("max_reg_trials", &neurodic::calibration::MultiviewCalibrationOptions::max_reg_trials)
        .def_readwrite("init_num_trials", &neurodic::calibration::MultiviewCalibrationOptions::init_num_trials)
        .def_readwrite("structure_less_registration_fallback", &neurodic::calibration::MultiviewCalibrationOptions::structure_less_registration_fallback)
        .def_readwrite("create_max_angle_error_degrees", &neurodic::calibration::MultiviewCalibrationOptions::create_max_angle_error_degrees)
        .def_readwrite("continue_max_angle_error_degrees", &neurodic::calibration::MultiviewCalibrationOptions::continue_max_angle_error_degrees)
        .def_readwrite("re_max_angle_error_degrees", &neurodic::calibration::MultiviewCalibrationOptions::re_max_angle_error_degrees)
        .def_readwrite("re_min_ratio", &neurodic::calibration::MultiviewCalibrationOptions::re_min_ratio)
        .def_readwrite("re_max_trials", &neurodic::calibration::MultiviewCalibrationOptions::re_max_trials)
        .def_readwrite("ba_local_max_refinements", &neurodic::calibration::MultiviewCalibrationOptions::ba_local_max_refinements)
        .def_readwrite("ba_local_max_refinement_change", &neurodic::calibration::MultiviewCalibrationOptions::ba_local_max_refinement_change)
        .def_readwrite("ba_global_max_refinements", &neurodic::calibration::MultiviewCalibrationOptions::ba_global_max_refinements)
        .def_readwrite("ba_global_max_refinement_change", &neurodic::calibration::MultiviewCalibrationOptions::ba_global_max_refinement_change)
        .def_readwrite("normalize_reconstruction", &neurodic::calibration::MultiviewCalibrationOptions::normalize_reconstruction)
        .def_readwrite("final_refine_focal_length", &neurodic::calibration::MultiviewCalibrationOptions::final_refine_focal_length)
        .def_readwrite("final_min_track_length", &neurodic::calibration::MultiviewCalibrationOptions::final_min_track_length)
        .def_readwrite("final_max_depth_ratio", &neurodic::calibration::MultiviewCalibrationOptions::final_max_depth_ratio)
        .def_readwrite("initial_cameras", &neurodic::calibration::MultiviewCalibrationOptions::initial_cameras);

    py::class_<neurodic::calibration::MultiviewStageStat>(sub, "MultiviewStageStat")
        .def(py::init<>())
        .def_readwrite("stage", &neurodic::calibration::MultiviewStageStat::stage)
        .def_readwrite("num_registered_cameras", &neurodic::calibration::MultiviewStageStat::num_registered_cameras)
        .def_readwrite("num_points3d", &neurodic::calibration::MultiviewStageStat::num_points3d)
        .def_readwrite("num_observations", &neurodic::calibration::MultiviewStageStat::num_observations)
        .def_readwrite("mean_reprojection_error", &neurodic::calibration::MultiviewStageStat::mean_reprojection_error)
        .def_readwrite("focal_length", &neurodic::calibration::MultiviewStageStat::focal_length)
        .def_readwrite("principal_point_x", &neurodic::calibration::MultiviewStageStat::principal_point_x)
        .def_readwrite("principal_point_y", &neurodic::calibration::MultiviewStageStat::principal_point_y)
        .def_readwrite("distortion_k1", &neurodic::calibration::MultiviewStageStat::distortion_k1);

    py::class_<neurodic::calibration::MultiviewRegistrationAttempt>(sub, "MultiviewRegistrationAttempt")
        .def(py::init<>())
        .def_readwrite("image_index", &neurodic::calibration::MultiviewRegistrationAttempt::image_index)
        .def_readwrite("success", &neurodic::calibration::MultiviewRegistrationAttempt::success)
        .def_readwrite("reason", &neurodic::calibration::MultiviewRegistrationAttempt::reason)
        .def_readwrite("num_visible_points", &neurodic::calibration::MultiviewRegistrationAttempt::num_visible_points)
        .def_readwrite("num_pnp_correspondences", &neurodic::calibration::MultiviewRegistrationAttempt::num_pnp_correspondences)
        .def_readwrite("num_pnp_inliers", &neurodic::calibration::MultiviewRegistrationAttempt::num_pnp_inliers);

    py::class_<neurodic::calibration::MultiviewCalibrationResult>(sub, "MultiviewCalibrationResult")
        .def(py::init<>())
        .def_readwrite("cameras", &neurodic::calibration::MultiviewCalibrationResult::cameras)
        .def_readwrite("sparse_points", &neurodic::calibration::MultiviewCalibrationResult::sparse_points)
        .def_readwrite("inlier_match_counts", &neurodic::calibration::MultiviewCalibrationResult::inlier_match_counts)
        .def_readwrite("mean_reprojection_error", &neurodic::calibration::MultiviewCalibrationResult::mean_reprojection_error)
        .def_readwrite("stage_stats", &neurodic::calibration::MultiviewCalibrationResult::stage_stats)
        .def_readwrite("registration_attempts", &neurodic::calibration::MultiviewCalibrationResult::registration_attempts)
        .def_readwrite("pipeline_log", &neurodic::calibration::MultiviewCalibrationResult::pipeline_log)
        .def_readwrite("point_diagnostics", &neurodic::calibration::MultiviewCalibrationResult::point_diagnostics);

    py::class_<neurodic::calibration::MultiviewScaleObservation>(sub, "MultiviewScaleObservation")
        .def(py::init<>())
        .def_readwrite("camera_index", &neurodic::calibration::MultiviewScaleObservation::camera_index)
        .def_readwrite("image_points", &neurodic::calibration::MultiviewScaleObservation::image_points);

    py::class_<neurodic::calibration::MultiviewScaleOptions>(sub, "MultiviewScaleOptions")
        .def(py::init<>())
        .def_readwrite("board_rows", &neurodic::calibration::MultiviewScaleOptions::board_rows)
        .def_readwrite("board_cols", &neurodic::calibration::MultiviewScaleOptions::board_cols)
        .def_readwrite("square_size", &neurodic::calibration::MultiviewScaleOptions::square_size)
        .def_readwrite("max_reprojection_error", &neurodic::calibration::MultiviewScaleOptions::max_reprojection_error)
        .def_readwrite("trim_fraction", &neurodic::calibration::MultiviewScaleOptions::trim_fraction)
        .def_readwrite("min_common_corners", &neurodic::calibration::MultiviewScaleOptions::min_common_corners);

    py::class_<neurodic::calibration::MultiviewScaleResult>(sub, "MultiviewScaleResult")
        .def(py::init<>())
        .def_readwrite("sfm_to_world_scale", &neurodic::calibration::MultiviewScaleResult::sfm_to_world_scale)
        .def_readwrite("world_to_sfm_scale", &neurodic::calibration::MultiviewScaleResult::world_to_sfm_scale)
        .def_readwrite("sfm_square_size_mean", &neurodic::calibration::MultiviewScaleResult::sfm_square_size_mean)
        .def_readwrite("sfm_square_size_median", &neurodic::calibration::MultiviewScaleResult::sfm_square_size_median)
        .def_readwrite("sfm_square_size_std", &neurodic::calibration::MultiviewScaleResult::sfm_square_size_std)
        .def_readwrite("edge_cv", &neurodic::calibration::MultiviewScaleResult::edge_cv)
        .def_readwrite("triangulated_corners", &neurodic::calibration::MultiviewScaleResult::triangulated_corners)
        .def_readwrite("valid_edges", &neurodic::calibration::MultiviewScaleResult::valid_edges)
        .def_readwrite("triangulated_board_points_sfm", &neurodic::calibration::MultiviewScaleResult::triangulated_board_points_sfm)
        .def_readwrite("edge_lengths_sfm", &neurodic::calibration::MultiviewScaleResult::edge_lengths_sfm)
        .def_readwrite("scaled_cameras", &neurodic::calibration::MultiviewScaleResult::scaled_cameras)
        .def_readwrite("scaled_sparse_points", &neurodic::calibration::MultiviewScaleResult::scaled_sparse_points);

    sub.def("detect_calibration_board", &neurodic::calibration::detect_calibration_board, py::arg("image_path"), py::arg("board"), py::arg("options") = neurodic::calibration::BoardDetectionOptions{});
    sub.def("calibrate_mono_zhang", &neurodic::calibration::calibrate_mono_zhang, py::arg("image_paths"), py::arg("board"), py::arg("options") = neurodic::calibration::MonoCalibrationOptions{});
    sub.def("calibrate_mono_from_points", &neurodic::calibration::calibrate_mono_from_points, py::arg("object_points"), py::arg("image_points"), py::arg("image_width"), py::arg("image_height"), py::arg("options") = neurodic::calibration::MonoCalibrationOptions{});
    sub.def("calibrate_stereo_zhang", &neurodic::calibration::calibrate_stereo_zhang, py::arg("left_image_paths"), py::arg("right_image_paths"), py::arg("board"), py::arg("options") = neurodic::calibration::StereoCalibrationOptions{});
    sub.def("calibrate_stereo_from_points", &neurodic::calibration::calibrate_stereo_from_points, py::arg("object_points"), py::arg("left_image_points"), py::arg("right_image_points"), py::arg("image_width"), py::arg("image_height"), py::arg("options") = neurodic::calibration::StereoCalibrationOptions{});
    sub.def("calibrate_multiview_colmap_like", &neurodic::calibration::calibrate_multiview_colmap_like, py::arg("image_paths"), py::arg("options") = neurodic::calibration::MultiviewCalibrationOptions{});
    sub.def("to_core_calibration_result", &neurodic::calibration::to_core_calibration_result,
            py::arg("reconstruction"));
    sub.def("estimate_multiview_chessboard_scale", &neurodic::calibration::estimate_multiview_chessboard_scale, py::arg("cameras"), py::arg("sparse_points"), py::arg("observations"), py::arg("options"));
}
