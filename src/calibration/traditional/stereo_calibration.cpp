#include <neurodic/calibration/traditional/stereo_calibration.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace neurodic::calibration {
namespace {

void validate_stereo_sets(const std::vector<std::vector<Eigen::Vector3d>>& object_points,
                          const std::vector<std::vector<Eigen::Vector2d>>& left_points,
                          const std::vector<std::vector<Eigen::Vector2d>>& right_points,
                          const int image_width,
                          const int image_height)
{
    if (object_points.empty()) {
        throw std::invalid_argument("Stereo calibration requires at least one valid image pair.");
    }
    if (object_points.size() != left_points.size() || object_points.size() != right_points.size()) {
        throw std::invalid_argument("Stereo calibration point view counts differ.");
    }
    if (image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("Image size must be positive.");
    }
    for (size_t i = 0; i < object_points.size(); ++i) {
        if (object_points[i].size() < 4 || object_points[i].size() != left_points[i].size() ||
            object_points[i].size() != right_points[i].size()) {
            throw std::invalid_argument("Each stereo view needs matching object/left/right points.");
        }
    }
}

#ifdef NEURODIC_HAS_OPENCV
cv::TermCriteria make_criteria(const int max_iterations, const double epsilon)
{
    return {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, max_iterations, epsilon};
}

std::vector<cv::Point3f> to_cv_object_points(const std::vector<Eigen::Vector3d>& points)
{
    std::vector<cv::Point3f> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
    }
    return out;
}

std::vector<cv::Point2f> to_cv_image_points(const std::vector<Eigen::Vector2d>& points)
{
    std::vector<cv::Point2f> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }
    return out;
}

cv::Mat to_cv_matrix3(const Eigen::Matrix3d& mat)
{
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out.at<double>(r, c) = mat(r, c);
        }
    }
    return out;
}

cv::Mat to_cv_distortion(const std::vector<double>& distortion)
{
    cv::Mat out = cv::Mat::zeros(std::max<size_t>(distortion.size(), 5), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(distortion.size()); ++i) {
        out.at<double>(i) = distortion[static_cast<size_t>(i)];
    }
    return out;
}

Eigen::Matrix3d to_eigen_matrix3(const cv::Mat& mat)
{
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

Eigen::Vector3d to_eigen_vector3(const cv::Mat& mat)
{
    return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

std::vector<double> to_distortion_vector(const cv::Mat& mat)
{
    std::vector<double> out;
    out.reserve(static_cast<size_t>(mat.total()));
    for (int i = 0; i < static_cast<int>(mat.total()); ++i) {
        out.push_back(mat.at<double>(i));
    }
    return out;
}

double combined_pair_error(const double left_error, const double right_error)
{
    return std::sqrt(0.5 * (left_error * left_error + right_error * right_error));
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    double result = *mid;
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), mid);
        result = 0.5 * (result + *lower);
    }
    return result;
}

double robust_sigma(const std::vector<double>& values)
{
    if (values.size() < 2) {
        return 0.0;
    }
    const double med = median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) {
        deviations.push_back(std::abs(value - med));
    }
    return 1.4826 * median(std::move(deviations));
}

template <typename T>
std::vector<T> subset_by_indices(const std::vector<T>& values, const std::vector<int>& indices)
{
    std::vector<T> out;
    out.reserve(indices.size());
    for (int index : indices) {
        out.push_back(values[static_cast<std::size_t>(index)]);
    }
    return out;
}

std::vector<int> sequential_indices(const std::size_t n)
{
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}
#endif

} // namespace

namespace {

StereoCalibrationResult calibrate_stereo_from_points_once(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& left_image_points,
    const std::vector<std::vector<Eigen::Vector2d>>& right_image_points,
    const int image_width,
    const int image_height,
    const StereoCalibrationOptions& options)
{
    validate_stereo_sets(object_points, left_image_points, right_image_points, image_width, image_height);
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw std::runtime_error("OpenCV is required for Zhang stereo calibration.");
#else
    std::vector<std::vector<cv::Point3f>> object_cv;
    std::vector<std::vector<cv::Point2f>> left_cv;
    std::vector<std::vector<cv::Point2f>> right_cv;
    object_cv.reserve(object_points.size());
    left_cv.reserve(left_image_points.size());
    right_cv.reserve(right_image_points.size());
    for (size_t i = 0; i < object_points.size(); ++i) {
        object_cv.push_back(to_cv_object_points(object_points[i]));
        left_cv.push_back(to_cv_image_points(left_image_points[i]));
        right_cv.push_back(to_cv_image_points(right_image_points[i]));
    }

    MonoCalibrationOptions mono_options;
    mono_options.estimate_tangential_distortion = options.estimate_tangential_distortion;
    mono_options.estimate_k3 = options.estimate_k3;
    mono_options.max_iterations = options.max_iterations;
    mono_options.epsilon = options.epsilon;
    MonoCalibrationResult left_mono =
        calibrate_mono_from_points(object_points, left_image_points, image_width, image_height, mono_options);
    MonoCalibrationResult right_mono =
        calibrate_mono_from_points(object_points, right_image_points, image_width, image_height, mono_options);

    cv::Mat K1 = to_cv_matrix3(left_mono.camera.K);
    cv::Mat K2 = to_cv_matrix3(right_mono.camera.K);
    cv::Mat d1 = to_cv_distortion(left_mono.camera.distortion);
    cv::Mat d2 = to_cv_distortion(right_mono.camera.distortion);
    cv::Mat R;
    cv::Mat T;
    cv::Mat E;
    cv::Mat F;
    std::vector<cv::Mat> stereo_rvecs;
    std::vector<cv::Mat> stereo_tvecs;
    cv::Mat per_view_errors;

    int flags = cv::CALIB_USE_INTRINSIC_GUESS;
    if (options.fix_intrinsics) {
        flags |= cv::CALIB_FIX_INTRINSIC;
    }
    if (!options.estimate_tangential_distortion) {
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
    }
    if (!options.estimate_k3) {
        flags |= cv::CALIB_FIX_K3;
    }

    const double rms = cv::stereoCalibrate(object_cv,
                                           left_cv,
                                           right_cv,
                                           K1,
                                           d1,
                                           K2,
                                           d2,
                                           {image_width, image_height},
                                           R,
                                           T,
                                           E,
                                           F,
                                           stereo_rvecs,
                                           stereo_tvecs,
                                           per_view_errors,
                                           flags,
                                           make_criteria(options.max_iterations, options.epsilon));

    StereoCalibrationResult result;
    result.rms_error = rms;
    result.left = left_mono.camera;
    result.right = right_mono.camera;
    result.left.K = to_eigen_matrix3(K1);
    result.right.K = to_eigen_matrix3(K2);
    result.left.distortion = to_distortion_vector(d1);
    result.right.distortion = to_distortion_vector(d2);
    result.left.rms_error = rms;
    result.right.rms_error = rms;
    result.R_lr = to_eigen_matrix3(R);
    result.t_lr = to_eigen_vector3(T);
    result.right.R = result.R_lr;
    result.right.t = result.t_lr;
    result.essential = to_eigen_matrix3(E);
    result.fundamental = to_eigen_matrix3(F);
    result.per_pair_left_errors.reserve(object_points.size());
    result.per_pair_right_errors.reserve(object_points.size());
    for (int i = 0; i < per_view_errors.rows; ++i) {
        result.per_pair_left_errors.push_back(per_view_errors.at<double>(i, 0));
        result.per_pair_right_errors.push_back(per_view_errors.at<double>(i, 1));
    }
    result.per_pair_errors.reserve(result.per_pair_left_errors.size());
    for (size_t i = 0; i < result.per_pair_left_errors.size(); ++i) {
        result.per_pair_errors.push_back(
            combined_pair_error(result.per_pair_left_errors[i], result.per_pair_right_errors[i]));
    }
    return result;
#endif
}

} // namespace

StereoCalibrationResult calibrate_stereo_from_points(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& left_image_points,
    const std::vector<std::vector<Eigen::Vector2d>>& right_image_points,
    const int image_width,
    const int image_height,
    const StereoCalibrationOptions& options)
{
    validate_stereo_sets(object_points, left_image_points, right_image_points, image_width, image_height);
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw std::runtime_error("OpenCV is required for Zhang stereo calibration.");
#else
    StereoCalibrationResult initial =
        calibrate_stereo_from_points_once(object_points, left_image_points, right_image_points, image_width, image_height, options);
    initial.initial_rms_error = initial.rms_error;
    initial.initial_per_pair_errors = initial.per_pair_errors;
    initial.initial_per_pair_left_errors = initial.per_pair_left_errors;
    initial.initial_per_pair_right_errors = initial.per_pair_right_errors;
    initial.kept_pair_indices = sequential_indices(object_points.size());

    if (!options.reject_outlier_pairs || object_points.size() <= static_cast<std::size_t>(options.min_pairs_after_rejection)) {
        return initial;
    }

    const double combined_median = median(initial.per_pair_errors);
    const double combined_sigma = robust_sigma(initial.per_pair_errors);
    const double left_median = median(initial.per_pair_left_errors);
    const double left_sigma = robust_sigma(initial.per_pair_left_errors);
    const double right_median = median(initial.per_pair_right_errors);
    const double right_sigma = robust_sigma(initial.per_pair_right_errors);
    const double combined_limit = combined_median + options.outlier_mad_factor * std::max(combined_sigma, 1.0e-12);
    const double left_limit = left_median + options.outlier_mad_factor * std::max(left_sigma, 1.0e-12);
    const double right_limit = right_median + options.outlier_mad_factor * std::max(right_sigma, 1.0e-12);

    std::vector<int> kept;
    std::vector<int> rejected;
    std::vector<std::string> reasons;
    kept.reserve(object_points.size());
    for (size_t i = 0; i < object_points.size(); ++i) {
        const double left_error = initial.per_pair_left_errors[i];
        const double right_error = initial.per_pair_right_errors[i];
        const double pair_error = initial.per_pair_errors[i];
        const double low = std::max(std::min(left_error, right_error), 1.0e-12);
        const double ratio = std::max(left_error, right_error) / low;
        const double abs_diff = std::abs(left_error - right_error);

        std::vector<std::string> reason_parts;
        if (pair_error > combined_limit || left_error > left_limit || right_error > right_limit) {
            reason_parts.push_back("reprojection_outlier");
        }
        if (abs_diff > options.left_right_error_abs_threshold &&
            ratio > options.left_right_error_ratio_threshold) {
            reason_parts.push_back("left_right_error_mismatch");
        }

        if (reason_parts.empty()) {
            kept.push_back(static_cast<int>(i));
        } else {
            rejected.push_back(static_cast<int>(i));
            std::string reason = reason_parts.front();
            for (size_t r = 1; r < reason_parts.size(); ++r) {
                reason += "+" + reason_parts[r];
            }
            reasons.push_back(reason);
        }
    }

    if (rejected.empty() || kept.size() < static_cast<std::size_t>(options.min_pairs_after_rejection)) {
        initial.kept_pair_indices = sequential_indices(object_points.size());
        initial.rejected_pair_indices = std::move(rejected);
        initial.rejection_reasons = std::move(reasons);
        initial.outlier_rejection_applied = false;
        return initial;
    }

    StereoCalibrationResult filtered =
        calibrate_stereo_from_points_once(subset_by_indices(object_points, kept),
                                          subset_by_indices(left_image_points, kept),
                                          subset_by_indices(right_image_points, kept),
                                          image_width,
                                          image_height,
                                          options);
    filtered.initial_rms_error = initial.rms_error;
    filtered.initial_per_pair_errors = initial.per_pair_errors;
    filtered.initial_per_pair_left_errors = initial.per_pair_left_errors;
    filtered.initial_per_pair_right_errors = initial.per_pair_right_errors;
    filtered.kept_pair_indices = std::move(kept);
    filtered.rejected_pair_indices = std::move(rejected);
    filtered.rejection_reasons = std::move(reasons);
    filtered.outlier_rejection_applied = true;
    return filtered;
#endif
}

StereoCalibrationResult calibrate_stereo_zhang(const std::vector<std::string>& left_image_paths,
                                               const std::vector<std::string>& right_image_paths,
                                               const CalibrationBoard& board,
                                               const StereoCalibrationOptions& options)
{
    if (left_image_paths.size() != right_image_paths.size() || left_image_paths.empty()) {
        throw std::invalid_argument("Stereo calibration requires equal non-empty left/right image path lists.");
    }

    const auto object_template = board.object_points();
    std::vector<std::vector<Eigen::Vector3d>> object_points;
    std::vector<std::vector<Eigen::Vector2d>> left_points;
    std::vector<std::vector<Eigen::Vector2d>> right_points;
    std::vector<CalibrationDetection> left_detections;
    std::vector<CalibrationDetection> right_detections;
    int width = 0;
    int height = 0;

    for (size_t i = 0; i < left_image_paths.size(); ++i) {
        CalibrationDetection left = detect_calibration_board(left_image_paths[i], board, options.detection);
        CalibrationDetection right = detect_calibration_board(right_image_paths[i], board, options.detection);
        if (width == 0 && height == 0) {
            width = left.image_width;
            height = left.image_height;
        }
        if (left.found && right.found) {
            object_points.push_back(object_template);
            left_points.push_back(left.image_points);
            right_points.push_back(right.image_points);
        }
        left_detections.push_back(std::move(left));
        right_detections.push_back(std::move(right));
    }

    if (object_points.empty()) {
        throw std::runtime_error("No stereo image pair contained a board detected in both views.");
    }

    StereoCalibrationResult result =
        calibrate_stereo_from_points(object_points, left_points, right_points, width, height, options);
    result.left_detections = std::move(left_detections);
    result.right_detections = std::move(right_detections);
    return result;
}

StereoCalibrationResult calibrate_stereo(int image_pair_count)
{
    (void)image_pair_count;
    throw std::runtime_error("Use calibrate_stereo_zhang with image paths or calibrate_stereo_from_points.");
}

}  // namespace neurodic::calibration
