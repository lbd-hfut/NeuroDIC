#include <neurodic/calibration/traditional/mono_calibration.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#endif

namespace neurodic::calibration {
namespace {

void validate_board(const CalibrationBoard& board)
{
    if (board.rows <= 0 || board.cols <= 0) {
        throw std::invalid_argument("Calibration board rows and cols must be positive.");
    }
    if (board.spacing <= 0.0) {
        throw std::invalid_argument("Calibration board spacing must be positive.");
    }
}

void validate_point_sets(const std::vector<std::vector<Eigen::Vector3d>>& object_points,
                         const std::vector<std::vector<Eigen::Vector2d>>& image_points,
                         const int image_width,
                         const int image_height)
{
    if (object_points.empty() || image_points.empty()) {
        throw std::invalid_argument("Calibration requires at least one view.");
    }
    if (object_points.size() != image_points.size()) {
        throw std::invalid_argument("Object point and image point view counts differ.");
    }
    if (image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("Image size must be positive.");
    }
    for (size_t i = 0; i < object_points.size(); ++i) {
        if (object_points[i].size() < 4 || object_points[i].size() != image_points[i].size()) {
            throw std::invalid_argument("Each calibration view needs matching object/image points.");
        }
    }
}

#ifdef NEURODIC_HAS_OPENCV
cv::TermCriteria make_criteria(const int max_iterations, const double epsilon)
{
    return {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, max_iterations, epsilon};
}

cv::Size make_pattern_size(const CalibrationBoard& board) { return {board.cols, board.rows}; }

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

std::vector<Eigen::Vector2d> to_eigen_image_points(const std::vector<cv::Point2f>& points)
{
    std::vector<Eigen::Vector2d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.emplace_back(p.x, p.y);
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

double compute_view_error(const std::vector<cv::Point3f>& object_points,
                          const std::vector<cv::Point2f>& image_points,
                          const cv::Mat& rvec,
                          const cv::Mat& tvec,
                          const cv::Mat& K,
                          const cv::Mat& distortion)
{
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, K, distortion, projected);
    double squared_sum = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const cv::Point2f d = projected[i] - image_points[i];
        squared_sum += d.x * d.x + d.y * d.y;
    }
    return std::sqrt(squared_sum / static_cast<double>(projected.size()));
}
#endif

} // namespace

std::vector<Eigen::Vector3d> CalibrationBoard::object_points() const
{
    validate_board(*this);
    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<size_t>(point_count()));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double x = static_cast<double>(c) * spacing;
            const double y = static_cast<double>(r) * spacing;
            if (type == CalibrationBoardType::AsymmetricCircles) {
                x = static_cast<double>(2 * c + (r % 2)) * spacing;
            }
            points.emplace_back(x, y, 0.0);
        }
    }
    return points;
}

CalibrationDetection detect_calibration_board(const std::string& image_path,
                                              const CalibrationBoard& board,
                                              const BoardDetectionOptions& options)
{
    validate_board(board);
#ifndef NEURODIC_HAS_OPENCV
    (void)image_path;
    (void)options;
    throw std::runtime_error("OpenCV is required for calibration board detection.");
#else
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Failed to read calibration image: " + image_path);
    }

    cv::Mat work = image;
    if (options.normalize_image) {
        cv::equalizeHist(image, work);
    }

    std::vector<cv::Point2f> corners;
    bool found = false;
    const cv::Size pattern_size = make_pattern_size(board);
    if (board.type == CalibrationBoardType::Chessboard) {
        const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        found = cv::findChessboardCorners(work, pattern_size, corners, flags);
        if (found && options.refine_corners) {
            cv::cornerSubPix(work,
                             corners,
                             {11, 11},
                             {-1, -1},
                             make_criteria(options.max_iterations, options.epsilon));
        }
    } else {
        const int flags = board.type == CalibrationBoardType::AsymmetricCircles
                              ? cv::CALIB_CB_ASYMMETRIC_GRID
                              : cv::CALIB_CB_SYMMETRIC_GRID;
        found = cv::findCirclesGrid(work, pattern_size, corners, flags);
    }

    CalibrationDetection detection;
    detection.found = found;
    detection.image_path = image_path;
    detection.image_width = image.cols;
    detection.image_height = image.rows;
    if (found) {
        detection.image_points = to_eigen_image_points(corners);
    }
    return detection;
#endif
}

MonoCalibrationResult calibrate_mono_from_points(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& image_points,
    const int image_width,
    const int image_height,
    const MonoCalibrationOptions& options)
{
    validate_point_sets(object_points, image_points, image_width, image_height);
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw std::runtime_error("OpenCV is required for Zhang mono calibration.");
#else
    std::vector<std::vector<cv::Point3f>> object_cv;
    std::vector<std::vector<cv::Point2f>> image_cv;
    object_cv.reserve(object_points.size());
    image_cv.reserve(image_points.size());
    for (size_t i = 0; i < object_points.size(); ++i) {
        object_cv.push_back(to_cv_object_points(object_points[i]));
        image_cv.push_back(to_cv_image_points(image_points[i]));
    }

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distortion = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    int flags = 0;
    if (!options.estimate_tangential_distortion) {
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
    }
    if (!options.estimate_k3) {
        flags |= cv::CALIB_FIX_K3;
    }

    const double rms = cv::calibrateCamera(object_cv,
                                           image_cv,
                                           {image_width, image_height},
                                           K,
                                           distortion,
                                           rvecs,
                                           tvecs,
                                           flags,
                                           make_criteria(options.max_iterations, options.epsilon));

    MonoCalibrationResult result;
    result.rms_error = rms;
    result.camera.K = to_eigen_matrix3(K);
    result.camera.distortion = to_distortion_vector(distortion);
    result.camera.image_width = image_width;
    result.camera.image_height = image_height;
    result.camera.rms_error = rms;
    result.per_view_errors.reserve(object_cv.size());
    result.board_rotations.reserve(rvecs.size());
    result.board_translations.reserve(tvecs.size());
    for (size_t i = 0; i < object_cv.size(); ++i) {
        cv::Mat R_cv;
        cv::Rodrigues(rvecs[i], R_cv);
        result.board_rotations.push_back(to_eigen_matrix3(R_cv));
        result.board_translations.push_back(to_eigen_vector3(tvecs[i]));
        result.per_view_errors.push_back(compute_view_error(object_cv[i], image_cv[i], rvecs[i], tvecs[i], K, distortion));
    }
    return result;
#endif
}

MonoCalibrationResult calibrate_mono_zhang(const std::vector<std::string>& image_paths,
                                           const CalibrationBoard& board,
                                           const MonoCalibrationOptions& options)
{
    if (image_paths.empty()) {
        throw std::invalid_argument("Mono calibration requires at least one image.");
    }
    const auto object_template = board.object_points();
    std::vector<std::vector<Eigen::Vector3d>> object_points;
    std::vector<std::vector<Eigen::Vector2d>> image_points;
    std::vector<CalibrationDetection> detections;
    int width = 0;
    int height = 0;

    for (const auto& path : image_paths) {
        CalibrationDetection detection = detect_calibration_board(path, board, options.detection);
        if (width == 0 && height == 0) {
            width = detection.image_width;
            height = detection.image_height;
        }
        if (detection.found) {
            object_points.push_back(object_template);
            image_points.push_back(detection.image_points);
        }
        detections.push_back(std::move(detection));
    }

    if (image_points.empty()) {
        throw std::runtime_error("No calibration board was detected in the mono image set.");
    }

    MonoCalibrationResult result = calibrate_mono_from_points(object_points, image_points, width, height, options);
    result.detections = std::move(detections);
    return result;
}

CameraModel calibrate_mono(int image_count)
{
    (void)image_count;
    throw std::runtime_error("Use calibrate_mono_zhang with image paths or calibrate_mono_from_points.");
}

}  // namespace neurodic::calibration
