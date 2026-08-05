#include <neurodic/calibration/traditional/multiview_calibration.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#ifdef TRADITIONAL_DIC_HAS_CERES
#include <ceres/ceres.h>
#include <ceres/product_manifold.h>
#include <ceres/rotation.h>
#endif

namespace neurodic::calibration {
namespace {

void validate_scale_options(const MultiviewScaleOptions& options)
{
    if (options.board_rows <= 0 || options.board_cols <= 0) {
        throw std::invalid_argument("Scale board rows and cols must be positive.");
    }
    if (options.square_size <= 0.0) {
        throw std::invalid_argument("Scale square size must be positive.");
    }
    if (options.trim_fraction < 0.0 || options.trim_fraction >= 0.5) {
        throw std::invalid_argument("Scale trim_fraction must be in [0, 0.5).");
    }
}

Eigen::Vector3d triangulate_linear(const std::vector<CameraModel>& cameras,
                                   const std::vector<FeatureTrackObservation>& observations)
{
    if (observations.size() < 2) {
        throw std::invalid_argument("Triangulation requires at least two observations.");
    }
    Eigen::MatrixXd A(static_cast<int>(observations.size() * 2), 4);
    int row = 0;
    for (const auto& observation : observations) {
        if (observation.image_index < 0 || observation.image_index >= static_cast<int>(cameras.size())) {
            throw std::out_of_range("Scale observation camera index is out of range.");
        }
        const auto& camera = cameras[static_cast<size_t>(observation.image_index)];
        const double f = 0.5 * (camera.K(0, 0) + camera.K(1, 1));
        double x = (observation.point.x() - camera.K(0, 2)) / f;
        double y = (observation.point.y() - camera.K(1, 2)) / f;
        const double k1 = camera.distortion.empty() ? 0.0 : camera.distortion[0];
        for (int iter = 0; iter < 20; ++iter) {
            const double r2 = x * x + y * y;
            const double radial = 1.0 + k1 * r2;
            if (std::abs(radial) < std::numeric_limits<double>::epsilon()) {
                break;
            }
            x = (observation.point.x() - camera.K(0, 2)) / (f * radial);
            y = (observation.point.y() - camera.K(1, 2)) / (f * radial);
        }
        Eigen::Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0) = camera.R;
        P.col(3) = camera.t;
        A.row(row++) = x * P.row(2) - P.row(0);
        A.row(row++) = y * P.row(2) - P.row(1);
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
    if (std::abs(homogeneous.w()) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Triangulated homogeneous point has near-zero scale.");
    }
    return homogeneous.head<3>() / homogeneous.w();
}

double reprojection_error(const Eigen::Vector3d& point,
                          const CameraModel& camera,
                          const Eigen::Vector2d& observation)
{
    const Eigen::Vector3d cam = camera.R * point + camera.t;
    if (cam.z() <= std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }
    const double x = cam.x() / cam.z();
    const double y = cam.y() / cam.z();
    const double r2 = x * x + y * y;
    const double k1 = camera.distortion.empty() ? 0.0 : camera.distortion[0];
    const double radial = 1.0 + k1 * r2;
    const double f = 0.5 * (camera.K(0, 0) + camera.K(1, 1));
    const Eigen::Vector2d uv(f * x * radial + camera.K(0, 2),
                             f * y * radial + camera.K(1, 2));
    return (uv - observation).norm();
}

double mean_reprojection_error(const Eigen::Vector3d& point,
                               const std::vector<CameraModel>& cameras,
                               const std::vector<FeatureTrackObservation>& observations)
{
    double sum = 0.0;
    for (const auto& observation : observations) {
        sum += reprojection_error(point, cameras[static_cast<size_t>(observation.image_index)], observation.point);
    }
    return sum / static_cast<double>(observations.size());
}

double mean_value(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

double std_value(const std::vector<double>& values, const double mean)
{
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

std::vector<double> trim_values(std::vector<double> values, const double trim_fraction)
{
    if (values.empty() || trim_fraction <= 0.0) {
        return values;
    }
    std::sort(values.begin(), values.end());
    const size_t trim = static_cast<size_t>(std::floor(trim_fraction * static_cast<double>(values.size())));
    if (2 * trim >= values.size()) {
        return values;
    }
    return {values.begin() + static_cast<std::ptrdiff_t>(trim), values.end() - static_cast<std::ptrdiff_t>(trim)};
}

#ifdef NEURODIC_HAS_OPENCV
struct ImageFeatures {
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct PairGeometry {
    int i = -1;
    int j = -1;
    std::vector<cv::DMatch> matches;
    std::vector<unsigned char> inlier_mask;
    cv::Mat R;
    cv::Mat t;
    int inliers = 0;
};

struct SfMObservationId {
    int image = -1;
    int point2d = -1;

    bool operator==(const SfMObservationId& other) const
    {
        return image == other.image && point2d == other.point2d;
    }
};

struct SfMObservationIdHash {
    size_t operator()(const SfMObservationId& id) const
    {
        const uint64_t hi = static_cast<uint32_t>(id.image);
        const uint64_t lo = static_cast<uint32_t>(id.point2d);
        return std::hash<uint64_t>{}((hi << 32) | lo);
    }
};

class SfMCorrespondenceGraph {
public:
    void add_image(const int image_id, const size_t num_points2d)
    {
        if (image_id < 0) {
            throw std::invalid_argument("SfM correspondence graph image id must be non-negative.");
        }
        if (static_cast<size_t>(image_id) >= images_.size()) {
            images_.resize(static_cast<size_t>(image_id) + 1);
        }
        images_[static_cast<size_t>(image_id)].corrs.assign(num_points2d, {});
    }

    void add_two_view_geometry(const PairGeometry& geometry)
    {
        if (geometry.i < 0 || geometry.j < 0 || geometry.i == geometry.j ||
            static_cast<size_t>(geometry.i) >= images_.size() ||
            static_cast<size_t>(geometry.j) >= images_.size()) {
            return;
        }

        auto& image_i = images_[static_cast<size_t>(geometry.i)];
        auto& image_j = images_[static_cast<size_t>(geometry.j)];
        int num_unique_matches = 0;

        for (size_t k = 0; k < geometry.matches.size(); ++k) {
            if (k >= geometry.inlier_mask.size() || geometry.inlier_mask[k] == 0) {
                continue;
            }
            const auto& match = geometry.matches[k];
            if (match.queryIdx < 0 || match.trainIdx < 0 ||
                static_cast<size_t>(match.queryIdx) >= image_i.corrs.size() ||
                static_cast<size_t>(match.trainIdx) >= image_j.corrs.size()) {
                continue;
            }

            auto& corrs_i = image_i.corrs[static_cast<size_t>(match.queryIdx)];
            const bool duplicate = std::any_of(corrs_i.begin(), corrs_i.end(), [&](const SfMObservationId& corr) {
                return corr.image == geometry.j && corr.point2d == match.trainIdx;
            });
            if (duplicate) {
                continue;
            }

            corrs_i.push_back({geometry.j, match.trainIdx});
            image_j.corrs[static_cast<size_t>(match.trainIdx)].push_back({geometry.i, match.queryIdx});
            ++num_unique_matches;
        }

        if (num_unique_matches > 0) {
            image_pair_match_counts_[pair_key(geometry.i, geometry.j)] = num_unique_matches;
        }
    }

    const std::vector<SfMObservationId>& find_correspondences(const int image_id, const int point2d_idx) const
    {
        return images_.at(static_cast<size_t>(image_id)).corrs.at(static_cast<size_t>(point2d_idx));
    }

    bool has_correspondences(const int image_id, const int point2d_idx) const
    {
        return !find_correspondences(image_id, point2d_idx).empty();
    }

    bool is_two_view_observation(const int image_id, const int point2d_idx) const
    {
        const auto& corrs = find_correspondences(image_id, point2d_idx);
        if (corrs.size() != 1) {
            return false;
        }
        const auto& reverse_corrs = find_correspondences(corrs[0].image, corrs[0].point2d);
        return reverse_corrs.size() == 1 && reverse_corrs[0].image == image_id && reverse_corrs[0].point2d == point2d_idx;
    }

    void extract_transitive_correspondences(const int image_id,
                                            const int point2d_idx,
                                            const int max_transitivity,
                                            std::vector<SfMObservationId>& corrs) const
    {
        corrs.clear();
        if (max_transitivity <= 0 || !has_correspondences(image_id, point2d_idx)) {
            return;
        }

        std::queue<std::pair<SfMObservationId, int>> queue;
        std::unordered_set<SfMObservationId, SfMObservationIdHash> visited;
        const SfMObservationId root{image_id, point2d_idx};
        queue.push({root, 0});
        visited.insert(root);

        while (!queue.empty()) {
            const auto [current, depth] = queue.front();
            queue.pop();
            corrs.push_back(current);
            if (depth >= max_transitivity) {
                continue;
            }
            for (const auto& next : find_correspondences(current.image, current.point2d)) {
                if (visited.insert(next).second) {
                    queue.push({next, depth + 1});
                }
            }
        }
    }

    size_t num_images() const
    {
        return images_.size();
    }

    size_t num_points2d(const int image_id) const
    {
        return images_.at(static_cast<size_t>(image_id)).corrs.size();
    }

private:
    struct Image {
        std::vector<std::vector<SfMObservationId>> corrs;
    };

    static int64_t pair_key(const int image1, const int image2)
    {
        const int a = std::min(image1, image2);
        const int b = std::max(image1, image2);
        return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
    }

    std::vector<Image> images_;
    std::unordered_map<int64_t, int> image_pair_match_counts_;
};

struct SfMPoint2DState {
    Eigen::Vector2d xy = Eigen::Vector2d::Zero();
    int point3d = -1;
};

struct SfMImageState {
    CameraModel camera;
    bool registered = false;
    std::vector<SfMPoint2DState> points2d;
};

struct SfMPoint3DState {
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    std::vector<SfMObservationId> track;
    double reprojection_error = 0.0;
    bool valid = true;
};

class SfMReconstructionState {
public:
    explicit SfMReconstructionState(const size_t num_images = 0) : images(num_images) {}

    int add_point3d(const Eigen::Vector3d& xyz, const std::vector<SfMObservationId>& track)
    {
        const int point3d_id = static_cast<int>(points3d.size());
        points3d.push_back({xyz, {}, 0.0, true});
        for (const auto& obs : track) {
            add_observation(point3d_id, obs);
        }
        return point3d_id;
    }

    bool add_observation(const int point3d_id, const SfMObservationId& obs)
    {
        if (point3d_id < 0 || static_cast<size_t>(point3d_id) >= points3d.size() ||
            !points3d[static_cast<size_t>(point3d_id)].valid || !valid_observation(obs)) {
            return false;
        }
        auto& point2d = images[static_cast<size_t>(obs.image)].points2d[static_cast<size_t>(obs.point2d)];
        if (point2d.point3d >= 0) {
            return point2d.point3d == point3d_id;
        }
        auto& track = points3d[static_cast<size_t>(point3d_id)].track;
        const bool duplicate_image = std::any_of(track.begin(), track.end(), [&](const SfMObservationId& existing) {
            return existing.image == obs.image;
        });
        if (duplicate_image) {
            return false;
        }
        point2d.point3d = point3d_id;
        track.push_back(obs);
        return true;
    }

    void delete_point3d(const int point3d_id)
    {
        if (point3d_id < 0 || static_cast<size_t>(point3d_id) >= points3d.size() ||
            !points3d[static_cast<size_t>(point3d_id)].valid) {
            return;
        }
        for (const auto& obs : points3d[static_cast<size_t>(point3d_id)].track) {
            if (valid_observation(obs)) {
                auto& point2d = images[static_cast<size_t>(obs.image)].points2d[static_cast<size_t>(obs.point2d)];
                if (point2d.point3d == point3d_id) {
                    point2d.point3d = -1;
                }
            }
        }
        points3d[static_cast<size_t>(point3d_id)].track.clear();
        points3d[static_cast<size_t>(point3d_id)].valid = false;
    }

    bool delete_observation(const SfMObservationId& obs)
    {
        if (!valid_observation(obs)) {
            return false;
        }
        auto& point2d = images[static_cast<size_t>(obs.image)].points2d[static_cast<size_t>(obs.point2d)];
        const int point3d_id = point2d.point3d;
        if (point3d_id < 0 || static_cast<size_t>(point3d_id) >= points3d.size() ||
            !points3d[static_cast<size_t>(point3d_id)].valid) {
            return false;
        }
        auto& track = points3d[static_cast<size_t>(point3d_id)].track;
        if (track.size() <= 2) {
            delete_point3d(point3d_id);
            return true;
        }
        point2d.point3d = -1;
        track.erase(std::remove_if(track.begin(), track.end(), [&](const SfMObservationId& candidate) {
                        return candidate.image == obs.image && candidate.point2d == obs.point2d;
                    }),
                    track.end());
        return true;
    }

    bool merge_point3d(const int target_id, const int source_id, const Eigen::Vector3d& merged_xyz)
    {
        if (target_id < 0 || source_id < 0 || target_id == source_id ||
            static_cast<size_t>(target_id) >= points3d.size() ||
            static_cast<size_t>(source_id) >= points3d.size() ||
            !points3d[static_cast<size_t>(target_id)].valid ||
            !points3d[static_cast<size_t>(source_id)].valid) {
            return false;
        }
        std::vector<SfMObservationId> source_track = points3d[static_cast<size_t>(source_id)].track;
        points3d[static_cast<size_t>(target_id)].xyz = merged_xyz;
        for (const auto& obs : source_track) {
            if (!valid_observation(obs)) {
                continue;
            }
            images[static_cast<size_t>(obs.image)].points2d[static_cast<size_t>(obs.point2d)].point3d = -1;
            add_observation(target_id, obs);
        }
        points3d[static_cast<size_t>(source_id)].track.clear();
        points3d[static_cast<size_t>(source_id)].valid = false;
        return true;
    }

    bool valid_observation(const SfMObservationId& obs) const
    {
        return obs.image >= 0 && obs.point2d >= 0 &&
               static_cast<size_t>(obs.image) < images.size() &&
               static_cast<size_t>(obs.point2d) < images[static_cast<size_t>(obs.image)].points2d.size();
    }

    std::vector<SfMImageState> images;
    std::vector<SfMPoint3DState> points3d;
};

class SfMObservationManager {
public:
    SfMObservationManager(SfMReconstructionState& reconstruction, const SfMCorrespondenceGraph& graph)
        : reconstruction_(reconstruction), graph_(graph)
    {
    }

    int add_point3d(const Eigen::Vector3d& xyz, const std::vector<SfMObservationId>& track)
    {
        return reconstruction_.add_point3d(xyz, track);
    }

    bool add_observation(const int point3d_id, const SfMObservationId& obs)
    {
        return reconstruction_.add_observation(point3d_id, obs);
    }

    bool delete_observation(const SfMObservationId& obs)
    {
        return reconstruction_.delete_observation(obs);
    }

    bool merge_point3d(const int target_id, const int source_id, const Eigen::Vector3d& merged_xyz)
    {
        return reconstruction_.merge_point3d(target_id, source_id, merged_xyz);
    }

    size_t num_visible_points3d(const int image_id) const
    {
        size_t count = 0;
        std::unordered_set<int> seen;
        for (int point_idx = 0; point_idx < static_cast<int>(graph_.num_points2d(image_id)); ++point_idx) {
            seen.clear();
            for (const auto& corr : graph_.find_correspondences(image_id, point_idx)) {
                if (!reconstruction_.valid_observation(corr) ||
                    !reconstruction_.images[static_cast<size_t>(corr.image)].registered) {
                    continue;
                }
                const int point3d = reconstruction_.images[static_cast<size_t>(corr.image)]
                                        .points2d[static_cast<size_t>(corr.point2d)]
                                        .point3d;
                if (point3d >= 0 && seen.insert(point3d).second) {
                    ++count;
                    break;
                }
            }
        }
        return count;
    }

private:
    SfMReconstructionState& reconstruction_;
    const SfMCorrespondenceGraph& graph_;
};

Eigen::Matrix3d cv_to_eigen3x3(const cv::Mat& mat)
{
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

Eigen::Vector3d cv_to_eigen3(const cv::Mat& mat)
{
    return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

CameraModel make_initial_camera(const std::string& label, const int width, const int height)
{
    CameraModel camera;
    const double focal = 1.2 * static_cast<double>(std::max(width, height));
    camera.K << focal, 0.0, static_cast<double>(width - 1) * 0.5, 0.0, focal,
        static_cast<double>(height - 1) * 0.5, 0.0, 0.0, 1.0;
    camera.distortion = {0.0, 0.0, 0.0, 0.0};
    camera.image_width = width;
    camera.image_height = height;
    camera.label = label;
    return camera;
}

cv::Point2f normalize_point(const cv::Point2f& point, const CameraModel& camera)
{
    return {static_cast<float>((point.x - camera.K(0, 2)) / camera.K(0, 0)),
            static_cast<float>((point.y - camera.K(1, 2)) / camera.K(1, 1))};
}

cv::Point2f project_point_cv(const Eigen::Vector3d& point, const CameraModel& camera)
{
    const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    const Eigen::Vector3d projected = camera.projection_matrix() * homogeneous;
    return {static_cast<float>(projected.x() / projected.z()), static_cast<float>(projected.y() / projected.z())};
}

std::vector<cv::DMatch> ratio_match(const cv::Mat& desc1, const cv::Mat& desc2, const double ratio)
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) {
        return good;
    }
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

PairGeometry estimate_pair_geometry(const int i,
                                    const int j,
                                    const std::vector<ImageFeatures>& features,
                                    const std::vector<CameraModel>& cameras,
                                    const MultiviewCalibrationOptions& options)
{
    PairGeometry geometry;
    geometry.i = i;
    geometry.j = j;
    geometry.matches = ratio_match(features[static_cast<size_t>(i)].descriptors,
                                   features[static_cast<size_t>(j)].descriptors,
                                   options.match_ratio);
    if (geometry.matches.size() < static_cast<size_t>(options.min_inlier_matches)) {
        return geometry;
    }

    std::vector<cv::Point2f> points_i;
    std::vector<cv::Point2f> points_j;
    points_i.reserve(geometry.matches.size());
    points_j.reserve(geometry.matches.size());
    for (const auto& match : geometry.matches) {
        const auto pi = features[static_cast<size_t>(i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const auto pj = features[static_cast<size_t>(j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        points_i.push_back(normalize_point(pi, cameras[static_cast<size_t>(i)]));
        points_j.push_back(normalize_point(pj, cameras[static_cast<size_t>(j)]));
    }

    cv::Mat inlier_mask;
    const double threshold = options.ransac_reprojection_threshold /
                             std::max(cameras[static_cast<size_t>(i)].K(0, 0), 1.0);
    cv::Mat E = cv::findEssentialMat(
        points_i, points_j, 1.0, {0.0, 0.0}, cv::RANSAC, 0.999, threshold, 1000, inlier_mask);
    if (E.empty()) {
        return geometry;
    }
    cv::Mat R;
    cv::Mat t;
    const int inliers = cv::recoverPose(E, points_i, points_j, R, t, 1.0, {0.0, 0.0}, inlier_mask);
    geometry.R = R;
    geometry.t = t;
    geometry.inliers = inliers;
    geometry.inlier_mask.assign(inlier_mask.begin<unsigned char>(), inlier_mask.end<unsigned char>());
    return geometry;
}

cv::Mat make_normalized_projection(const CameraModel& camera)
{
    cv::Mat P(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            P.at<double>(r, c) = camera.R(r, c);
        }
        P.at<double>(r, 3) = camera.t(r);
    }
    return P;
}

void triangulate_pair(const PairGeometry& pair,
                      const std::vector<ImageFeatures>& features,
                      const std::vector<CameraModel>& cameras,
                      MultiviewCalibrationResult& result)
{
    const CameraModel& cam_i = cameras[static_cast<size_t>(pair.i)];
    const CameraModel& cam_j = cameras[static_cast<size_t>(pair.j)];
    const cv::Mat P1 = make_normalized_projection(cam_i);
    const cv::Mat P2 = make_normalized_projection(cam_j);

    std::vector<cv::Point2f> points_i;
    std::vector<cv::Point2f> points_j;
    std::vector<cv::DMatch> inlier_matches;
    for (size_t k = 0; k < pair.matches.size(); ++k) {
        if (k >= pair.inlier_mask.size() || pair.inlier_mask[k] == 0) {
            continue;
        }
        const auto& match = pair.matches[k];
        const auto pi = features[static_cast<size_t>(pair.i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const auto pj = features[static_cast<size_t>(pair.j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        points_i.push_back(normalize_point(pi, cam_i));
        points_j.push_back(normalize_point(pj, cam_j));
        inlier_matches.push_back(match);
    }
    if (points_i.empty()) {
        return;
    }

    cv::Mat homogeneous;
    cv::triangulatePoints(P1, P2, points_i, points_j, homogeneous);
    const auto homogeneous_at = [&homogeneous](const int r, const int c) {
        return homogeneous.depth() == CV_64F ? homogeneous.at<double>(r, c)
                                             : static_cast<double>(homogeneous.at<float>(r, c));
    };
    for (int c = 0; c < homogeneous.cols; ++c) {
        const double w = homogeneous_at(3, c);
        if (std::abs(w) < std::numeric_limits<double>::epsilon()) {
            continue;
        }
        Eigen::Vector3d point(homogeneous_at(0, c) / w, homogeneous_at(1, c) / w, homogeneous_at(2, c) / w);
        const Eigen::Vector3d xi = cam_i.R * point + cam_i.t;
        const Eigen::Vector3d xj = cam_j.R * point + cam_j.t;
        if (xi.z() <= 0.0 || xj.z() <= 0.0) {
            continue;
        }

        const auto& match = inlier_matches[static_cast<size_t>(c)];
        const cv::Point2f raw_i = features[static_cast<size_t>(pair.i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const cv::Point2f raw_j = features[static_cast<size_t>(pair.j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        const cv::Point2f proj_i = project_point_cv(point, cam_i);
        const cv::Point2f proj_j = project_point_cv(point, cam_j);
        const cv::Point2f di = proj_i - raw_i;
        const cv::Point2f dj = proj_j - raw_j;
        const double error =
            0.5 * (std::sqrt(di.x * di.x + di.y * di.y) + std::sqrt(dj.x * dj.x + dj.y * dj.y));

        SparsePoint3D sparse;
        sparse.point = point;
        sparse.reprojection_error = error;
        sparse.observations.push_back({pair.i, {raw_i.x, raw_i.y}});
        sparse.observations.push_back({pair.j, {raw_j.x, raw_j.y}});
        result.sparse_points.push_back(std::move(sparse));
    }
}

struct Track {
    std::vector<std::pair<int, int>> observations;
    int point3d = -1;
};

class UnionFind {
public:
    explicit UnionFind(const int n) : parent_(static_cast<size_t>(n)), rank_(static_cast<size_t>(n), 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(const int x)
    {
        if (parent_[static_cast<size_t>(x)] != x) {
            parent_[static_cast<size_t>(x)] = find(parent_[static_cast<size_t>(x)]);
        }
        return parent_[static_cast<size_t>(x)];
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank_[static_cast<size_t>(a)] < rank_[static_cast<size_t>(b)]) {
            std::swap(a, b);
        }
        parent_[static_cast<size_t>(b)] = a;
        if (rank_[static_cast<size_t>(a)] == rank_[static_cast<size_t>(b)]) {
            ++rank_[static_cast<size_t>(a)];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

CameraModel make_initial_camera_with_options(const std::string& label,
                                             const int width,
                                             const int height,
                                             const MultiviewCalibrationOptions& options)
{
    CameraModel camera = make_initial_camera(label, width, height);
    const double factor = options.initial_focal_length_factor > 0.0 ? options.initial_focal_length_factor : 1.2;
    const double focal = factor * static_cast<double>(std::max(width, height));
    camera.K(0, 0) = focal;
    camera.K(1, 1) = focal;
    return camera;
}

cv::Mat eigen_intrinsics_to_cv(const Eigen::Matrix3d& K)
{
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out.at<double>(r, c) = K(r, c);
        }
    }
    return out;
}

cv::Mat simple_radial_distortion_to_cv(const CameraModel& camera)
{
    cv::Mat out = cv::Mat::zeros(4, 1, CV_64F);
    if (!camera.distortion.empty()) {
        out.at<double>(0) = camera.distortion[0];
    }
    return out;
}

bool pair_is_enabled(const int i, const int j, const int n, const MultiviewCalibrationOptions& options)
{
    if (options.matching_window <= 0) {
        return true;
    }
    int distance = std::abs(i - j);
    if (options.wrap_matching) {
        distance = std::min(distance, n - distance);
    }
    return distance <= options.matching_window ||
           (i == options.initial_image1 && j == options.initial_image2) ||
           (i == options.initial_image2 && j == options.initial_image1);
}

std::vector<cv::DMatch> sift_ratio_match(const cv::Mat& desc1, const cv::Mat& desc2, const double ratio)
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) {
        return good;
    }
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

PairGeometry estimate_sift_pair_geometry(const int i,
                                         const int j,
                                         const std::vector<ImageFeatures>& features,
                                         const std::vector<CameraModel>& cameras,
                                         const MultiviewCalibrationOptions& options)
{
    PairGeometry geometry;
    geometry.i = i;
    geometry.j = j;
    geometry.matches = sift_ratio_match(features[static_cast<size_t>(i)].descriptors,
                                        features[static_cast<size_t>(j)].descriptors,
                                        options.match_ratio);
    if (geometry.matches.size() < static_cast<size_t>(std::max(8, options.min_inlier_matches))) {
        return geometry;
    }

    std::vector<cv::Point2f> points_i;
    std::vector<cv::Point2f> points_j;
    points_i.reserve(geometry.matches.size());
    points_j.reserve(geometry.matches.size());
    for (const auto& match : geometry.matches) {
        points_i.push_back(features[static_cast<size_t>(i)].keypoints[static_cast<size_t>(match.queryIdx)].pt);
        points_j.push_back(features[static_cast<size_t>(j)].keypoints[static_cast<size_t>(match.trainIdx)].pt);
    }

    cv::Mat inlier_mask;
    const cv::Mat K = eigen_intrinsics_to_cv(cameras[static_cast<size_t>(i)].K);
    cv::Mat E = cv::findEssentialMat(points_i,
                                     points_j,
                                     K,
                                     cv::RANSAC,
                                     0.999,
                                     options.ransac_reprojection_threshold,
                                     1000,
                                     inlier_mask);
    if (E.empty()) {
        return geometry;
    }
    cv::Mat R;
    cv::Mat t;
    const int inliers = cv::recoverPose(E, points_i, points_j, K, R, t, inlier_mask);
    geometry.R = R;
    geometry.t = t;
    geometry.inliers = inliers;
    geometry.inlier_mask.assign(inlier_mask.begin<unsigned char>(), inlier_mask.end<unsigned char>());
    return geometry;
}

std::vector<Track> build_tracks(const std::vector<ImageFeatures>& features, const std::vector<PairGeometry>& pairs)
{
    std::vector<int> offsets(features.size() + 1, 0);
    for (size_t i = 0; i < features.size(); ++i) {
        offsets[i + 1] = offsets[i] + static_cast<int>(features[i].keypoints.size());
    }
    UnionFind uf(offsets.back());
    for (const auto& pair : pairs) {
        for (size_t k = 0; k < pair.matches.size(); ++k) {
            if (k < pair.inlier_mask.size() && pair.inlier_mask[k] != 0) {
                const auto& m = pair.matches[k];
                uf.unite(offsets[static_cast<size_t>(pair.i)] + m.queryIdx,
                         offsets[static_cast<size_t>(pair.j)] + m.trainIdx);
            }
        }
    }

    std::unordered_map<int, std::vector<std::pair<int, int>>> grouped;
    for (int image_idx = 0; image_idx < static_cast<int>(features.size()); ++image_idx) {
        for (int point_idx = 0; point_idx < static_cast<int>(features[static_cast<size_t>(image_idx)].keypoints.size()); ++point_idx) {
            const int root = uf.find(offsets[static_cast<size_t>(image_idx)] + point_idx);
            grouped[root].push_back({image_idx, point_idx});
        }
    }

    std::vector<Track> tracks;
    for (auto& item : grouped) {
        if (item.second.size() < 2) {
            continue;
        }
        std::set<int> images;
        bool duplicate_image = false;
        for (const auto& obs : item.second) {
            duplicate_image = duplicate_image || !images.insert(obs.first).second;
        }
        if (!duplicate_image) {
            Track track;
            track.observations = std::move(item.second);
            tracks.push_back(std::move(track));
        }
    }
    return tracks;
}

SfMCorrespondenceGraph build_correspondence_graph(const std::vector<ImageFeatures>& features,
                                                  const std::vector<PairGeometry>& pairs)
{
    SfMCorrespondenceGraph graph;
    for (int image_idx = 0; image_idx < static_cast<int>(features.size()); ++image_idx) {
        graph.add_image(image_idx, features[static_cast<size_t>(image_idx)].keypoints.size());
    }
    for (const auto& pair : pairs) {
        graph.add_two_view_geometry(pair);
    }
    return graph;
}

std::vector<Track> build_tracks_from_correspondence_graph(const std::vector<ImageFeatures>& features,
                                                          const SfMCorrespondenceGraph& graph)
{
    std::vector<Track> tracks;
    std::unordered_set<SfMObservationId, SfMObservationIdHash> visited;
    std::vector<SfMObservationId> transitive_corrs;

    for (int image_idx = 0; image_idx < static_cast<int>(features.size()); ++image_idx) {
        for (int point_idx = 0;
             point_idx < static_cast<int>(features[static_cast<size_t>(image_idx)].keypoints.size());
             ++point_idx) {
            const SfMObservationId root{image_idx, point_idx};
            if (visited.count(root) > 0 || !graph.has_correspondences(image_idx, point_idx)) {
                continue;
            }

            graph.extract_transitive_correspondences(image_idx, point_idx, 100, transitive_corrs);
            if (transitive_corrs.size() < 2) {
                visited.insert(root);
                continue;
            }

            std::set<int> image_ids;
            bool duplicate_image = false;
            Track track;
            track.observations.reserve(transitive_corrs.size());
            for (const auto& corr : transitive_corrs) {
                visited.insert(corr);
                if (!image_ids.insert(corr.image).second) {
                    duplicate_image = true;
                }
                track.observations.push_back({corr.image, corr.point2d});
            }
            if (!duplicate_image && track.observations.size() >= 2) {
                tracks.push_back(std::move(track));
            }
        }
    }
    return tracks;
}

SfMReconstructionState make_reconstruction_state(const std::vector<ImageFeatures>& features,
                                                 const std::vector<CameraModel>& cameras,
                                                 const std::vector<bool>& registered)
{
    SfMReconstructionState reconstruction(features.size());
    for (size_t image_idx = 0; image_idx < features.size(); ++image_idx) {
        reconstruction.images[image_idx].camera = cameras[image_idx];
        reconstruction.images[image_idx].registered = registered[image_idx];
        reconstruction.images[image_idx].points2d.reserve(features[image_idx].keypoints.size());
        for (const auto& keypoint : features[image_idx].keypoints) {
            reconstruction.images[image_idx].points2d.push_back({{keypoint.pt.x, keypoint.pt.y}, -1});
        }
    }
    return reconstruction;
}

cv::Mat eigen_rotation_to_rvec(const Eigen::Matrix3d& R)
{
    cv::Mat Rcv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Rcv.at<double>(r, c) = R(r, c);
        }
    }
    cv::Mat rvec;
    cv::Rodrigues(Rcv, rvec);
    return rvec;
}

cv::Mat eigen_translation_to_cv(const Eigen::Vector3d& t)
{
    cv::Mat out(3, 1, CV_64F);
    out.at<double>(0) = t.x();
    out.at<double>(1) = t.y();
    out.at<double>(2) = t.z();
    return out;
}

void set_camera_pose_from_cv(CameraModel& camera, const cv::Mat& rvec, const cv::Mat& tvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    camera.R = cv_to_eigen3x3(Rcv);
    camera.t = cv_to_eigen3(tvec);
}

#ifdef TRADITIONAL_DIC_HAS_CERES
struct BundleObservation {
    int camera = -1;
    int point = -1;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
};

struct SimpleRadialReprojectionCost {
    SimpleRadialReprojectionCost(const double u, const double v) : u_(u), v_(v)
    {
    }

    template <typename T>
    bool operator()(const T* const point,
                    const T* const cam_from_world,
                    const T* const camera_params,
                    T* residuals) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> q(cam_from_world);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(cam_from_world + 4);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> xyz(point);
        const Eigen::Matrix<T, 3, 1> p = q * xyz + t;
        if (p.z() <= T(std::numeric_limits<double>::epsilon())) {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        const T inv_z = T(1.0) / p.z();
        const T x = p.x() * inv_z;
        const T y = p.y() * inv_z;
        const T r2 = x * x + y * y;
        const T radial = T(1.0) + camera_params[3] * r2;
        residuals[0] = camera_params[0] * x * radial + camera_params[1] - T(u_);
        residuals[1] = camera_params[0] * y * radial + camera_params[2] - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& uv)
    {
        return new ceres::AutoDiffCostFunction<SimpleRadialReprojectionCost, 2, 3, 7, 4>(
            new SimpleRadialReprojectionCost(uv.x(), uv.y()));
    }

    double u_;
    double v_;
};

void camera_to_pose_params(const CameraModel& camera, std::array<double, 7>& pose)
{
    Eigen::Quaterniond q(camera.R);
    q.normalize();
    pose[0] = q.x();
    pose[1] = q.y();
    pose[2] = q.z();
    pose[3] = q.w();
    for (int k = 0; k < 3; ++k) {
        pose[static_cast<size_t>(4 + k)] = camera.t(k);
    }
}

void pose_params_to_camera(const std::array<double, 7>& pose, CameraModel& camera)
{
    Eigen::Map<const Eigen::Quaterniond> q(pose.data());
    camera.R = q.normalized().toRotationMatrix();
    camera.t = {pose[4], pose[5], pose[6]};
}

bool simple_radial_params_are_bogus(const std::array<double, 4>& params,
                                    const double max_image_dim)
{
    if (max_image_dim <= 0.0) {
        return true;
    }
    const double focal = params[0];
    const double k1 = params[3];
    if (!std::isfinite(focal) || !std::isfinite(params[1]) ||
        !std::isfinite(params[2]) || !std::isfinite(k1)) {
        return true;
    }
    if (focal < 0.1 * max_image_dim || focal > 10.0 * max_image_dim) {
        return true;
    }
    return std::abs(k1) >= 0.999;
}

std::vector<BundleObservation> collect_bundle_observations(const std::vector<SparsePoint3D>& points)
{
    std::vector<BundleObservation> observations;
    for (int point_idx = 0; point_idx < static_cast<int>(points.size()); ++point_idx) {
        for (const auto& obs : points[static_cast<size_t>(point_idx)].observations) {
            observations.push_back({obs.image_index, point_idx, obs.point});
        }
    }
    return observations;
}

bool run_global_bundle_adjustment(std::vector<CameraModel>& cameras,
                                  std::vector<SparsePoint3D>& points,
                                  const std::vector<bool>& variable_images,
                                  const int anchor_i,
                                  const int anchor_j,
                                  const MultiviewCalibrationOptions& options,
                                  const std::vector<bool>& constant_images = {},
                                  const bool use_robust_loss = true)
{
    if (points.empty() || anchor_i < 0 || anchor_j < 0) {
        return false;
    }
    const std::vector<BundleObservation> observations = collect_bundle_observations(points);
    if (observations.size() < 16) {
        return false;
    }

    std::vector<std::array<double, 7>> poses(cameras.size());
    std::vector<std::array<double, 4>> camera_params(cameras.size());
    std::vector<std::array<double, 3>> point_params(points.size());
    std::vector<bool> active_images(cameras.size(), false);
    for (size_t i = 0; i < cameras.size(); ++i) {
        const bool variable = i < variable_images.size() && variable_images[i];
        const bool constant = i < constant_images.size() && constant_images[i];
        active_images[i] = variable || constant;
    }

    for (size_t i = 0; i < cameras.size(); ++i) {
        camera_to_pose_params(cameras[i], poses[i]);
        camera_params[i][0] = 0.5 * (cameras[i].K(0, 0) + cameras[i].K(1, 1));
        camera_params[i][1] = cameras[i].K(0, 2);
        camera_params[i][2] = cameras[i].K(1, 2);
        camera_params[i][3] = cameras[i].distortion.empty() ? 0.0 : cameras[i].distortion[0];
    }
    if (options.share_intrinsics) {
        std::array<double, 4> mean_params = {0.0, 0.0, 0.0, 0.0};
        size_t num_active = 0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (!active_images[i]) {
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                mean_params[static_cast<size_t>(k)] += camera_params[i][static_cast<size_t>(k)];
            }
            ++num_active;
        }
        if (num_active > 0) {
            for (int k = 0; k < 4; ++k) {
                mean_params[static_cast<size_t>(k)] /= static_cast<double>(num_active);
                camera_params[0][static_cast<size_t>(k)] = mean_params[static_cast<size_t>(k)];
            }
        }
    }
    for (size_t i = 0; i < points.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            point_params[i][static_cast<size_t>(k)] = points[i].point(k);
        }
    }

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    std::unique_ptr<ceres::LossFunction> loss;
    if (use_robust_loss) {
        loss = std::make_unique<ceres::CauchyLoss>(options.filter_max_reproj_error);
    }
    if (options.share_intrinsics) {
        problem.AddParameterBlock(camera_params[0].data(), 4);
    }
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (!active_images[i]) {
            continue;
        }
        problem.AddParameterBlock(poses[i].data(), 7);
        problem.SetManifold(poses[i].data(),
                            new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>(
                                ceres::EigenQuaternionManifold{},
                                ceres::EuclideanManifold<3>{}));
        if (!options.share_intrinsics) {
            problem.AddParameterBlock(camera_params[i].data(), 4);
        }
        if (i >= variable_images.size() || !variable_images[i]) {
            problem.SetParameterBlockConstant(poses[i].data());
            if (!options.share_intrinsics) {
                problem.SetParameterBlockConstant(camera_params[i].data());
            }
        }
    }
    for (size_t i = 0; i < points.size(); ++i) {
        problem.AddParameterBlock(point_params[i].data(), 3);
    }
    for (const auto& obs : observations) {
        if (obs.camera < 0 || obs.camera >= static_cast<int>(cameras.size()) ||
            obs.point < 0 || obs.point >= static_cast<int>(points.size()) ||
            !active_images[static_cast<size_t>(obs.camera)]) {
            continue;
        }
        ceres::CostFunction* cost = SimpleRadialReprojectionCost::Create(obs.uv);
        problem.AddResidualBlock(cost,
                                 loss.get(),
                                 point_params[static_cast<size_t>(obs.point)].data(),
                                 poses[static_cast<size_t>(obs.camera)].data(),
                                 (options.share_intrinsics ? camera_params[0].data()
                                                           : camera_params[static_cast<size_t>(obs.camera)].data()));
    }

    if (anchor_i >= 0 && anchor_i < static_cast<int>(poses.size()) &&
        anchor_i < static_cast<int>(variable_images.size()) &&
        variable_images[static_cast<size_t>(anchor_i)]) {
        problem.SetParameterBlockConstant(poses[static_cast<size_t>(anchor_i)].data());
    }
    if (anchor_j >= 0 && anchor_j < static_cast<int>(poses.size()) &&
        anchor_j < static_cast<int>(variable_images.size()) &&
        variable_images[static_cast<size_t>(anchor_j)] && anchor_j != anchor_i) {
        const Eigen::Vector3d baseline = cameras[static_cast<size_t>(anchor_j)].t -
                                         cameras[static_cast<size_t>(anchor_i)].t;
        Eigen::Index fixed_dim = 0;
        baseline.cwiseAbs().maxCoeff(&fixed_dim);
        problem.SetManifold(poses[static_cast<size_t>(anchor_j)].data(),
                            new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::SubsetManifold>(
                                ceres::EigenQuaternionManifold{},
                                ceres::SubsetManifold(3, {static_cast<int>(fixed_dim)})));
    }

    const auto parameterize_camera_params = [&](double* const params, const double max_dim) {
        if (!options.refine_focal_length && !options.refine_principal_point && !options.refine_extra_params) {
            problem.SetParameterBlockConstant(params);
        } else {
            std::vector<int> constant_params;
            if (!options.refine_focal_length) {
                constant_params.push_back(0);
            }
            if (!options.refine_principal_point) {
                constant_params.push_back(1);
                constant_params.push_back(2);
            }
            if (!options.refine_extra_params) {
                constant_params.push_back(3);
            }
            if (!constant_params.empty()) {
                problem.SetManifold(params, new ceres::SubsetManifold(4, constant_params));
            }
        }
        problem.SetParameterLowerBound(params, 0, 0.1 * max_dim);
        problem.SetParameterUpperBound(params, 0, 10.0 * max_dim);
        problem.SetParameterLowerBound(params, 3, -1.0);
        problem.SetParameterUpperBound(params, 3, 1.0);
    };

    if (options.share_intrinsics) {
        double max_dim = 0.0;
        for (size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
            if (active_images[cam_idx]) {
                max_dim = std::max(max_dim,
                                   static_cast<double>(std::max(cameras[cam_idx].image_width,
                                                                cameras[cam_idx].image_height)));
            }
        }
        parameterize_camera_params(camera_params[0].data(), max_dim);
    } else {
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (!active_images[i] || i >= variable_images.size() || !variable_images[i]) {
                continue;
            }
            const double max_dim = static_cast<double>(std::max(cameras[i].image_width, cameras[i].image_height));
            parameterize_camera_params(camera_params[i].data(), max_dim);
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = 50;
    solver_options.function_tolerance = 1e-6;
    solver_options.gradient_tolerance = 1e-10;
    solver_options.parameter_tolerance = 1e-8;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.num_threads = 1;
    solver_options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        return false;
    }

    if (options.share_intrinsics) {
        double max_dim = 0.0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (active_images[i]) {
                max_dim = std::max(max_dim,
                                   static_cast<double>(std::max(cameras[i].image_width,
                                                                cameras[i].image_height)));
            }
        }
        if (simple_radial_params_are_bogus(camera_params[0], max_dim)) {
            return false;
        }
    } else {
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (!active_images[i]) {
                continue;
            }
            const double max_dim = static_cast<double>(std::max(cameras[i].image_width, cameras[i].image_height));
            if (simple_radial_params_are_bogus(camera_params[i], max_dim)) {
                return false;
            }
        }
    }

    for (size_t i = 0; i < cameras.size(); ++i) {
        if (!active_images[i]) {
            continue;
        }
        pose_params_to_camera(poses[i], cameras[i]);
        const auto& params = options.share_intrinsics ? camera_params[0] : camera_params[i];
        cameras[i].K(0, 0) = params[0];
        cameras[i].K(1, 1) = params[0];
        cameras[i].K(0, 2) = params[1];
        cameras[i].K(1, 2) = params[2];
        if (cameras[i].distortion.empty()) {
            cameras[i].distortion.assign(1, 0.0);
        }
        cameras[i].distortion[0] = params[3];
    }
    for (size_t i = 0; i < points.size(); ++i) {
        points[i].point = {point_params[i][0], point_params[i][1], point_params[i][2]};
        points[i].reprojection_error = mean_reprojection_error(points[i].point, cameras, points[i].observations);
    }
    return true;
}
#endif

double triangulation_angle_degrees(const Eigen::Vector3d& point, const CameraModel& a, const CameraModel& b)
{
    const Eigen::Vector3d ca = a.camera_center();
    const Eigen::Vector3d cb = b.camera_center();
    const Eigen::Vector3d va = (ca - point).normalized();
    const Eigen::Vector3d vb = (cb - point).normalized();
    const double dot = std::clamp(va.dot(vb), -1.0, 1.0);
    return std::acos(dot) * 180.0 / 3.14159265358979323846;
}

FeatureTrackObservation feature_observation_from_id(const std::vector<ImageFeatures>& features,
                                                    const SfMObservationId& obs)
{
    const auto pt = features[static_cast<size_t>(obs.image)].keypoints[static_cast<size_t>(obs.point2d)].pt;
    return {obs.image, {pt.x, pt.y}};
}

bool has_duplicate_image(const std::vector<SfMObservationId>& observations)
{
    std::set<int> images;
    for (const auto& obs : observations) {
        if (!images.insert(obs.image).second) {
            return true;
        }
    }
    return false;
}

std::vector<FeatureTrackObservation> feature_observations_from_ids(const std::vector<ImageFeatures>& features,
                                                                   const std::vector<SfMObservationId>& observations)
{
    std::vector<FeatureTrackObservation> out;
    out.reserve(observations.size());
    for (const auto& obs : observations) {
        out.push_back(feature_observation_from_id(features, obs));
    }
    return out;
}

bool triangulate_sfm_observations(const std::vector<ImageFeatures>& features,
                                  const std::vector<CameraModel>& cameras,
                                  const std::vector<SfMObservationId>& observations,
                                  const MultiviewCalibrationOptions& options,
                                  const double max_reproj_error,
                                  Eigen::Vector3d& xyz,
                                  std::vector<char>& inlier_mask)
{
    inlier_mask.assign(observations.size(), 0);
    if (observations.size() < 2 || has_duplicate_image(observations)) {
        return false;
    }
    const std::vector<FeatureTrackObservation> feature_observations =
        feature_observations_from_ids(features, observations);
    try {
        xyz = triangulate_linear(cameras, feature_observations);
    } catch (const std::exception&) {
        return false;
    }

    double max_angle = 0.0;
    for (size_t i = 0; i < feature_observations.size(); ++i) {
        const auto& cam_i = cameras[static_cast<size_t>(feature_observations[i].image_index)];
        const Eigen::Vector3d xi = cam_i.R * xyz + cam_i.t;
        if (xi.z() <= 0.0) {
            return false;
        }
        for (size_t j = i + 1; j < feature_observations.size(); ++j) {
            const auto& cam_j = cameras[static_cast<size_t>(feature_observations[j].image_index)];
            max_angle = std::max(max_angle, triangulation_angle_degrees(xyz, cam_i, cam_j));
        }
    }
    if (max_angle < options.min_triangulation_angle_degrees) {
        return false;
    }

    size_t num_inliers = 0;
    for (size_t i = 0; i < feature_observations.size(); ++i) {
        const auto& obs = feature_observations[i];
        const double error = reprojection_error(xyz, cameras[static_cast<size_t>(obs.image_index)], obs.point);
        if (std::isfinite(error) && error <= max_reproj_error) {
            inlier_mask[i] = 1;
            ++num_inliers;
        }
    }
    return num_inliers >= 2;
}

class SfMIncrementalTriangulator {
public:
    struct Options {
        int max_transitivity = 1;
        int complete_max_transitivity = 5;
        double create_max_reproj_error = 4.0;
        double continue_max_reproj_error = 4.0;
        double merge_max_reproj_error = 4.0;
        double complete_max_reproj_error = 4.0;
        bool ignore_two_view_tracks = true;
    };

    SfMIncrementalTriangulator(const SfMCorrespondenceGraph& graph,
                               const std::vector<ImageFeatures>& features,
                               SfMReconstructionState& reconstruction,
                               SfMObservationManager& observation_manager,
                               const MultiviewCalibrationOptions& calibration_options)
        : graph_(graph),
          features_(features),
          reconstruction_(reconstruction),
          observation_manager_(observation_manager),
          calibration_options_(calibration_options)
    {
    }

    size_t triangulate_image(const int image_id, const Options& options)
    {
        if (!is_registered(image_id)) {
            return 0;
        }
        size_t changed = 0;
        for (int point_idx = 0; point_idx < static_cast<int>(graph_.num_points2d(image_id)); ++point_idx) {
            std::vector<SfMCorrData> corrs_data;
            const size_t num_triangulated = find(image_id, point_idx, options.max_transitivity, corrs_data);
            if (corrs_data.empty()) {
                continue;
            }
            const SfMCorrData ref{image_id,
                                  point_idx,
                                  reconstruction_.images[static_cast<size_t>(image_id)]
                                      .points2d[static_cast<size_t>(point_idx)]
                                      .point3d};
            if (num_triangulated > 0) {
                changed += continue_track(ref, corrs_data, options);
            }
            corrs_data.push_back(ref);
            changed += create(corrs_data, options);
        }
        return changed;
    }

    size_t complete_image(const int image_id, const Options& options)
    {
        if (!is_registered(image_id)) {
            return 0;
        }
        size_t changed = 0;
        for (int point_idx = 0; point_idx < static_cast<int>(graph_.num_points2d(image_id)); ++point_idx) {
            const int point3d = reconstruction_.images[static_cast<size_t>(image_id)]
                                    .points2d[static_cast<size_t>(point_idx)]
                                    .point3d;
            if (point3d >= 0) {
                changed += complete(point3d, options);
                continue;
            }
            if (options.ignore_two_view_tracks && graph_.is_two_view_observation(image_id, point_idx)) {
                continue;
            }
            std::vector<SfMCorrData> corrs_data;
            const size_t num_triangulated = find(image_id, point_idx, options.max_transitivity, corrs_data);
            if (num_triangulated > 0 || corrs_data.empty()) {
                continue;
            }
            corrs_data.push_back({image_id, point_idx, -1});
            changed += create(corrs_data, options);
        }
        return changed;
    }

    size_t complete_all_tracks(const Options& options)
    {
        size_t changed = 0;
        const size_t num_points = reconstruction_.points3d.size();
        for (int point3d = 0; point3d < static_cast<int>(num_points); ++point3d) {
            changed += complete(point3d, options);
        }
        return changed;
    }

    size_t merge_all_tracks(const Options& options)
    {
        size_t changed = 0;
        const size_t num_points = reconstruction_.points3d.size();
        for (int point3d = 0; point3d < static_cast<int>(num_points); ++point3d) {
            changed += merge(point3d, options);
        }
        return changed;
    }

private:
    struct SfMCorrData {
        int image = -1;
        int point2d = -1;
        int point3d = -1;
    };

    bool is_registered(const int image_id) const
    {
        return image_id >= 0 && static_cast<size_t>(image_id) < reconstruction_.images.size() &&
               reconstruction_.images[static_cast<size_t>(image_id)].registered;
    }

    size_t find(const int image_id, const int point2d_idx, const int transitivity, std::vector<SfMCorrData>& corrs_data) const
    {
        std::vector<SfMObservationId> corrs;
        graph_.extract_transitive_correspondences(image_id, point2d_idx, transitivity, corrs);
        corrs_data.clear();
        corrs_data.reserve(corrs.size());
        size_t num_triangulated = 0;
        for (const auto& corr : corrs) {
            if (corr.image == image_id && corr.point2d == point2d_idx) {
                continue;
            }
            if (!reconstruction_.valid_observation(corr) || !is_registered(corr.image)) {
                continue;
            }
            const int point3d = reconstruction_.images[static_cast<size_t>(corr.image)]
                                    .points2d[static_cast<size_t>(corr.point2d)]
                                    .point3d;
            corrs_data.push_back({corr.image, corr.point2d, point3d});
            if (point3d >= 0 && static_cast<size_t>(point3d) < reconstruction_.points3d.size() &&
                reconstruction_.points3d[static_cast<size_t>(point3d)].valid) {
                ++num_triangulated;
            }
        }
        return num_triangulated;
    }

    size_t create(const std::vector<SfMCorrData>& corrs_data, const Options& options)
    {
        std::vector<SfMObservationId> create_observations;
        create_observations.reserve(corrs_data.size());
        for (const auto& corr : corrs_data) {
            if (corr.point3d < 0) {
                create_observations.push_back({corr.image, corr.point2d});
            }
        }
        if (create_observations.size() < 2 ||
            (options.ignore_two_view_tracks && create_observations.size() == 2 &&
             graph_.is_two_view_observation(create_observations[0].image, create_observations[0].point2d))) {
            return 0;
        }

        Eigen::Vector3d xyz;
        std::vector<char> inlier_mask;
        const std::vector<CameraModel> cameras = cameras_from_state();
        if (!triangulate_sfm_observations(features_,
                                          cameras,
                                          create_observations,
                                          calibration_options_,
                                          options.create_max_reproj_error,
                                          xyz,
                                          inlier_mask)) {
            return 0;
        }

        std::vector<SfMObservationId> inlier_track;
        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                inlier_track.push_back(create_observations[i]);
            }
        }
        if (inlier_track.size() < 2 || has_duplicate_image(inlier_track)) {
            return 0;
        }
        observation_manager_.add_point3d(xyz, inlier_track);
        return inlier_track.size();
    }

    size_t continue_track(const SfMCorrData& ref_corr, const std::vector<SfMCorrData>& corrs_data, const Options& options)
    {
        if (ref_corr.point3d >= 0) {
            return 0;
        }
        const auto ref_obs = feature_observation_from_id(features_, {ref_corr.image, ref_corr.point2d});
        double best_error = std::numeric_limits<double>::infinity();
        int best_point3d = -1;
        for (const auto& corr : corrs_data) {
            if (corr.point3d < 0 || static_cast<size_t>(corr.point3d) >= reconstruction_.points3d.size() ||
                !reconstruction_.points3d[static_cast<size_t>(corr.point3d)].valid) {
                continue;
            }
            const double error = reprojection_error(reconstruction_.points3d[static_cast<size_t>(corr.point3d)].xyz,
                                                   reconstruction_.images[static_cast<size_t>(ref_corr.image)].camera,
                                                   ref_obs.point);
            if (error < best_error) {
                best_error = error;
                best_point3d = corr.point3d;
            }
        }
        if (best_point3d >= 0 && best_error <= options.continue_max_reproj_error &&
            observation_manager_.add_observation(best_point3d, {ref_corr.image, ref_corr.point2d})) {
            return 1;
        }
        return 0;
    }

    size_t merge(const int point3d_id, const Options& options)
    {
        if (point3d_id < 0 || static_cast<size_t>(point3d_id) >= reconstruction_.points3d.size() ||
            !reconstruction_.points3d[static_cast<size_t>(point3d_id)].valid) {
            return 0;
        }
        std::vector<SfMObservationId> track = reconstruction_.points3d[static_cast<size_t>(point3d_id)].track;
        for (const auto& obs : track) {
            for (const auto& corr : graph_.find_correspondences(obs.image, obs.point2d)) {
                if (!reconstruction_.valid_observation(corr) || !is_registered(corr.image)) {
                    continue;
                }
                const int other_id = reconstruction_.images[static_cast<size_t>(corr.image)]
                                         .points2d[static_cast<size_t>(corr.point2d)]
                                         .point3d;
                if (other_id < 0 || other_id == point3d_id ||
                    static_cast<size_t>(other_id) >= reconstruction_.points3d.size() ||
                    !reconstruction_.points3d[static_cast<size_t>(other_id)].valid) {
                    continue;
                }
                const auto& point = reconstruction_.points3d[static_cast<size_t>(point3d_id)];
                const auto& other = reconstruction_.points3d[static_cast<size_t>(other_id)];
                const Eigen::Vector3d merged_xyz =
                    (static_cast<double>(point.track.size()) * point.xyz +
                     static_cast<double>(other.track.size()) * other.xyz) /
                    static_cast<double>(point.track.size() + other.track.size());
                if (track_reprojection_within(point.track, merged_xyz, options.merge_max_reproj_error) &&
                    track_reprojection_within(other.track, merged_xyz, options.merge_max_reproj_error) &&
                    observation_manager_.merge_point3d(point3d_id, other_id, merged_xyz)) {
                    return point.track.size() + other.track.size();
                }
            }
        }
        return 0;
    }

    size_t complete(const int point3d_id, const Options& options)
    {
        if (point3d_id < 0 || static_cast<size_t>(point3d_id) >= reconstruction_.points3d.size() ||
            !reconstruction_.points3d[static_cast<size_t>(point3d_id)].valid) {
            return 0;
        }
        size_t changed = 0;
        std::vector<SfMObservationId> current = reconstruction_.points3d[static_cast<size_t>(point3d_id)].track;
        std::unordered_set<SfMObservationId, SfMObservationIdHash> visited(current.begin(), current.end());
        for (int depth = 0; depth < options.complete_max_transitivity && !current.empty(); ++depth) {
            std::vector<SfMObservationId> next_queue;
            for (const auto& seed : current) {
                for (const auto& corr : graph_.find_correspondences(seed.image, seed.point2d)) {
                    if (!visited.insert(corr).second || !reconstruction_.valid_observation(corr) || !is_registered(corr.image)) {
                        continue;
                    }
                    auto& point2d = reconstruction_.images[static_cast<size_t>(corr.image)].points2d[static_cast<size_t>(corr.point2d)];
                    if (point2d.point3d >= 0) {
                        continue;
                    }
                    const auto feature_obs = feature_observation_from_id(features_, corr);
                    const double error = reprojection_error(reconstruction_.points3d[static_cast<size_t>(point3d_id)].xyz,
                                                           reconstruction_.images[static_cast<size_t>(corr.image)].camera,
                                                           feature_obs.point);
                    if (std::isfinite(error) && error <= options.complete_max_reproj_error &&
                        observation_manager_.add_observation(point3d_id, corr)) {
                        ++changed;
                        next_queue.push_back(corr);
                    }
                }
            }
            current = std::move(next_queue);
        }
        return changed;
    }

    bool track_reprojection_within(const std::vector<SfMObservationId>& track,
                                   const Eigen::Vector3d& xyz,
                                   const double max_error) const
    {
        for (const auto& obs : track) {
            const auto feature_obs = feature_observation_from_id(features_, obs);
            const double error = reprojection_error(xyz,
                                                   reconstruction_.images[static_cast<size_t>(obs.image)].camera,
                                                   feature_obs.point);
            if (!std::isfinite(error) || error > max_error ||
                (reconstruction_.images[static_cast<size_t>(obs.image)].camera.R * xyz +
                 reconstruction_.images[static_cast<size_t>(obs.image)].camera.t)
                        .z() <= 0.0) {
                return false;
            }
        }
        return true;
    }

    std::vector<CameraModel> cameras_from_state() const
    {
        std::vector<CameraModel> cameras;
        cameras.reserve(reconstruction_.images.size());
        for (const auto& image : reconstruction_.images) {
            cameras.push_back(image.camera);
        }
        return cameras;
    }

    const SfMCorrespondenceGraph& graph_;
    const std::vector<ImageFeatures>& features_;
    SfMReconstructionState& reconstruction_;
    SfMObservationManager& observation_manager_;
    const MultiviewCalibrationOptions& calibration_options_;
};

bool triangulate_track(const Track& track,
                       const std::vector<ImageFeatures>& features,
                       const std::vector<CameraModel>& cameras,
                       const std::vector<bool>& registered,
                       const MultiviewCalibrationOptions& options,
                       SparsePoint3D& sparse)
{
    std::vector<FeatureTrackObservation> observations;
    for (const auto& obs : track.observations) {
        if (registered[static_cast<size_t>(obs.first)]) {
            const auto pt = features[static_cast<size_t>(obs.first)].keypoints[static_cast<size_t>(obs.second)].pt;
            observations.push_back({obs.first, {pt.x, pt.y}});
        }
    }
    if (observations.size() < 2) {
        return false;
    }
    Eigen::Vector3d point;
    try {
        point = triangulate_linear(cameras, observations);
    } catch (const std::exception&) {
        return false;
    }
    double max_angle = 0.0;
    for (size_t a = 0; a < observations.size(); ++a) {
        const auto& cam_a = cameras[static_cast<size_t>(observations[a].image_index)];
        const Eigen::Vector3d xa = cam_a.R * point + cam_a.t;
        if (xa.z() <= 0.0) {
            return false;
        }
        for (size_t b = a + 1; b < observations.size(); ++b) {
            const auto& cam_b = cameras[static_cast<size_t>(observations[b].image_index)];
            max_angle = std::max(max_angle, triangulation_angle_degrees(point, cam_a, cam_b));
        }
    }
    if (max_angle < options.min_triangulation_angle_degrees) {
        return false;
    }
    const double error = mean_reprojection_error(point, cameras, observations);
    if (!std::isfinite(error) || error > options.filter_max_reproj_error) {
        return false;
    }
    sparse.point = point;
    sparse.observations = std::move(observations);
    sparse.reprojection_error = error;
    return true;
}

int triangulate_registered_tracks(std::vector<Track>& tracks,
                                  const std::vector<ImageFeatures>& features,
                                  const std::vector<CameraModel>& cameras,
                                  const std::vector<bool>& registered,
                                  const MultiviewCalibrationOptions& options,
                                  std::vector<SparsePoint3D>& points)
{
    int changed = 0;
    for (auto& track : tracks) {
        SparsePoint3D sparse;
        if (!triangulate_track(track, features, cameras, registered, options, sparse)) {
            continue;
        }
        if (track.point3d < 0) {
            track.point3d = static_cast<int>(points.size());
            points.push_back(std::move(sparse));
            ++changed;
        } else {
            points[static_cast<size_t>(track.point3d)] = std::move(sparse);
        }
    }
    return changed;
}

int filter_points_after_bundle(std::vector<Track>& tracks,
                               const std::vector<CameraModel>& cameras,
                               std::vector<SparsePoint3D>& points,
                               const MultiviewCalibrationOptions& options)
{
    std::vector<int> remap(points.size(), -1);
    std::vector<SparsePoint3D> kept;
    kept.reserve(points.size());
    int filtered = 0;
    for (size_t point_idx = 0; point_idx < points.size(); ++point_idx) {
        points[point_idx].reprojection_error =
            mean_reprojection_error(points[point_idx].point, cameras, points[point_idx].observations);
        bool keep = std::isfinite(points[point_idx].reprojection_error) &&
                    points[point_idx].reprojection_error <= options.filter_max_reproj_error;
        for (const auto& obs : points[point_idx].observations) {
            const auto& cam = cameras[static_cast<size_t>(obs.image_index)];
            if ((cam.R * points[point_idx].point + cam.t).z() <= 0.0) {
                keep = false;
                break;
            }
        }
        if (keep) {
            remap[point_idx] = static_cast<int>(kept.size());
            kept.push_back(std::move(points[point_idx]));
        } else {
            ++filtered;
        }
    }
    points = std::move(kept);
    for (auto& track : tracks) {
        if (track.point3d < 0) {
            continue;
        }
        const int new_idx = remap[static_cast<size_t>(track.point3d)];
        track.point3d = new_idx;
    }
    return filtered;
}

bool register_image_pnp(const int image_idx,
                        const std::vector<Track>& tracks,
                        const std::vector<ImageFeatures>& features,
                        const std::vector<SparsePoint3D>& points,
                        CameraModel& camera,
                        const MultiviewCalibrationOptions& options)
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    for (const auto& track : tracks) {
        if (track.point3d < 0) {
            continue;
        }
        for (const auto& obs : track.observations) {
            if (obs.first == image_idx) {
                const Eigen::Vector3d& xyz = points[static_cast<size_t>(track.point3d)].point;
                const auto uv = features[static_cast<size_t>(image_idx)].keypoints[static_cast<size_t>(obs.second)].pt;
                object_points.emplace_back(static_cast<float>(xyz.x()), static_cast<float>(xyz.y()), static_cast<float>(xyz.z()));
                image_points.push_back(uv);
            }
        }
    }
    if (object_points.size() < static_cast<size_t>(std::max(6, options.abs_pose_min_num_inliers))) {
        return false;
    }
    const cv::Mat K = eigen_intrinsics_to_cv(camera.K);
    const cv::Mat distortion = simple_radial_distortion_to_cv(camera);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool ok = cv::solvePnPRansac(object_points,
                                       image_points,
                                       K,
                                       distortion,
                                       rvec,
                                       tvec,
                                       false,
                                       1000,
                                       static_cast<float>(options.abs_pose_max_error),
                                       0.999,
                                       inliers,
                                       cv::SOLVEPNP_EPNP);
    if (!ok || inliers.rows < options.abs_pose_min_num_inliers ||
        static_cast<double>(inliers.rows) / static_cast<double>(object_points.size()) < options.abs_pose_min_inlier_ratio) {
        return false;
    }
    std::vector<cv::Point3f> inlier_object_points;
    std::vector<cv::Point2f> inlier_image_points;
    for (int r = 0; r < inliers.rows; ++r) {
        const int idx = inliers.at<int>(r);
        inlier_object_points.push_back(object_points[static_cast<size_t>(idx)]);
        inlier_image_points.push_back(image_points[static_cast<size_t>(idx)]);
    }
    cv::solvePnP(inlier_object_points, inlier_image_points, K, distortion, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    set_camera_pose_from_cv(camera, rvec, tvec);
    return true;
}

std::vector<std::vector<int>> build_point3d_lookup(const std::vector<ImageFeatures>& features,
                                                   const std::vector<Track>& tracks)
{
    std::vector<std::vector<int>> lookup(features.size());
    for (size_t image_idx = 0; image_idx < features.size(); ++image_idx) {
        lookup[image_idx].assign(features[image_idx].keypoints.size(), -1);
    }
    for (const auto& track : tracks) {
        if (track.point3d < 0) {
            continue;
        }
        for (const auto& obs : track.observations) {
            if (obs.first < 0 || obs.second < 0 ||
                static_cast<size_t>(obs.first) >= lookup.size() ||
                static_cast<size_t>(obs.second) >= lookup[static_cast<size_t>(obs.first)].size()) {
                continue;
            }
            lookup[static_cast<size_t>(obs.first)][static_cast<size_t>(obs.second)] = track.point3d;
        }
    }
    return lookup;
}

size_t count_visible_points3d_from_graph(const int image_idx,
                                         const SfMCorrespondenceGraph& graph,
                                         const std::vector<std::vector<int>>& point3d_lookup,
                                         const std::vector<bool>& registered)
{
    size_t visible = 0;
    std::unordered_set<int> seen_points3d;
    for (int point_idx = 0; point_idx < static_cast<int>(graph.num_points2d(image_idx)); ++point_idx) {
        bool point_has_visible_3d = false;
        seen_points3d.clear();
        for (const auto& corr : graph.find_correspondences(image_idx, point_idx)) {
            if (corr.image < 0 || corr.point2d < 0 ||
                static_cast<size_t>(corr.image) >= point3d_lookup.size() ||
                static_cast<size_t>(corr.point2d) >= point3d_lookup[static_cast<size_t>(corr.image)].size() ||
                !registered[static_cast<size_t>(corr.image)]) {
                continue;
            }
            const int point3d = point3d_lookup[static_cast<size_t>(corr.image)][static_cast<size_t>(corr.point2d)];
            if (point3d >= 0 && seen_points3d.insert(point3d).second) {
                point_has_visible_3d = true;
            }
        }
        if (point_has_visible_3d) {
            ++visible;
        }
    }
    return visible;
}

bool register_image_pnp_from_graph(const int image_idx,
                                   const SfMCorrespondenceGraph& graph,
                                   const std::vector<std::vector<int>>& point3d_lookup,
                                   const std::vector<ImageFeatures>& features,
                                   const std::vector<SparsePoint3D>& points,
                                   const std::vector<bool>& registered,
                                   CameraModel& camera,
                                   const MultiviewCalibrationOptions& options)
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::unordered_set<int> corr_point3d_ids;

    for (int point_idx = 0; point_idx < static_cast<int>(graph.num_points2d(image_idx)); ++point_idx) {
        corr_point3d_ids.clear();
        for (const auto& corr : graph.find_correspondences(image_idx, point_idx)) {
            if (corr.image < 0 || corr.point2d < 0 ||
                static_cast<size_t>(corr.image) >= point3d_lookup.size() ||
                static_cast<size_t>(corr.point2d) >= point3d_lookup[static_cast<size_t>(corr.image)].size() ||
                !registered[static_cast<size_t>(corr.image)]) {
                continue;
            }
            const int point3d = point3d_lookup[static_cast<size_t>(corr.image)][static_cast<size_t>(corr.point2d)];
            if (point3d < 0 || point3d >= static_cast<int>(points.size()) || !corr_point3d_ids.insert(point3d).second) {
                continue;
            }
            const Eigen::Vector3d& xyz = points[static_cast<size_t>(point3d)].point;
            const auto uv = features[static_cast<size_t>(image_idx)].keypoints[static_cast<size_t>(point_idx)].pt;
            object_points.emplace_back(static_cast<float>(xyz.x()), static_cast<float>(xyz.y()), static_cast<float>(xyz.z()));
            image_points.push_back(uv);
        }
    }

    if (object_points.size() < static_cast<size_t>(std::max(6, options.abs_pose_min_num_inliers))) {
        return false;
    }
    const cv::Mat K = eigen_intrinsics_to_cv(camera.K);
    const cv::Mat distortion = simple_radial_distortion_to_cv(camera);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool ok = cv::solvePnPRansac(object_points,
                                       image_points,
                                       K,
                                       distortion,
                                       rvec,
                                       tvec,
                                       false,
                                       1000,
                                       static_cast<float>(options.abs_pose_max_error),
                                       0.999,
                                       inliers,
                                       cv::SOLVEPNP_EPNP);
    if (!ok || inliers.rows < options.abs_pose_min_num_inliers ||
        static_cast<double>(inliers.rows) / static_cast<double>(object_points.size()) < options.abs_pose_min_inlier_ratio) {
        return false;
    }
    std::vector<cv::Point3f> inlier_object_points;
    std::vector<cv::Point2f> inlier_image_points;
    for (int r = 0; r < inliers.rows; ++r) {
        const int idx = inliers.at<int>(r);
        inlier_object_points.push_back(object_points[static_cast<size_t>(idx)]);
        inlier_image_points.push_back(image_points[static_cast<size_t>(idx)]);
    }
    cv::solvePnP(inlier_object_points, inlier_image_points, K, distortion, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    set_camera_pose_from_cv(camera, rvec, tvec);
    return true;
}

bool register_image_pnp_from_state(const int image_idx,
                                   const SfMCorrespondenceGraph& graph,
                                   SfMReconstructionState& reconstruction,
                                   SfMObservationManager& observation_manager,
                                   CameraModel& camera,
                                   const MultiviewCalibrationOptions& options)
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<std::pair<int, int>> tri_corrs;
    std::unordered_set<int> corr_point3d_ids;

    for (int point_idx = 0; point_idx < static_cast<int>(graph.num_points2d(image_idx)); ++point_idx) {
        corr_point3d_ids.clear();
        for (const auto& corr : graph.find_correspondences(image_idx, point_idx)) {
            if (!reconstruction.valid_observation(corr) ||
                !reconstruction.images[static_cast<size_t>(corr.image)].registered) {
                continue;
            }
            const int point3d = reconstruction.images[static_cast<size_t>(corr.image)]
                                    .points2d[static_cast<size_t>(corr.point2d)]
                                    .point3d;
            if (point3d < 0 || static_cast<size_t>(point3d) >= reconstruction.points3d.size() ||
                !reconstruction.points3d[static_cast<size_t>(point3d)].valid ||
                !corr_point3d_ids.insert(point3d).second) {
                continue;
            }
            const Eigen::Vector3d& xyz = reconstruction.points3d[static_cast<size_t>(point3d)].xyz;
            const Eigen::Vector2d& uv = reconstruction.images[static_cast<size_t>(image_idx)]
                                            .points2d[static_cast<size_t>(point_idx)]
                                            .xy;
            object_points.emplace_back(static_cast<float>(xyz.x()), static_cast<float>(xyz.y()), static_cast<float>(xyz.z()));
            image_points.emplace_back(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
            tri_corrs.push_back({point_idx, point3d});
        }
    }

    if (object_points.size() < static_cast<size_t>(std::max(6, options.abs_pose_min_num_inliers))) {
        return false;
    }
    const cv::Mat K = eigen_intrinsics_to_cv(camera.K);
    const cv::Mat distortion = simple_radial_distortion_to_cv(camera);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool ok = cv::solvePnPRansac(object_points,
                                       image_points,
                                       K,
                                       distortion,
                                       rvec,
                                       tvec,
                                       false,
                                       1000,
                                       static_cast<float>(options.abs_pose_max_error),
                                       0.999,
                                       inliers,
                                       cv::SOLVEPNP_EPNP);
    if (!ok || inliers.rows < options.abs_pose_min_num_inliers ||
        static_cast<double>(inliers.rows) / static_cast<double>(object_points.size()) < options.abs_pose_min_inlier_ratio) {
        return false;
    }

    std::vector<cv::Point3f> inlier_object_points;
    std::vector<cv::Point2f> inlier_image_points;
    inlier_object_points.reserve(static_cast<size_t>(inliers.rows));
    inlier_image_points.reserve(static_cast<size_t>(inliers.rows));
    for (int r = 0; r < inliers.rows; ++r) {
        const int idx = inliers.at<int>(r);
        inlier_object_points.push_back(object_points[static_cast<size_t>(idx)]);
        inlier_image_points.push_back(image_points[static_cast<size_t>(idx)]);
    }
    cv::solvePnP(inlier_object_points, inlier_image_points, K, distortion, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    set_camera_pose_from_cv(camera, rvec, tvec);
    reconstruction.images[static_cast<size_t>(image_idx)].camera = camera;
    reconstruction.images[static_cast<size_t>(image_idx)].registered = true;

    for (int r = 0; r < inliers.rows; ++r) {
        const int idx = inliers.at<int>(r);
        observation_manager.add_observation(tri_corrs[static_cast<size_t>(idx)].second,
                                            {image_idx, tri_corrs[static_cast<size_t>(idx)].first});
    }
    return true;
}

std::vector<SparsePoint3D> export_sparse_points_from_state(SfMReconstructionState& reconstruction)
{
    std::vector<SparsePoint3D> sparse_points;
    sparse_points.reserve(reconstruction.points3d.size());
    for (auto& point3d : reconstruction.points3d) {
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        SparsePoint3D sparse;
        sparse.point = point3d.xyz;
        sparse.observations.reserve(point3d.track.size());
        for (const auto& obs : point3d.track) {
            if (!reconstruction.valid_observation(obs) ||
                !reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                continue;
            }
            sparse.observations.push_back({obs.image,
                                           reconstruction.images[static_cast<size_t>(obs.image)]
                                               .points2d[static_cast<size_t>(obs.point2d)]
                                               .xy});
        }
        if (sparse.observations.size() < 2) {
            continue;
        }
        std::vector<CameraModel> cameras;
        cameras.reserve(reconstruction.images.size());
        for (const auto& image : reconstruction.images) {
            cameras.push_back(image.camera);
        }
        sparse.reprojection_error = mean_reprojection_error(sparse.point, cameras, sparse.observations);
        point3d.reprojection_error = sparse.reprojection_error;
        sparse_points.push_back(std::move(sparse));
    }
    return sparse_points;
}

void import_bundle_result_to_state(const std::vector<CameraModel>& cameras,
                                   const std::vector<SparsePoint3D>& sparse_points,
                                   SfMReconstructionState& reconstruction)
{
    for (size_t i = 0; i < cameras.size() && i < reconstruction.images.size(); ++i) {
        reconstruction.images[i].camera = cameras[i];
    }
    size_t sparse_idx = 0;
    for (auto& point3d : reconstruction.points3d) {
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        if (sparse_idx >= sparse_points.size()) {
            break;
        }
        point3d.xyz = sparse_points[sparse_idx].point;
        point3d.reprojection_error = sparse_points[sparse_idx].reprojection_error;
        ++sparse_idx;
    }
}

size_t count_state_observations(const SfMReconstructionState& reconstruction)
{
    size_t count = 0;
    for (const auto& point3d : reconstruction.points3d) {
        if (point3d.valid) {
            count += point3d.track.size();
        }
    }
    return count;
}

double max_triangulation_angle_degrees(const Eigen::Vector3d& point,
                                       const std::vector<FeatureTrackObservation>& observations,
                                       const std::vector<CameraModel>& cameras)
{
    double max_angle = 0.0;
    for (size_t i = 0; i < observations.size(); ++i) {
        const int image_i = observations[i].image_index;
        if (image_i < 0 || static_cast<size_t>(image_i) >= cameras.size()) {
            continue;
        }
        for (size_t j = i + 1; j < observations.size(); ++j) {
            const int image_j = observations[j].image_index;
            if (image_j < 0 || static_cast<size_t>(image_j) >= cameras.size()) {
                continue;
            }
            max_angle = std::max(max_angle,
                                 triangulation_angle_degrees(point,
                                                             cameras[static_cast<size_t>(image_i)],
                                                             cameras[static_cast<size_t>(image_j)]));
        }
    }
    return max_angle;
}

size_t filter_state_points(SfMReconstructionState& reconstruction,
                           const MultiviewCalibrationOptions& options,
                           const std::vector<int>* filter_image_ids = nullptr,
                           const std::vector<int>* filter_point_ids = nullptr)
{
    std::vector<CameraModel> cameras;
    cameras.reserve(reconstruction.images.size());
    for (const auto& image : reconstruction.images) {
        cameras.push_back(image.camera);
    }
    std::unordered_set<int> candidate_point_ids;
    if (filter_image_ids == nullptr && filter_point_ids == nullptr) {
        for (int point_id = 0; point_id < static_cast<int>(reconstruction.points3d.size()); ++point_id) {
            candidate_point_ids.insert(point_id);
        }
    } else {
        if (filter_point_ids != nullptr) {
            candidate_point_ids.insert(filter_point_ids->begin(), filter_point_ids->end());
        }
        if (filter_image_ids != nullptr) {
            for (const int image_id : *filter_image_ids) {
                if (image_id < 0 || static_cast<size_t>(image_id) >= reconstruction.images.size()) {
                    continue;
                }
                for (const auto& point2d : reconstruction.images[static_cast<size_t>(image_id)].points2d) {
                    if (point2d.point3d >= 0) {
                        candidate_point_ids.insert(point2d.point3d);
                    }
                }
            }
        }
    }
    size_t filtered_observations = 0;
    for (const int point_id : candidate_point_ids) {
        if (point_id < 0 || static_cast<size_t>(point_id) >= reconstruction.points3d.size()) {
            continue;
        }
        auto& point3d = reconstruction.points3d[static_cast<size_t>(point_id)];
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        std::vector<SfMObservationId> observations_to_delete;
        observations_to_delete.reserve(point3d.track.size());
        for (const auto& obs_id : point3d.track) {
            if (!reconstruction.valid_observation(obs_id) ||
                !reconstruction.images[static_cast<size_t>(obs_id.image)].registered) {
                observations_to_delete.push_back(obs_id);
                continue;
            }
            const auto& camera = reconstruction.images[static_cast<size_t>(obs_id.image)].camera;
            if ((camera.R * point3d.xyz + camera.t).z() <= 0.0) {
                observations_to_delete.push_back(obs_id);
                continue;
            }
            const Eigen::Vector2d uv = reconstruction.images[static_cast<size_t>(obs_id.image)]
                                           .points2d[static_cast<size_t>(obs_id.point2d)]
                                           .xy;
            if (!std::isfinite(reprojection_error(point3d.xyz, camera, uv)) ||
                reprojection_error(point3d.xyz, camera, uv) > options.filter_max_reproj_error) {
                observations_to_delete.push_back(obs_id);
            }
        }

        if (observations_to_delete.size() >= point3d.track.size() - 1) {
            filtered_observations += point3d.track.size();
            reconstruction.delete_point3d(point_id);
            continue;
        }
        for (const auto& obs_id : observations_to_delete) {
            if (reconstruction.delete_observation(obs_id)) {
                ++filtered_observations;
            }
        }
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }

        std::vector<FeatureTrackObservation> observations;
        observations.reserve(point3d.track.size());
        for (const auto& obs_id : point3d.track) {
            if (reconstruction.valid_observation(obs_id) &&
                reconstruction.images[static_cast<size_t>(obs_id.image)].registered) {
                observations.push_back({obs_id.image,
                                        reconstruction.images[static_cast<size_t>(obs_id.image)]
                                            .points2d[static_cast<size_t>(obs_id.point2d)]
                                            .xy});
            }
        }
        point3d.reprojection_error = mean_reprojection_error(point3d.xyz, cameras, observations);
        if (!std::isfinite(point3d.reprojection_error) ||
            max_triangulation_angle_degrees(point3d.xyz, observations, cameras) <
                options.min_triangulation_angle_degrees) {
            filtered_observations += point3d.track.size();
            reconstruction.delete_point3d(point_id);
        }
    }
    return filtered_observations;
}

bool run_state_bundle_adjustment(SfMReconstructionState& reconstruction,
                                 const int anchor_i,
                                 const int anchor_j,
                                 const MultiviewCalibrationOptions& options,
                                 const bool use_robust_loss)
{
#ifndef TRADITIONAL_DIC_HAS_CERES
    (void)reconstruction;
    (void)anchor_i;
    (void)anchor_j;
    (void)options;
    (void)use_robust_loss;
    return false;
#else
    std::vector<CameraModel> cameras;
    std::vector<bool> registered;
    cameras.reserve(reconstruction.images.size());
    registered.reserve(reconstruction.images.size());
    for (const auto& image : reconstruction.images) {
        cameras.push_back(image.camera);
        registered.push_back(image.registered);
    }
    std::vector<SparsePoint3D> sparse_points = export_sparse_points_from_state(reconstruction);
    if (sparse_points.size() < 8) {
        return false;
    }
    const bool ok = run_global_bundle_adjustment(cameras,
                                                 sparse_points,
                                                 registered,
                                                 anchor_i,
                                                 anchor_j,
                                                 options,
                                                 {},
                                                 use_robust_loss);
    if (!ok) {
        return false;
    }
    import_bundle_result_to_state(cameras, sparse_points, reconstruction);
    return true;
#endif
}

bool run_state_local_bundle_adjustment(SfMReconstructionState& reconstruction,
                                       const int image_idx,
                                       const int anchor_i,
                                       const int anchor_j,
                                       const MultiviewCalibrationOptions& options,
                                       std::vector<int>* adjusted_images,
                                       const bool use_robust_loss)
{
#ifndef TRADITIONAL_DIC_HAS_CERES
    (void)reconstruction;
    (void)image_idx;
    (void)anchor_i;
    (void)anchor_j;
    (void)options;
    (void)adjusted_images;
    (void)use_robust_loss;
    return false;
#else
    if (image_idx < 0 || static_cast<size_t>(image_idx) >= reconstruction.images.size() ||
        !reconstruction.images[static_cast<size_t>(image_idx)].registered) {
        return false;
    }

    std::set<int> local_images;
    local_images.insert(image_idx);
    if (anchor_i >= 0 && static_cast<size_t>(anchor_i) < reconstruction.images.size() &&
        reconstruction.images[static_cast<size_t>(anchor_i)].registered) {
        local_images.insert(anchor_i);
    }
    if (anchor_j >= 0 && static_cast<size_t>(anchor_j) < reconstruction.images.size() &&
        reconstruction.images[static_cast<size_t>(anchor_j)].registered) {
        local_images.insert(anchor_j);
    }
    std::set<int> local_point_ids;
    std::map<int, int> covisible_image_counts;
    for (const auto& point2d : reconstruction.images[static_cast<size_t>(image_idx)].points2d) {
        if (point2d.point3d >= 0 &&
            static_cast<size_t>(point2d.point3d) < reconstruction.points3d.size() &&
            reconstruction.points3d[static_cast<size_t>(point2d.point3d)].valid) {
            local_point_ids.insert(point2d.point3d);
            for (const auto& obs : reconstruction.points3d[static_cast<size_t>(point2d.point3d)].track) {
                if (obs.image != image_idx && reconstruction.valid_observation(obs) &&
                    reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                    ++covisible_image_counts[obs.image];
                }
            }
        }
    }
    std::vector<std::pair<int, int>> covisible_images(covisible_image_counts.begin(), covisible_image_counts.end());
    std::sort(covisible_images.begin(), covisible_images.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    const size_t max_local_bundle_images = static_cast<size_t>(std::max(2, options.ba_local_num_images));
    for (const auto& [img, count] : covisible_images) {
        (void)count;
        if (local_images.size() >= max_local_bundle_images) {
            break;
        }
        local_images.insert(img);
    }
    // COLMAP's local bundle contains all points observed by the local image
    // set, not only points seen in the newly registered image. This keeps
    // short tracks and neighboring observations available to stabilize the
    // local camera pose before filtering.
    for (const int local_image_id : local_images) {
        if (local_image_id < 0 || static_cast<size_t>(local_image_id) >= reconstruction.images.size()) {
            continue;
        }
        for (const auto& point2d : reconstruction.images[static_cast<size_t>(local_image_id)].points2d) {
            if (point2d.point3d >= 0 && static_cast<size_t>(point2d.point3d) < reconstruction.points3d.size() &&
                reconstruction.points3d[static_cast<size_t>(point2d.point3d)].valid) {
                local_point_ids.insert(point2d.point3d);
            }
        }
    }
    if (local_images.size() < 2 || local_point_ids.size() < 8) {
        return false;
    }

    MultiviewCalibrationOptions ba_options = options;
    if (options.share_intrinsics) {
        size_t num_registered_images = 0;
        for (const auto& image : reconstruction.images) {
            if (image.registered) {
                ++num_registered_images;
            }
        }
        if (local_images.size() < num_registered_images) {
            ba_options.refine_focal_length = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_extra_params = false;
        }
    }

    std::vector<CameraModel> cameras;
    std::vector<bool> variable_images(reconstruction.images.size(), false);
    std::vector<bool> constant_images(reconstruction.images.size(), false);
    cameras.reserve(reconstruction.images.size());
    for (size_t i = 0; i < reconstruction.images.size(); ++i) {
        cameras.push_back(reconstruction.images[i].camera);
        variable_images[i] = local_images.count(static_cast<int>(i)) > 0 &&
                             reconstruction.images[i].registered;
    }

    std::vector<int> sparse_to_point_id;
    std::vector<SparsePoint3D> sparse_points;
    sparse_points.reserve(local_point_ids.size());
    for (const int point_id : local_point_ids) {
        const auto& point3d = reconstruction.points3d[static_cast<size_t>(point_id)];
        SparsePoint3D sparse;
        sparse.point = point3d.xyz;
        for (const auto& obs : point3d.track) {
            if (!reconstruction.valid_observation(obs) ||
                !reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                continue;
            }
            if (local_images.count(obs.image) == 0) {
                constant_images[static_cast<size_t>(obs.image)] = true;
            }
            sparse.observations.push_back({obs.image,
                                           reconstruction.images[static_cast<size_t>(obs.image)]
                                               .points2d[static_cast<size_t>(obs.point2d)]
                                               .xy});
        }
        if (sparse.observations.size() >= 2) {
            sparse.reprojection_error = mean_reprojection_error(sparse.point, cameras, sparse.observations);
            sparse_to_point_id.push_back(point_id);
            sparse_points.push_back(std::move(sparse));
        }
    }
    if (sparse_points.size() < 8) {
        return false;
    }

    int local_anchor_i = anchor_i;
    int local_anchor_j = anchor_j;
    if (local_anchor_i < 0 || local_anchor_j < 0 ||
        static_cast<size_t>(local_anchor_i) >= variable_images.size() ||
        static_cast<size_t>(local_anchor_j) >= variable_images.size() ||
        !variable_images[static_cast<size_t>(local_anchor_i)] ||
        !variable_images[static_cast<size_t>(local_anchor_j)]) {
        local_anchor_i = *local_images.begin();
        local_anchor_j = *std::next(local_images.begin());
    }
    const bool ok = run_global_bundle_adjustment(cameras,
                                                 sparse_points,
                                                 variable_images,
                                                 local_anchor_i,
                                                 local_anchor_j,
                                                 ba_options,
                                                 constant_images,
                                                 use_robust_loss);
    if (!ok) {
        return false;
    }
    for (const int img : local_images) {
        if (img >= 0 && static_cast<size_t>(img) < reconstruction.images.size()) {
            reconstruction.images[static_cast<size_t>(img)].camera = cameras[static_cast<size_t>(img)];
        }
    }
    if (adjusted_images != nullptr) {
        adjusted_images->assign(local_images.begin(), local_images.end());
    }
    for (size_t i = 0; i < sparse_points.size() && i < sparse_to_point_id.size(); ++i) {
        auto& point3d = reconstruction.points3d[static_cast<size_t>(sparse_to_point_id[i])];
        point3d.xyz = sparse_points[i].point;
        point3d.reprojection_error = sparse_points[i].reprojection_error;
    }
    return true;
#endif
}

size_t complete_and_merge_tracks(SfMIncrementalTriangulator& triangulator,
                                 const SfMIncrementalTriangulator::Options& tri_options)
{
    size_t changed = 0;
    changed += triangulator.complete_all_tracks(tri_options);
    changed += triangulator.merge_all_tracks(tri_options);
    return changed;
}

void run_iterative_local_refinement(SfMReconstructionState& reconstruction,
                                    SfMIncrementalTriangulator& triangulator,
                                    const SfMIncrementalTriangulator::Options& tri_options,
                                    const int image_idx,
                                    const int anchor_i,
                                    const int anchor_j,
                                    const MultiviewCalibrationOptions& options)
{
    constexpr int kMaxLocalRefinements = 2;
    constexpr double kMaxRefinementChange = 0.001;
    for (int iter = 0; iter < kMaxLocalRefinements; ++iter) {
        const size_t num_observations = count_state_observations(reconstruction);
        std::vector<int> adjusted_images;
        if (!run_state_local_bundle_adjustment(reconstruction,
                                               image_idx,
                                               anchor_i,
                                               anchor_j,
                                               options,
                                               &adjusted_images,
                                               iter == 0)) {
            break;
        }
        size_t changed = 0;
        changed += triangulator.merge_all_tracks(tri_options);
        changed += triangulator.complete_all_tracks(tri_options);
        changed += triangulator.complete_image(image_idx, tri_options);
        changed += filter_state_points(reconstruction, options, &adjusted_images, nullptr);
        if (num_observations == 0 ||
            static_cast<double>(changed) / static_cast<double>(num_observations) < kMaxRefinementChange) {
            break;
        }
    }
}

void run_iterative_global_refinement(SfMReconstructionState& reconstruction,
                                     SfMIncrementalTriangulator& triangulator,
                                     const SfMIncrementalTriangulator::Options& tri_options,
                                     const int anchor_i,
                                     const int anchor_j,
                                     const MultiviewCalibrationOptions& options)
{
    constexpr int kMaxGlobalRefinements = 2;
    constexpr double kMaxRefinementChange = 0.0005;
    complete_and_merge_tracks(triangulator, tri_options);
    for (int iter = 0; iter < kMaxGlobalRefinements; ++iter) {
        const size_t num_observations = count_state_observations(reconstruction);
        if (!run_state_bundle_adjustment(reconstruction,
                                         anchor_i,
                                         anchor_j,
                                         options,
                                         iter == 0)) {
            break;
        }
        size_t changed = complete_and_merge_tracks(triangulator, tri_options);
        changed += filter_state_points(reconstruction, options);
        if (num_observations == 0 ||
            static_cast<double>(changed) / static_cast<double>(num_observations) < kMaxRefinementChange) {
            break;
        }
    }
}

void refine_registered_poses_and_points(std::vector<Track>& tracks,
                                        const std::vector<ImageFeatures>& features,
                                        std::vector<CameraModel>& cameras,
                                        const std::vector<bool>& registered,
                                        const MultiviewCalibrationOptions& options,
                                        std::vector<SparsePoint3D>& points)
{
    const int first_registered = static_cast<int>(std::distance(registered.begin(), std::find(registered.begin(), registered.end(), true)));
    for (int iter = 0; iter < 3; ++iter) {
        for (int image_idx = 0; image_idx < static_cast<int>(cameras.size()); ++image_idx) {
            if (!registered[static_cast<size_t>(image_idx)] || image_idx == first_registered) {
                continue;
            }
            register_image_pnp(image_idx, tracks, features, points, cameras[static_cast<size_t>(image_idx)], options);
        }
        triangulate_registered_tracks(tracks, features, cameras, registered, options, points);
    }
}

#endif

} // namespace

MultiviewCalibrationResult calibrate_multiview_colmap_like(const std::vector<std::string>& image_paths,
                                                           const MultiviewCalibrationOptions& options)
{
    if (image_paths.size() < 2) {
        throw std::invalid_argument("Multiview calibration requires at least two images.");
    }
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw std::runtime_error("OpenCV is required for the simplified COLMAP-style multiview calibration.");
#else
    std::vector<ImageFeatures> features(image_paths.size());
    std::vector<CameraModel> cameras = options.initial_cameras;
    cameras.resize(image_paths.size());

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(options.max_features);
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to read SfM image: " + image_paths[i]);
        }
        features[i].image = image;
        sift->detectAndCompute(image, cv::noArray(), features[i].keypoints, features[i].descriptors);
        if (cameras[i].image_width == 0 || cameras[i].image_height == 0) {
            cameras[i] = make_initial_camera_with_options(image_paths[i], image.cols, image.rows, options);
        } else if (cameras[i].label.empty()) {
            cameras[i].label = image_paths[i];
        }
    }

    MultiviewCalibrationResult result;
    result.inlier_match_counts.assign(image_paths.size(), std::vector<int>(image_paths.size(), 0));

    std::vector<PairGeometry> pairs;
    pairs.reserve(image_paths.size() * image_paths.size());
    for (int i = 0; i < static_cast<int>(image_paths.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(image_paths.size()); ++j) {
            if (!pair_is_enabled(i, j, static_cast<int>(image_paths.size()), options)) {
                continue;
            }
            PairGeometry pair = estimate_sift_pair_geometry(i, j, features, cameras, options);
            result.inlier_match_counts[static_cast<size_t>(i)][static_cast<size_t>(j)] = pair.inliers;
            result.inlier_match_counts[static_cast<size_t>(j)][static_cast<size_t>(i)] = pair.inliers;
            if (pair.inliers >= options.min_inlier_matches) {
                pairs.push_back(std::move(pair));
            }
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error("No robust image pair was found for multiview calibration.");
    }

    SfMCorrespondenceGraph correspondence_graph = build_correspondence_graph(features, pairs);
    const std::vector<Track> tracks_for_validation = build_tracks_from_correspondence_graph(features, correspondence_graph);
    if (tracks_for_validation.empty()) {
        throw std::runtime_error("No feature tracks could be built for multiview calibration.");
    }

    const PairGeometry* initial_pair = nullptr;
    for (const auto& pair : pairs) {
        if (pair.i == options.initial_image1 && pair.j == options.initial_image2) {
            initial_pair = &pair;
            break;
        }
    }
    if (initial_pair == nullptr) {
        initial_pair = &*std::max_element(pairs.begin(), pairs.end(), [](const PairGeometry& a, const PairGeometry& b) {
            return a.inliers < b.inliers;
        });
    }

    std::vector<bool> registered(image_paths.size(), false);
    registered[static_cast<size_t>(initial_pair->i)] = true;
    registered[static_cast<size_t>(initial_pair->j)] = true;
    cameras[static_cast<size_t>(initial_pair->i)].R = Eigen::Matrix3d::Identity();
    cameras[static_cast<size_t>(initial_pair->i)].t = Eigen::Vector3d::Zero();
    cameras[static_cast<size_t>(initial_pair->j)].R = cv_to_eigen3x3(initial_pair->R);
    cameras[static_cast<size_t>(initial_pair->j)].t = cv_to_eigen3(initial_pair->t);
    SfMReconstructionState reconstruction_state = make_reconstruction_state(features, cameras, registered);
    SfMObservationManager observation_manager(reconstruction_state, correspondence_graph);
    SfMIncrementalTriangulator::Options tri_options;
    tri_options.max_transitivity = 1;
    tri_options.complete_max_transitivity = 5;
    tri_options.create_max_reproj_error = options.filter_max_reproj_error;
    tri_options.continue_max_reproj_error = options.filter_max_reproj_error;
    tri_options.merge_max_reproj_error = options.filter_max_reproj_error;
    tri_options.complete_max_reproj_error = options.filter_max_reproj_error;
    tri_options.ignore_two_view_tracks = options.ignore_two_view_tracks;
    SfMIncrementalTriangulator triangulator(correspondence_graph,
                                            features,
                                            reconstruction_state,
                                            observation_manager,
                                            options);

    triangulator.triangulate_image(initial_pair->i, tri_options);
    triangulator.triangulate_image(initial_pair->j, tri_options);
    if (options.refine_bundle) {
        run_iterative_global_refinement(reconstruction_state,
                                        triangulator,
                                        tri_options,
                                        initial_pair->i,
                                        initial_pair->j,
                                        options);
    } else {
        complete_and_merge_tracks(triangulator, tri_options);
        filter_state_points(reconstruction_state, options);
    }

    while (true) {
        int best_image = -1;
        size_t best_visible = 0;
        for (int image_idx = 0; image_idx < static_cast<int>(image_paths.size()); ++image_idx) {
            if (reconstruction_state.images[static_cast<size_t>(image_idx)].registered) {
                continue;
            }
            const size_t visible = observation_manager.num_visible_points3d(image_idx);
            if (visible > best_visible) {
                best_visible = visible;
                best_image = image_idx;
            }
        }
        if (best_image < 0) {
            break;
        }
        if (!register_image_pnp_from_state(best_image,
                                           correspondence_graph,
                                           reconstruction_state,
                                           observation_manager,
                                           cameras[static_cast<size_t>(best_image)],
                                           options)) {
            break;
        }
        registered[static_cast<size_t>(best_image)] = true;
        cameras[static_cast<size_t>(best_image)] = reconstruction_state.images[static_cast<size_t>(best_image)].camera;
        triangulator.triangulate_image(best_image, tri_options);
        triangulator.complete_image(best_image, tri_options);
        if (options.refine_bundle) {
            run_iterative_local_refinement(reconstruction_state,
                                           triangulator,
                                           tri_options,
                                           best_image,
                                           initial_pair->i,
                                           initial_pair->j,
                                           options);
            for (size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
                cameras[cam_idx] = reconstruction_state.images[cam_idx].camera;
            }
        } else {
            complete_and_merge_tracks(triangulator, tri_options);
            filter_state_points(reconstruction_state, options);
        }
    }

    if (options.refine_bundle) {
        run_iterative_global_refinement(reconstruction_state,
                                        triangulator,
                                        tri_options,
                                        initial_pair->i,
                                        initial_pair->j,
                                        options);
    }

    result.sparse_points = export_sparse_points_from_state(reconstruction_state);

    result.cameras.reserve(cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (reconstruction_state.images[i].registered) {
            result.cameras.push_back(reconstruction_state.images[i].camera);
        }
    }
    if (result.cameras.size() < 2) {
        throw std::runtime_error("Multiview calibration could not register a connected camera model.");
    }

    double error_sum = 0.0;
    for (const auto& point : result.sparse_points) {
        error_sum += point.reprojection_error;
    }
    if (!result.sparse_points.empty()) {
        result.mean_reprojection_error = error_sum / static_cast<double>(result.sparse_points.size());
    }
    return result;
#endif
}

MultiviewScaleResult estimate_multiview_chessboard_scale(
    const std::vector<CameraModel>& cameras,
    const std::vector<SparsePoint3D>& sparse_points,
    const std::vector<MultiviewScaleObservation>& observations,
    const MultiviewScaleOptions& options)
{
    validate_scale_options(options);
    if (cameras.size() < 2) {
        throw std::invalid_argument("Scale estimation requires at least two cameras.");
    }
    const int corner_count = options.board_rows * options.board_cols;
    std::vector<std::vector<FeatureTrackObservation>> tracks(static_cast<size_t>(corner_count));
    for (const auto& observation_set : observations) {
        if (observation_set.camera_index < 0 || observation_set.camera_index >= static_cast<int>(cameras.size())) {
            throw std::out_of_range("Scale observation camera index is out of range.");
        }
        if (static_cast<int>(observation_set.image_points.size()) != corner_count) {
            throw std::invalid_argument("Every scale observation must contain board_rows * board_cols image points.");
        }
        for (int i = 0; i < corner_count; ++i) {
            tracks[static_cast<size_t>(i)].push_back({observation_set.camera_index, observation_set.image_points[static_cast<size_t>(i)]});
        }
    }

    std::vector<Eigen::Vector3d> board_points(static_cast<size_t>(corner_count), Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()));
    std::vector<bool> valid(static_cast<size_t>(corner_count), false);
    int triangulated = 0;
    for (int i = 0; i < corner_count; ++i) {
        const auto& track = tracks[static_cast<size_t>(i)];
        if (static_cast<int>(track.size()) < std::max(2, options.min_common_corners > corner_count ? 2 : 2)) {
            continue;
        }
        try {
            Eigen::Vector3d point = triangulate_linear(cameras, track);
            if (mean_reprojection_error(point, cameras, track) <= options.max_reprojection_error) {
                board_points[static_cast<size_t>(i)] = point;
                valid[static_cast<size_t>(i)] = true;
                ++triangulated;
            }
        } catch (const std::exception&) {
        }
    }

    std::vector<double> edges;
    for (int r = 0; r < options.board_rows; ++r) {
        for (int c = 0; c < options.board_cols; ++c) {
            const int idx = r * options.board_cols + c;
            if (c + 1 < options.board_cols) {
                const int right = r * options.board_cols + c + 1;
                if (valid[static_cast<size_t>(idx)] && valid[static_cast<size_t>(right)]) {
                    edges.push_back((board_points[static_cast<size_t>(idx)] - board_points[static_cast<size_t>(right)]).norm());
                }
            }
            if (r + 1 < options.board_rows) {
                const int down = (r + 1) * options.board_cols + c;
                if (valid[static_cast<size_t>(idx)] && valid[static_cast<size_t>(down)]) {
                    edges.push_back((board_points[static_cast<size_t>(idx)] - board_points[static_cast<size_t>(down)]).norm());
                }
            }
        }
    }
    if (static_cast<int>(edges.size()) < options.min_common_corners) {
        throw std::runtime_error("Not enough valid chessboard edges for scale estimation.");
    }

    std::vector<double> trimmed = trim_values(edges, options.trim_fraction);
    const double sfm_square_mean = mean_value(trimmed);
    if (sfm_square_mean <= 0.0) {
        throw std::runtime_error("Estimated SfM square size is non-positive.");
    }
    const double sfm_square_median = median_value(trimmed);
    const double sfm_square_std = std_value(trimmed, sfm_square_mean);
    const double scale = options.square_size / sfm_square_mean;

    MultiviewScaleResult result;
    result.sfm_to_world_scale = scale;
    result.world_to_sfm_scale = 1.0 / scale;
    result.sfm_square_size_mean = sfm_square_mean;
    result.sfm_square_size_median = sfm_square_median;
    result.sfm_square_size_std = sfm_square_std;
    result.edge_cv = sfm_square_mean > 0.0 ? sfm_square_std / sfm_square_mean : 0.0;
    result.triangulated_corners = triangulated;
    result.valid_edges = static_cast<int>(edges.size());
    result.triangulated_board_points_sfm = std::move(board_points);
    result.edge_lengths_sfm = std::move(edges);

    result.scaled_cameras = cameras;
    for (auto& camera : result.scaled_cameras) {
        camera.t *= scale;
    }
    result.scaled_sparse_points = sparse_points;
    for (auto& point : result.scaled_sparse_points) {
        point.point *= scale;
    }
    return result;
}

std::vector<CameraModel> calibrate_multiview(int image_count)
{
    (void)image_count;
    throw std::runtime_error("Use calibrate_multiview_colmap_like with image paths.");
}

}  // namespace neurodic::calibration
