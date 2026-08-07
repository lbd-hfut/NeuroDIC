#include <neurodic/calibration/multiview_calibration.hpp>

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

#ifdef NEURODIC_HAS_CERES
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

// COLMAP CalculateAngularReprojectionError: the angle between the ray through
// the 3D point and the (undistorted) observation ray from the camera center.
double angular_reprojection_error(const Eigen::Vector3d& point,
                                  const CameraModel& camera,
                                  const Eigen::Vector2d& observation)
{
    const Eigen::Vector3d cam = camera.R * point + camera.t;
    if (cam.z() <= std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::Vector3d point_ray = cam.normalized();
    const double f = 0.5 * (camera.K(0, 0) + camera.K(1, 1));
    double x = (observation.x() - camera.K(0, 2)) / f;
    double y = (observation.y() - camera.K(1, 2)) / f;
    const double k1 = camera.distortion.empty() ? 0.0 : camera.distortion[0];
    for (int iter = 0; iter < 20; ++iter) {
        const double r2 = x * x + y * y;
        const double radial = 1.0 + k1 * r2;
        if (std::abs(radial) < std::numeric_limits<double>::epsilon()) {
            break;
        }
        x = (observation.x() - camera.K(0, 2)) / (f * radial);
        y = (observation.y() - camera.K(1, 2)) / (f * radial);
    }
    Eigen::Vector3d obs_ray(x, y, 1.0);
    if (obs_ray.norm() < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }
    obs_ray.normalize();
    const double cosine = std::clamp(point_ray.dot(obs_ray), -1.0, 1.0);
    return std::acos(cosine);
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
    // 0 = create, 2 = merge, 3 = retriangulate (complete never creates points).
    int creation_source = 0;
};

class SfMReconstructionState {
public:
    explicit SfMReconstructionState(const size_t num_images = 0) : images(num_images) {}

    int add_point3d(const Eigen::Vector3d& xyz,
                    const std::vector<SfMObservationId>& track,
                    const int creation_source = 0)
    {
        const int point3d_id = static_cast<int>(points3d.size());
        points3d.push_back({xyz, {}, 0.0, true, creation_source});
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

    int add_point3d(const Eigen::Vector3d& xyz,
                    const std::vector<SfMObservationId>& track,
                    const int creation_source = 0)
    {
        return reconstruction_.add_point3d(xyz, track, creation_source);
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
    if (options.matching_mode == "exhaustive") {
        // COLMAP match_exhaustive: every image pair is a candidate; the graph
        // is filtered afterwards by geometric verification and the inlier
        // threshold.
        return true;
    }
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

std::vector<cv::DMatch> sift_ratio_match(const cv::Mat& desc1,
                                         const cv::Mat& desc2,
                                         const double ratio,
                                         const bool bidirectional)
{
    auto ratio_test = [](const cv::Mat& d1, const cv::Mat& d2, const double r) {
        std::vector<cv::DMatch> good;
        if (d1.empty() || d2.empty()) {
            return good;
        }
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(d1, d2, knn, 2);
        for (const auto& pair : knn) {
            if (pair.size() == 2 && pair[0].distance < r * pair[1].distance) {
                good.push_back(pair[0]);
            }
        }
        return good;
    };

    std::vector<cv::DMatch> forward = ratio_test(desc1, desc2, ratio);
    if (!bidirectional || forward.empty()) {
        return forward;
    }
    // Mutually-consistent matching: keep only one-to-one matches that survive
    // the ratio test in both directions.
    std::vector<cv::DMatch> backward = ratio_test(desc2, desc1, ratio);
    std::unordered_set<int64_t> backward_keys;
    backward_keys.reserve(backward.size());
    for (const auto& match : backward) {
        backward_keys.insert((static_cast<int64_t>(match.trainIdx) << 32) |
                             static_cast<int64_t>(match.queryIdx));
    }
    std::vector<cv::DMatch> good;
    good.reserve(forward.size());
    for (const auto& match : forward) {
        const int64_t key = (static_cast<int64_t>(match.queryIdx) << 32) |
                            static_cast<int64_t>(match.trainIdx);
        if (backward_keys.count(key) > 0) {
            good.push_back(match);
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
                                        options.match_ratio,
                                        options.bidirectional_matching);
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

void set_camera_pose_from_cv(CameraModel& camera, const cv::Mat& rvec, const cv::Mat& tvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    camera.R = cv_to_eigen3x3(Rcv);
    camera.t = cv_to_eigen3(tvec);
}

#ifdef NEURODIC_HAS_CERES
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

// COLMAP IncrementalTriangulator::Create / Continue inlier criterion: the
// angular reprojection error of every observation against the triangulated
// point must stay below `max_angle_error_rad` (default 2 deg).
bool triangulate_sfm_observations_angular(const std::vector<ImageFeatures>& features,
                                          const std::vector<CameraModel>& cameras,
                                          const std::vector<SfMObservationId>& observations,
                                          const MultiviewCalibrationOptions& options,
                                          const double max_angle_error_rad,
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
        const double error = angular_reprojection_error(xyz,
                                                        cameras[static_cast<size_t>(obs.image_index)],
                                                        obs.point);
        if (std::isfinite(error) && error <= max_angle_error_rad) {
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
        // COLMAP IncrementalTriangulator angle thresholds (degrees).
        double create_max_angle_error_degrees = 2.0;
        double continue_max_reproj_error = 4.0;
        double continue_max_angle_error_degrees = 2.0;
        double merge_max_reproj_error = 4.0;
        double complete_max_reproj_error = 4.0;
        // Retriangulation options for under-reconstructed image pairs.
        double re_max_angle_error_degrees = 5.0;
        double re_min_ratio = 0.2;
        int re_max_trials = 1;
        // Additional pixel-error bound applied to newly created points during
        // retriangulation (0 = disabled). Retriangulation re-creates matches
        // that failed the initial triangulation; without a pixel bound those
        // marginal points (angular error up to 2 deg can be ~80 px at this
        // focal length) can drag the subsequent global bundle adjustment into
        // a degenerate basin (baseline collapse).
        double create_max_pixel_error = 0.0;
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

    // COLMAP IncrementalTriangulator::Retriangulate: re-triangulate
    // under-reconstructed image pairs (tri_ratio < re_min_ratio) to heal
    // tracks broken by pose/scale drift during incremental mapping.
    size_t retriangulate(const Options& options)
    {
        if (options.re_max_trials <= 0) {
            return 0;
        }
        Options re_options = options;
        re_options.continue_max_angle_error_degrees = options.re_max_angle_error_degrees;
        // Retriangulated points must also satisfy the pixel reprojection bound
        // that the regular filter enforces after bundle adjustment, so the
        // global BA never sees marginal near-collinear points.
        re_options.create_max_pixel_error = calibration_options_.filter_max_reproj_error;
        size_t changed = 0;

        for (int image1 = 0; image1 < static_cast<int>(reconstruction_.images.size()); ++image1) {
            if (!is_registered(image1)) {
                continue;
            }
            for (int image2 = image1 + 1; image2 < static_cast<int>(reconstruction_.images.size()); ++image2) {
                if (!is_registered(image2)) {
                    continue;
                }
                const int64_t pair_key = image_pair_key(image1, image2);

                // Under-reconstructed pair ratio: triangulated matches over
                // total matches between the pair.
                int num_total = 0;
                int num_tri = 0;
                for (int point_idx = 0; point_idx < static_cast<int>(graph_.num_points2d(image1)); ++point_idx) {
                    for (const auto& corr : graph_.find_correspondences(image1, point_idx)) {
                        if (corr.image != image2) {
                            continue;
                        }
                        ++num_total;
                        const int id1 = reconstruction_.images[static_cast<size_t>(image1)]
                                            .points2d[static_cast<size_t>(point_idx)]
                                            .point3d;
                        const int id2 = reconstruction_.images[static_cast<size_t>(corr.image)]
                                            .points2d[static_cast<size_t>(corr.point2d)]
                                            .point3d;
                        if (id1 >= 0 && id1 == id2 && valid_point3d(id1)) {
                            ++num_tri;
                        }
                    }
                }
                if (num_total == 0) {
                    continue;
                }
                const double tri_ratio = static_cast<double>(num_tri) / static_cast<double>(num_total);
                if (tri_ratio >= options.re_min_ratio) {
                    continue;
                }

                int& num_re_trials = re_num_trials_[pair_key];
                if (num_re_trials >= options.re_max_trials) {
                    continue;
                }
                ++num_re_trials;

                for (int point_idx = 0; point_idx < static_cast<int>(graph_.num_points2d(image1)); ++point_idx) {
                    for (const auto& corr : graph_.find_correspondences(image1, point_idx)) {
                        if (corr.image != image2) {
                            continue;
                        }
                        const int id1 = reconstruction_.images[static_cast<size_t>(image1)]
                                            .points2d[static_cast<size_t>(point_idx)]
                                            .point3d;
                        const int id2 = reconstruction_.images[static_cast<size_t>(corr.image)]
                                            .points2d[static_cast<size_t>(corr.point2d)]
                                            .point3d;
                        if (id1 >= 0 && id2 >= 0) {
                            // Both endpoints are already triangulated: never
                            // merge during retriangulation (COLMAP).
                            continue;
                        }
                        if (id1 >= 0) {
                            changed += continue_track({image2, corr.point2d, -1},
                                                      {{image1, point_idx, id1}},
                                                      re_options);
                        } else if (id2 >= 0) {
                            changed += continue_track({image1, point_idx, -1},
                                                      {{image2, corr.point2d, id2}},
                                                      re_options);
                        } else {
                            // COLMAP uses the regular Create options here
                            // (never the relaxed retriangulation thresholds);
                            // the extra pixel bound set on re_options keeps
                            // marginal near-collinear points out of the model.
                            changed += create({{image1, point_idx, -1}, {image2, corr.point2d, -1}},
                                              re_options,
                                              /*creation_source=*/3);
                        }
                    }
                }
            }
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

    size_t create(const std::vector<SfMCorrData>& corrs_data,
                  const Options& options,
                  const int creation_source = 0)
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
        // COLMAP Create() judges inliers by angular reprojection error.
        if (!triangulate_sfm_observations_angular(features_,
                                                  cameras,
                                                  create_observations,
                                                  calibration_options_,
                                                  options.create_max_angle_error_degrees * 3.14159265358979323846 / 180.0,
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
        // COLMAP's two-view check runs on the candidate list above (never on
        // the inlier track); inlier-filtered 2-view points are legal here and
        // are cleaned up later by the final general geometric filter
        // (final_min_track_length) once multi-view tracks are available.
        if (inlier_track.size() < 2 || has_duplicate_image(inlier_track)) {
            return 0;
        }
        if (options.create_max_pixel_error > 0.0) {
            const std::vector<CameraModel> cameras = cameras_from_state();
            std::vector<FeatureTrackObservation> feature_observations;
            feature_observations.reserve(inlier_track.size());
            for (const auto& obs : inlier_track) {
                feature_observations.push_back(feature_observation_from_id(features_, obs));
            }
            if (mean_reprojection_error(xyz, cameras, feature_observations) > options.create_max_pixel_error) {
                return 0;
            }
        }
        observation_manager_.add_point3d(xyz, inlier_track, creation_source);
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
            // COLMAP Continue() uses the angular reprojection error.
            const double error = angular_reprojection_error(reconstruction_.points3d[static_cast<size_t>(corr.point3d)].xyz,
                                                            reconstruction_.images[static_cast<size_t>(ref_corr.image)].camera,
                                                            ref_obs.point);
            if (error < best_error) {
                best_error = error;
                best_point3d = corr.point3d;
            }
        }
        if (best_point3d >= 0 &&
            best_error <= options.continue_max_angle_error_degrees * 3.14159265358979323846 / 180.0 &&
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
                    if (point3d_id >= 0 && static_cast<size_t>(point3d_id) < reconstruction_.points3d.size()) {
                        reconstruction_.points3d[static_cast<size_t>(point3d_id)].creation_source = 2;
                    }
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

    bool valid_point3d(const int point3d_id) const
    {
        return point3d_id >= 0 && static_cast<size_t>(point3d_id) < reconstruction_.points3d.size() &&
               reconstruction_.points3d[static_cast<size_t>(point3d_id)].valid;
    }

    static int64_t image_pair_key(const int image1, const int image2)
    {
        const int a = std::min(image1, image2);
        const int b = std::max(image1, image2);
        return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
    }

    const SfMCorrespondenceGraph& graph_;
    const std::vector<ImageFeatures>& features_;
    SfMReconstructionState& reconstruction_;
    SfMObservationManager& observation_manager_;
    const MultiviewCalibrationOptions& calibration_options_;
    // Number of retriangulation trials per image pair (COLMAP re_num_trials_).
    std::unordered_map<int64_t, int> re_num_trials_;
};

bool register_image_pnp_from_state(const int image_idx,
                                   const SfMCorrespondenceGraph& graph,
                                   SfMReconstructionState& reconstruction,
                                   SfMObservationManager& observation_manager,
                                   CameraModel& camera,
                                   const MultiviewCalibrationOptions& options,
                                   std::string* failure_reason = nullptr,
                                   int* num_correspondences = nullptr,
                                   int* num_inliers_out = nullptr)
{
    auto fail = [&failure_reason](const std::string& reason) {
        if (failure_reason != nullptr) {
            *failure_reason = reason;
        }
        return false;
    };
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
    if (num_correspondences != nullptr) {
        *num_correspondences = static_cast<int>(object_points.size());
    }

    if (object_points.size() < static_cast<size_t>(std::max(6, options.abs_pose_min_num_inliers))) {
        return fail("insufficient 2D-3D correspondences for PnP");
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
    if (num_inliers_out != nullptr) {
        *num_inliers_out = ok ? inliers.rows : 0;
    }
    if (!ok) {
        return fail("PnP RANSAC failed");
    }
    if (inliers.rows < options.abs_pose_min_num_inliers) {
        return fail("PnP inliers below abs_pose_min_num_inliers");
    }
    if (static_cast<double>(inliers.rows) / static_cast<double>(object_points.size()) < options.abs_pose_min_inlier_ratio) {
        return fail("PnP inlier ratio below abs_pose_min_inlier_ratio");
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

size_t count_registered_images(const SfMReconstructionState& reconstruction)
{
    size_t count = 0;
    for (const auto& image : reconstruction.images) {
        if (image.registered) {
            ++count;
        }
    }
    return count;
}

// COLMAP Reconstruction::Normalize(): translate the reconstruction so the
// point centroid moves to the origin and scale it so the bbox diagonal of the
// 3D points spans unit extent. Cameras follow the same similarity transform
// (R unchanged, t -> s * (t + R * centroid)), keeping every reprojection
// intact.
void normalize_reconstruction_state(SfMReconstructionState& reconstruction)
{
    std::vector<Eigen::Vector3d> points;
    points.reserve(reconstruction.points3d.size());
    for (const auto& point3d : reconstruction.points3d) {
        if (point3d.valid) {
            points.push_back(point3d.xyz);
        }
    }
    if (points.size() < 2) {
        return;
    }
    Eigen::Vector3d min_coord = points[0];
    Eigen::Vector3d max_coord = points[0];
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& point : points) {
        min_coord = min_coord.cwiseMin(point);
        max_coord = max_coord.cwiseMax(point);
        centroid += point;
    }
    centroid /= static_cast<double>(points.size());
    const double extent = (max_coord - min_coord).norm();
    if (extent < std::numeric_limits<double>::epsilon()) {
        return;
    }
    const double scale = 1.0 / extent;

    for (auto& point3d : reconstruction.points3d) {
        if (point3d.valid) {
            point3d.xyz = scale * (point3d.xyz - centroid);
        }
    }
    for (auto& image : reconstruction.images) {
        if (!image.registered) {
            continue;
        }
        image.camera.t = scale * (image.camera.t + image.camera.R * centroid);
    }
}

// COLMAP IncrementalMapperImpl::FindNextImages: rank unregistered images by
// visible 3D points. Images that never failed before form the first bucket,
// previously failed / filtered images form the second bucket.
std::vector<int> rank_next_images(const SfMReconstructionState& reconstruction,
                                  const SfMObservationManager& observation_manager,
                                  const std::vector<int>& num_reg_trials,
                                  const MultiviewCalibrationOptions& options)
{
    std::vector<std::pair<size_t, int>> first_bucket;
    std::vector<std::pair<size_t, int>> second_bucket;
    for (int image_idx = 0; image_idx < static_cast<int>(reconstruction.images.size()); ++image_idx) {
        if (reconstruction.images[static_cast<size_t>(image_idx)].registered) {
            continue;
        }
        const size_t visible = observation_manager.num_visible_points3d(image_idx);
        if (visible < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
            continue;
        }
        const int trials = image_idx < static_cast<int>(num_reg_trials.size()) ? num_reg_trials[static_cast<size_t>(image_idx)] : 0;
        if (trials >= options.max_reg_trials) {
            continue;
        }
        if (trials == 0) {
            first_bucket.emplace_back(visible, image_idx);
        } else {
            second_bucket.emplace_back(visible, image_idx);
        }
    }
    const auto sort_desc = [](const std::pair<size_t, int>& a, const std::pair<size_t, int>& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    };
    std::sort(first_bucket.begin(), first_bucket.end(), sort_desc);
    std::sort(second_bucket.begin(), second_bucket.end(), sort_desc);
    std::vector<int> ranked;
    ranked.reserve(first_bucket.size() + second_bucket.size());
    for (const auto& item : first_bucket) {
        ranked.push_back(item.second);
    }
    for (const auto& item : second_bucket) {
        ranked.push_back(item.second);
    }
    return ranked;
}

MultiviewStageStat make_stage_stat(const std::string& stage,
                                   const SfMReconstructionState& reconstruction)
{
    MultiviewStageStat stat;
    stat.stage = stage;
    std::vector<CameraModel> cameras;
    cameras.reserve(reconstruction.images.size());
    for (const auto& image : reconstruction.images) {
        cameras.push_back(image.camera);
        if (image.registered) {
            ++stat.num_registered_cameras;
        }
    }
    double error_sum = 0.0;
    double focal_sum = 0.0;
    int focal_count = 0;
    double pp_x_sum = 0.0;
    double pp_y_sum = 0.0;
    double k1_sum = 0.0;
    for (const auto& point3d : reconstruction.points3d) {
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        ++stat.num_points3d;
        stat.num_observations += point3d.track.size();
        std::vector<FeatureTrackObservation> observations;
        observations.reserve(point3d.track.size());
        for (const auto& obs : point3d.track) {
            if (!reconstruction.valid_observation(obs) ||
                !reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                continue;
            }
            observations.push_back({obs.image,
                                    reconstruction.images[static_cast<size_t>(obs.image)]
                                        .points2d[static_cast<size_t>(obs.point2d)]
                                        .xy});
        }
        if (observations.size() < 2) {
            continue;
        }
        error_sum += mean_reprojection_error(point3d.xyz, cameras, observations);
    }
    for (const auto& image : reconstruction.images) {
        if (!image.registered) {
            continue;
        }
        focal_sum += 0.5 * (image.camera.K(0, 0) + image.camera.K(1, 1));
        pp_x_sum += image.camera.K(0, 2);
        pp_y_sum += image.camera.K(1, 2);
        k1_sum += image.camera.distortion.empty() ? 0.0 : image.camera.distortion[0];
        ++focal_count;
    }
    if (stat.num_points3d > 0) {
        stat.mean_reprojection_error = error_sum / static_cast<double>(stat.num_points3d);
    }
    if (focal_count > 0) {
        stat.focal_length = focal_sum / static_cast<double>(focal_count);
        stat.principal_point_x = pp_x_sum / static_cast<double>(focal_count);
        stat.principal_point_y = pp_y_sum / static_cast<double>(focal_count);
        stat.distortion_k1 = k1_sum / static_cast<double>(focal_count);
    }
    return stat;
}

// COLMAP IncrementalMapperImpl::EstimateInitialTwoViewGeometry checks applied
// to an already-verified geometric pair: the relative translation must not be
// dominated by the forward component and the median triangulation angle of the
// inlier matches must exceed init_min_tri_angle.
bool initial_pair_geometry_ok(const PairGeometry& pair,
                              const std::vector<ImageFeatures>& features,
                              const std::vector<CameraModel>& cameras,
                              const MultiviewCalibrationOptions& options,
                              const double min_tri_angle_degrees,
                              std::string* reason)
{
    const Eigen::Vector3d translation = cv_to_eigen3(pair.t);
    const double translation_norm = translation.norm();
    if (translation_norm < std::numeric_limits<double>::epsilon()) {
        if (reason != nullptr) {
            *reason = "zero relative translation";
        }
        return false;
    }
    if (std::abs(translation.z()) / translation_norm >= options.init_max_forward_motion) {
        if (reason != nullptr) {
            *reason = "forward-motion dominated translation (init_max_forward_motion)";
        }
        return false;
    }

    CameraModel cam_i = cameras[static_cast<size_t>(pair.i)];
    CameraModel cam_j = cameras[static_cast<size_t>(pair.j)];
    cam_i.R = Eigen::Matrix3d::Identity();
    cam_i.t = Eigen::Vector3d::Zero();
    cam_j.R = cv_to_eigen3x3(pair.R);
    cam_j.t = cv_to_eigen3(pair.t);

    std::vector<double> angles;
    angles.reserve(pair.matches.size());
    for (size_t k = 0; k < pair.matches.size(); ++k) {
        if (k >= pair.inlier_mask.size() || pair.inlier_mask[k] == 0) {
            continue;
        }
        const auto& match = pair.matches[k];
        const auto pi = features[static_cast<size_t>(pair.i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const auto pj = features[static_cast<size_t>(pair.j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        Eigen::Vector3d xyz;
        try {
            // triangulate_linear() indexes its camera vector by the observation
            // image_index, so the two cameras must be passed as indices 0/1.
            xyz = triangulate_linear({cam_i, cam_j},
                                     {{0, {pi.x, pi.y}}, {1, {pj.x, pj.y}}});
        } catch (const std::exception&) {
            continue;
        }
        const Eigen::Vector3d xi = cam_i.R * xyz + cam_i.t;
        const Eigen::Vector3d xj = cam_j.R * xyz + cam_j.t;
        if (xi.z() <= 0.0 || xj.z() <= 0.0) {
            continue;
        }
        angles.push_back(triangulation_angle_degrees(xyz, cam_i, cam_j));
    }
    if (angles.empty()) {
        if (reason != nullptr) {
            *reason = "no positive-depth triangulation for the pair";
        }
        return false;
    }
    const double median_angle = median_value(angles);
    if (median_angle < min_tri_angle_degrees) {
        if (reason != nullptr) {
            *reason = "insufficient median triangulation angle";
        }
        return false;
    }
    return true;
}

// COLMAP IncrementalMapperImpl::FindFirstInitialImage / FindSecondInitialImage
// candidate ordering: first images by total correspondences, second images by
// the number of correspondences to the first image.
std::vector<std::pair<int, int>> rank_initial_pairs(const std::vector<PairGeometry>& pairs,
                                                    const MultiviewCalibrationOptions& options,
                                                    const int min_inlier_matches)
{
    std::unordered_map<int, size_t> total_corrs;
    std::unordered_map<int, std::vector<std::pair<int, int>>> second_images;  // first -> {(second, inliers)}
    for (const auto& pair : pairs) {
        if (pair.inliers < min_inlier_matches) {
            continue;
        }
        total_corrs[static_cast<size_t>(pair.i)] += static_cast<size_t>(pair.inliers);
        total_corrs[static_cast<size_t>(pair.j)] += static_cast<size_t>(pair.inliers);
        second_images[pair.i].emplace_back(pair.j, pair.inliers);
        second_images[pair.j].emplace_back(pair.i, pair.inliers);
    }
    std::vector<int> first_images;
    first_images.reserve(total_corrs.size());
    for (const auto& item : total_corrs) {
        first_images.push_back(item.first);
    }
    std::sort(first_images.begin(), first_images.end(), [&](const int a, const int b) {
        if (total_corrs[static_cast<size_t>(a)] != total_corrs[static_cast<size_t>(b)]) {
            return total_corrs[static_cast<size_t>(a)] > total_corrs[static_cast<size_t>(b)];
        }
        return a < b;
    });
    std::vector<std::pair<int, int>> ranked;
    if (options.initial_image1 >= 0 && options.initial_image2 >= 0 &&
        options.initial_image1 != options.initial_image2) {
        ranked.emplace_back(options.initial_image1, options.initial_image2);
    }
    for (const int first : first_images) {
        auto& seconds = second_images[first];
        std::sort(seconds.begin(), seconds.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }
            return a.first < b.first;
        });
        for (const auto& [second, inliers] : seconds) {
            (void)inliers;
            ranked.emplace_back(first, second);
        }
    }
    return ranked;
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

// Collect traceable per-point diagnostics for the current state (valid points
// with >= 2 observations). Returns diagnostics indexed by point3d id; entries
// for invalid/removed points are left at their defaults.
std::vector<SparsePointDiagnostic> collect_point_diagnostics(const SfMReconstructionState& reconstruction)
{
    std::vector<SparsePointDiagnostic> diagnostics(reconstruction.points3d.size());
    std::vector<CameraModel> cameras;
    cameras.reserve(reconstruction.images.size());
    for (const auto& image : reconstruction.images) {
        cameras.push_back(image.camera);
    }
    for (size_t pid = 0; pid < reconstruction.points3d.size(); ++pid) {
        const auto& point3d = reconstruction.points3d[pid];
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        SparsePointDiagnostic& diag = diagnostics[pid];
        diag.point_id = static_cast<int>(pid);
        diag.track_length = static_cast<int>(point3d.track.size());
        switch (point3d.creation_source) {
            case 0: diag.creation_source = "create"; break;
            case 2: diag.creation_source = "merge"; break;
            case 3: diag.creation_source = "retriangulate"; break;
            default: diag.creation_source = "unknown"; break;
        }
        std::vector<FeatureTrackObservation> observations;
        observations.reserve(point3d.track.size());
        std::vector<double> depths;
        depths.reserve(point3d.track.size());
        for (const auto& obs : point3d.track) {
            if (!reconstruction.valid_observation(obs) ||
                !reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                diag.all_positive_depth = false;
                continue;
            }
            const auto& camera = reconstruction.images[static_cast<size_t>(obs.image)].camera;
            diag.images.push_back(obs.image);
            const Eigen::Vector3d pc = camera.R * point3d.xyz + camera.t;
            if (pc.z() <= 0.0) {
                diag.all_positive_depth = false;
            }
            depths.push_back(pc.z());
            const Eigen::Vector2d uv = reconstruction.images[static_cast<size_t>(obs.image)]
                                           .points2d[static_cast<size_t>(obs.point2d)]
                                           .xy;
            diag.per_observation_errors.push_back(reprojection_error(point3d.xyz, camera, uv));
            observations.push_back({obs.image, uv});
        }
        if (observations.size() < 2) {
            continue;
        }
        std::vector<double> angles;
        for (size_t i = 0; i < observations.size(); ++i) {
            for (size_t j = i + 1; j < observations.size(); ++j) {
                angles.push_back(triangulation_angle_degrees(point3d.xyz,
                                                             cameras[static_cast<size_t>(observations[i].image_index)],
                                                             cameras[static_cast<size_t>(observations[j].image_index)]));
            }
        }
        if (!angles.empty()) {
            std::sort(angles.begin(), angles.end());
            diag.max_triangulation_angle_degrees = angles.back();
            diag.median_triangulation_angle_degrees =
                angles[angles.size() / 2];  // upper median for even sizes
        }
        diag.rms_after_final_ba = mean_reprojection_error(point3d.xyz, cameras, observations);
        diag.xyz_after_final_ba = point3d.xyz;
        // Depth-consistency ratio (max/min), only meaningful for >= 3 views.
        if (depths.size() >= 3 && depths.front() > 0.0) {
            const double dmin = *std::min_element(depths.begin(), depths.end());
            const double dmax = *std::max_element(depths.begin(), depths.end());
            diag.max_depth_ratio = dmax / dmin;
        }
    }
    return diagnostics;
}

// General final geometric filter (never radius/ground-truth based):
//  1. minimum track length;
//  2. per-observation reprojection error <= filter_max_reproj_error;
//  3. positive depth in every observation;
//  4. depth consistency for multi-view tracks (max/min <= final_max_depth_ratio);
//  5. min triangulation angle >= min_triangulation_angle_degrees.
// Returns the number of removed observations.
size_t final_filter_state_points(SfMReconstructionState& reconstruction,
                                 const MultiviewCalibrationOptions& options)
{
    const int min_track_length = std::max(2, options.final_min_track_length);
    const double max_depth_ratio = std::max(1.0, options.final_max_depth_ratio);
    std::vector<CameraModel> cameras;
    cameras.reserve(reconstruction.images.size());
    for (const auto& image : reconstruction.images) {
        cameras.push_back(image.camera);
    }
    size_t filtered_observations = 0;
    for (int point_id = 0; point_id < static_cast<int>(reconstruction.points3d.size()); ++point_id) {
        auto& point3d = reconstruction.points3d[static_cast<size_t>(point_id)];
        if (!point3d.valid || point3d.track.size() < 2) {
            continue;
        }
        // 1) minimum track length
        if (static_cast<int>(point3d.track.size()) < min_track_length) {
            filtered_observations += point3d.track.size();
            reconstruction.delete_point3d(point_id);
            continue;
        }
        std::vector<FeatureTrackObservation> observations;
        observations.reserve(point3d.track.size());
        std::vector<double> depths;
        depths.reserve(point3d.track.size());
        bool bad = false;
        // 2) per-observation error + 3) positive depth
        for (const auto& obs : point3d.track) {
            if (!reconstruction.valid_observation(obs) ||
                !reconstruction.images[static_cast<size_t>(obs.image)].registered) {
                bad = true;
                break;
            }
            const auto& camera = reconstruction.images[static_cast<size_t>(obs.image)].camera;
            const Eigen::Vector3d pc = camera.R * point3d.xyz + camera.t;
            if (pc.z() <= 0.0) {
                bad = true;
                break;
            }
            depths.push_back(pc.z());
            const Eigen::Vector2d uv = reconstruction.images[static_cast<size_t>(obs.image)]
                                           .points2d[static_cast<size_t>(obs.point2d)]
                                           .xy;
            const double error = reprojection_error(point3d.xyz, camera, uv);
            if (!std::isfinite(error) || error > options.filter_max_reproj_error) {
                bad = true;
                break;
            }
            observations.push_back({obs.image, uv});
        }
        if (bad) {
            filtered_observations += point3d.track.size();
            reconstruction.delete_point3d(point_id);
            continue;
        }
        // 4) depth consistency (multi-view tracks only)
        if (depths.size() >= 3) {
            const double dmin = *std::min_element(depths.begin(), depths.end());
            const double dmax = *std::max_element(depths.begin(), depths.end());
            if (dmin <= 0.0 || dmax / dmin > max_depth_ratio) {
                filtered_observations += point3d.track.size();
                reconstruction.delete_point3d(point_id);
                continue;
            }
        }
        // 5) min triangulation angle
        if (max_triangulation_angle_degrees(point3d.xyz, observations, cameras) <
            options.min_triangulation_angle_degrees) {
            filtered_observations += point3d.track.size();
            reconstruction.delete_point3d(point_id);
            continue;
        }
        point3d.reprojection_error = mean_reprojection_error(point3d.xyz, cameras, observations);
    }
    return filtered_observations;
}

bool run_state_bundle_adjustment(SfMReconstructionState& reconstruction,
                                 const int anchor_i,
                                 const int anchor_j,
                                 const MultiviewCalibrationOptions& options,
                                 const bool use_robust_loss)
{
#ifndef NEURODIC_HAS_CERES
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
#ifndef NEURODIC_HAS_CERES
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
    const int max_refinements = std::max(1, options.ba_local_max_refinements);
    const double max_refinement_change = std::max(0.0, options.ba_local_max_refinement_change);
    for (int iter = 0; iter < max_refinements; ++iter) {
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
        // COLMAP AdjustLocalBundle ordering: merge and complete the modified
        // tracks after refinement, complete the new image, then filter.
        size_t changed = 0;
        changed += triangulator.merge_all_tracks(tri_options);
        changed += triangulator.complete_all_tracks(tri_options);
        changed += triangulator.complete_image(image_idx, tri_options);
        changed += filter_state_points(reconstruction, options, &adjusted_images, nullptr);
        if (num_observations == 0 ||
            static_cast<double>(changed) / static_cast<double>(num_observations) < max_refinement_change) {
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
    const int max_refinements = std::max(1, options.ba_global_max_refinements);
    const double max_refinement_change = std::max(0.0, options.ba_global_max_refinement_change);
    // COLMAP IterativeGlobalRefinement: complete+merge tracks, retriangulate
    // under-reconstructed pairs, then alternate global BA (TRIVIAL loss, like
    // COLMAP's GlobalBundleAdjustment) with complete+merge+filter until the
    // relative change is small.
    complete_and_merge_tracks(triangulator, tri_options);
    triangulator.retriangulate(tri_options);
    for (int iter = 0; iter < max_refinements; ++iter) {
        const size_t num_observations = count_state_observations(reconstruction);
        if (!run_state_bundle_adjustment(reconstruction,
                                         anchor_i,
                                         anchor_j,
                                         options,
                                         false)) {
            break;
        }
        if (options.normalize_reconstruction) {
            normalize_reconstruction_state(reconstruction);
        }
        size_t changed = complete_and_merge_tracks(triangulator, tri_options);
        changed += filter_state_points(reconstruction, options);
        if (num_observations == 0 ||
            static_cast<double>(changed) / static_cast<double>(num_observations) < max_refinement_change) {
            break;
        }
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

    cv::Ptr<cv::SIFT> sift =
        cv::SIFT::create(options.max_features,
                         /*nOctaveLayers=*/3,
                         options.sift_contrast_threshold,
                         /*edgeThreshold=*/10.0,
                         /*sigma=*/1.6);
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to read SfM image: " + image_paths[i]);
        }
        features[i].image = image;
        sift->detectAndCompute(image, cv::noArray(), features[i].keypoints, features[i].descriptors);
        if (options.root_sift && !features[i].descriptors.empty()) {
            // RootSIFT: L1-normalize each descriptor, then element-wise sqrt.
            for (int r = 0; r < features[i].descriptors.rows; ++r) {
                float sum = 0.0f;
                float* row_ptr = features[i].descriptors.ptr<float>(r);
                for (int c = 0; c < features[i].descriptors.cols; ++c) {
                    sum += row_ptr[c];
                }
                if (sum > 0.0f) {
                    for (int c = 0; c < features[i].descriptors.cols; ++c) {
                        row_ptr[c] = std::sqrt(row_ptr[c] / sum);
                    }
                }
            }
        }
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
    result.pipeline_log.push_back("Matching: " + std::to_string(pairs.size()) + " geometrically verified pairs kept " +
                                  "(mode=" + options.matching_mode + ")");

    SfMCorrespondenceGraph correspondence_graph = build_correspondence_graph(features, pairs);
    const std::vector<Track> tracks_for_validation = build_tracks_from_correspondence_graph(features, correspondence_graph);
    if (tracks_for_validation.empty()) {
        throw std::runtime_error("No feature tracks could be built for multiview calibration.");
    }

    const SfMIncrementalTriangulator::Options tri_options = [&]() {
        SfMIncrementalTriangulator::Options tri;
        tri.max_transitivity = 1;
        tri.complete_max_transitivity = 5;
        tri.create_max_reproj_error = options.filter_max_reproj_error;
        tri.create_max_angle_error_degrees = options.create_max_angle_error_degrees;
        tri.continue_max_reproj_error = options.filter_max_reproj_error;
        tri.continue_max_angle_error_degrees = options.continue_max_angle_error_degrees;
        tri.merge_max_reproj_error = options.filter_max_reproj_error;
        tri.complete_max_reproj_error = options.filter_max_reproj_error;
        tri.re_max_angle_error_degrees = options.re_max_angle_error_degrees;
        tri.re_min_ratio = options.re_min_ratio;
        tri.re_max_trials = options.re_max_trials;
        tri.ignore_two_view_tracks = options.ignore_two_view_tracks;
        return tri;
    }();

    //////////////////////////////////////////////////////////////////////////
    // Initial image pair: COLMAP FindInitialImagePair + InitializeReconstruction
    // with outer init_num_trials relaxation (init_min_num_inliers /= 2, then
    // init_min_tri_angle /= 2). Every candidate pair is tried at most once per
    // relaxation round; the first pair whose two-view geometry passes the
    // checks and whose triangulation survives the initial global BA / filter
    // seeds the model.
    //////////////////////////////////////////////////////////////////////////

    std::unordered_set<int64_t> tried_initial_pairs;
    int min_init_inliers = options.min_inlier_matches;
    double min_init_tri_angle = options.init_min_tri_angle_degrees;

    SfMReconstructionState reconstruction_state;
    int initial_pair_i = -1;
    int initial_pair_j = -1;
    bool initialized = false;
    int relaxation_round = 0;

    for (int trial = 0; trial < std::max(1, options.init_num_trials) && !initialized; ++trial) {
        const std::vector<std::pair<int, int>> ranked_pairs =
            rank_initial_pairs(pairs, options, min_init_inliers);
        bool tried_any = false;
        for (const auto& [image1, image2] : ranked_pairs) {
            const int64_t pair_key = (static_cast<int64_t>(std::min(image1, image2)) << 32) |
                                     static_cast<uint32_t>(std::max(image1, image2));
            if (!tried_initial_pairs.insert(pair_key).second) {
                continue;
            }
            tried_any = true;

            const PairGeometry* pair_ptr = nullptr;
            for (const auto& pair : pairs) {
                if ((pair.i == image1 && pair.j == image2) || (pair.i == image2 && pair.j == image1)) {
                    pair_ptr = &pair;
                    break;
                }
            }
            if (pair_ptr == nullptr) {
                continue;
            }

            std::string geometry_reason;
            if (!initial_pair_geometry_ok(*pair_ptr, features, cameras, options, min_init_tri_angle, &geometry_reason)) {
                result.pipeline_log.push_back("Init pair (" + std::to_string(image1) + "," + std::to_string(image2) +
                                              "): rejected, " + geometry_reason);
                continue;
            }

            // Build a fresh two-view model and check sustainability through
            // the initial global BA + filter.
            std::vector<bool> registered(image_paths.size(), false);
            registered[static_cast<size_t>(image1)] = true;
            registered[static_cast<size_t>(image2)] = true;
            SfMReconstructionState trial_state = make_reconstruction_state(features, cameras, registered);
            trial_state.images[static_cast<size_t>(image1)].camera.R = Eigen::Matrix3d::Identity();
            trial_state.images[static_cast<size_t>(image1)].camera.t = Eigen::Vector3d::Zero();
            trial_state.images[static_cast<size_t>(image2)].camera.R = cv_to_eigen3x3(pair_ptr->R);
            trial_state.images[static_cast<size_t>(image2)].camera.t = cv_to_eigen3(pair_ptr->t);
            SfMObservationManager trial_observation_manager(trial_state, correspondence_graph);
            SfMIncrementalTriangulator trial_triangulator(correspondence_graph,
                                                          features,
                                                          trial_state,
                                                          trial_observation_manager,
                                                          options);

            trial_triangulator.triangulate_image(image1, tri_options);
            trial_triangulator.triangulate_image(image2, tri_options);
            if (options.refine_bundle) {
                run_state_bundle_adjustment(trial_state, image1, image2, options, false);
            } else {
                complete_and_merge_tracks(trial_triangulator, tri_options);
            }
            if (options.normalize_reconstruction) {
                normalize_reconstruction_state(trial_state);
            }
            filter_state_points(trial_state, options);

            int trial_num_points = 0;
            for (const auto& point3d : trial_state.points3d) {
                if (point3d.valid && point3d.track.size() >= 2) {
                    ++trial_num_points;
                }
            }
            const bool sustainable = trial_state.images[static_cast<size_t>(image1)].registered &&
                                     trial_state.images[static_cast<size_t>(image2)].registered &&
                                     trial_num_points >= std::max(2, options.abs_pose_min_num_inliers);
            if (sustainable) {
                // COLMAP ReconstructSubModel survival check: a seed pair must
                // also support registering at least one further camera without
                // collapsing the model. Noisy wide-baseline pairs (e.g. (8,6)
                // on CylinderDIC) pass the two-view checks but lose nearly all
                // points during the first incremental registration; they are
                // rejected here and the next candidate is tried.
                std::vector<int> trial_reg_trials(image_paths.size(), 0);
                const std::vector<int> trial_candidates =
                    rank_next_images(trial_state, trial_observation_manager, trial_reg_trials, options);
                bool trial_registered = false;
                int trial_first_image = -1;
                for (const int image_idx : trial_candidates) {
                    ++trial_reg_trials[static_cast<size_t>(image_idx)];
                    std::string failure_reason;
                    int num_corrs = 0;
                    int num_inl = 0;
                    const bool ok = register_image_pnp_from_state(image_idx,
                                                                  correspondence_graph,
                                                                  trial_state,
                                                                  trial_observation_manager,
                                                                  trial_state.images[static_cast<size_t>(image_idx)].camera,
                                                                  options,
                                                                  &failure_reason,
                                                                  &num_corrs,
                                                                  &num_inl);
                    MultiviewRegistrationAttempt attempt;
                    attempt.image_index = image_idx;
                    attempt.num_visible_points =
                        static_cast<int>(trial_observation_manager.num_visible_points3d(image_idx));
                    attempt.num_pnp_correspondences = num_corrs;
                    attempt.num_pnp_inliers = num_inl;
                    attempt.success = ok;
                    attempt.reason = ok ? "PnP registration succeeded (init survival trial)" : failure_reason;
                    result.registration_attempts.push_back(std::move(attempt));
                    if (ok) {
                        trial_registered = true;
                        trial_first_image = image_idx;
                        break;
                    }
                }
                bool survives = false;
                int surviving_points = 0;
                if (trial_registered) {
                    trial_triangulator.triangulate_image(trial_first_image, tri_options);
                    trial_triangulator.complete_image(trial_first_image, tri_options);
                    if (options.refine_bundle) {
                        run_iterative_local_refinement(trial_state,
                                                       trial_triangulator,
                                                       tri_options,
                                                       trial_first_image,
                                                       image1,
                                                       image2,
                                                       options);
                    } else {
                        complete_and_merge_tracks(trial_triangulator, tri_options);
                        filter_state_points(trial_state, options);
                    }
                    for (const auto& point3d : trial_state.points3d) {
                        if (point3d.valid && point3d.track.size() >= 2) {
                            ++surviving_points;
                        }
                    }
                    survives = trial_state.images[static_cast<size_t>(image1)].registered &&
                               trial_state.images[static_cast<size_t>(image2)].registered &&
                               trial_state.images[static_cast<size_t>(trial_first_image)].registered &&
                               surviving_points >= std::max(2, options.abs_pose_min_num_inliers);
                }
                if (survives) {
                    reconstruction_state = std::move(trial_state);
                    initial_pair_i = image1;
                    initial_pair_j = image2;
                    initialized = true;
                    result.pipeline_log.push_back("Init pair (" + std::to_string(image1) + "," + std::to_string(image2) +
                                                  "): accepted, " + std::to_string(trial_num_points) +
                                                  " points after initial BA/filter, " +
                                                  std::to_string(surviving_points) +
                                                  " points after first incremental registration");
                    break;
                }
                result.pipeline_log.push_back("Init pair (" + std::to_string(image1) + "," + std::to_string(image2) +
                                              "): rejected, first incremental registration leaves " +
                                              std::to_string(surviving_points) + " points (< " +
                                              std::to_string(std::max(2, options.abs_pose_min_num_inliers)) + ")");
            } else {
                result.pipeline_log.push_back("Init pair (" + std::to_string(image1) + "," + std::to_string(image2) +
                                              "): rejected after initial BA/filter (" + std::to_string(trial_num_points) +
                                              " points)");
            }
        }

        if (!initialized) {
            if (!tried_any && relaxation_round == 0) {
                // All candidates were exhausted or the trial budget expired:
                // relax exactly like COLMAP's outer init loop.
                relaxation_round = 1;
                min_init_inliers = std::max(2, min_init_inliers / 2);
                tried_initial_pairs.clear();
                result.pipeline_log.push_back("Relaxing init_min_num_inliers to " + std::to_string(min_init_inliers));
            } else if (relaxation_round == 1) {
                relaxation_round = 2;
                min_init_tri_angle = std::max(0.5, min_init_tri_angle / 2.0);
                tried_initial_pairs.clear();
                result.pipeline_log.push_back("Relaxing init_min_tri_angle to " + std::to_string(min_init_tri_angle));
            }
        }
    }
    if (!initialized) {
        throw std::runtime_error("No good initial image pair found for multiview calibration.");
    }

    SfMObservationManager observation_manager(reconstruction_state, correspondence_graph);
    SfMIncrementalTriangulator triangulator(correspondence_graph,
                                            features,
                                            reconstruction_state,
                                            observation_manager,
                                            options);
    result.stage_stats.push_back(make_stage_stat("initial_pair", reconstruction_state));

    if (options.refine_bundle) {
        // COLMAP runs exactly one AdjustGlobalBundle on the initial pair
        // (already performed inside the trial above) and starts registering
        // images right away. Iterative global refinement is deferred until at
        // least a third image registers: repeated bundle adjustment of a
        // two-camera model is degenerate (the baseline can collapse while
        // reprojections stay intact), which is why COLMAP's first
        // IterativeGlobalRefinement only fires once CheckRunGlobalRefinement
        // sees >= 1.1x the initial frame count.
    } else {
        complete_and_merge_tracks(triangulator, tri_options);
        filter_state_points(reconstruction_state, options);
        result.stage_stats.push_back(make_stage_stat("initial_triangulation", reconstruction_state));
    }

    //////////////////////////////////////////////////////////////////////////
    // Incremental mapping: COLMAP ReconstructSubModel loop. Register the
    // highest-ranked candidate; on PnP failure keep trying the remaining
    // candidates and record the reason. If a whole round fails, run one final
    // global refinement and retry once (COLMAP's prev/reg_next_success logic).
    //////////////////////////////////////////////////////////////////////////

    std::vector<int> num_reg_trials(image_paths.size(), 0);
    bool prev_reg_next_success = true;
    bool reg_next_success = true;
    size_t ba_prev_num_reg_frames = count_registered_images(reconstruction_state);
    size_t ba_prev_num_points = reconstruction_state.points3d.size();
    do {
        prev_reg_next_success = reg_next_success;
        reg_next_success = false;
        int next_image = -1;

        const std::vector<int> candidates =
            rank_next_images(reconstruction_state, observation_manager, num_reg_trials, options);
        for (const int image_idx : candidates) {
            ++num_reg_trials[static_cast<size_t>(image_idx)];
            const size_t visible = observation_manager.num_visible_points3d(image_idx);
            MultiviewRegistrationAttempt attempt;
            attempt.image_index = image_idx;
            attempt.num_visible_points = static_cast<int>(visible);
            std::string failure_reason;
            int num_corrs = 0;
            int num_pnp_inliers = 0;
            const bool ok = register_image_pnp_from_state(image_idx,
                                                          correspondence_graph,
                                                          reconstruction_state,
                                                          observation_manager,
                                                          reconstruction_state.images[static_cast<size_t>(image_idx)].camera,
                                                          options,
                                                          &failure_reason,
                                                          &num_corrs,
                                                          &num_pnp_inliers);
            attempt.num_pnp_correspondences = num_corrs;
            attempt.num_pnp_inliers = num_pnp_inliers;
            attempt.success = ok;
            attempt.reason = ok ? "PnP registration succeeded" : failure_reason;
            result.registration_attempts.push_back(std::move(attempt));
            if (ok) {
                reg_next_success = true;
                next_image = image_idx;
                break;
            }
            result.pipeline_log.push_back("Register image " + std::to_string(image_idx) + ": failed, " + failure_reason);
        }

        if (reg_next_success) {
            triangulator.triangulate_image(next_image, tri_options);
            triangulator.complete_image(next_image, tri_options);
            if (options.refine_bundle) {
                run_iterative_local_refinement(reconstruction_state,
                                               triangulator,
                                               tri_options,
                                               next_image,
                                               initial_pair_i,
                                               initial_pair_j,
                                               options);
            } else {
                complete_and_merge_tracks(triangulator, tri_options);
                filter_state_points(reconstruction_state, options);
            }
            result.stage_stats.push_back(make_stage_stat("register_image_" + std::to_string(next_image),
                                                         reconstruction_state));

            // COLMAP CheckRunGlobalRefinement: global BA when the registered
            // frame/point counts grow past the configured ratios/frequencies.
            const size_t num_reg_frames = count_registered_images(reconstruction_state);
            const size_t num_points = reconstruction_state.points3d.size();
            const bool run_global =
                static_cast<double>(num_reg_frames) >= 1.1 * static_cast<double>(ba_prev_num_reg_frames) ||
                num_reg_frames >= 500 + ba_prev_num_reg_frames ||
                static_cast<double>(num_points) >= 1.1 * static_cast<double>(ba_prev_num_points) ||
                num_points >= 250000 + ba_prev_num_points;
            if (run_global && options.refine_bundle) {
                run_iterative_global_refinement(reconstruction_state,
                                                triangulator,
                                                tri_options,
                                                initial_pair_i,
                                                initial_pair_j,
                                                options);
                result.stage_stats.push_back(make_stage_stat("global_refinement", reconstruction_state));
                ba_prev_num_reg_frames = count_registered_images(reconstruction_state);
                ba_prev_num_points = reconstruction_state.points3d.size();
            }
        } else if (prev_reg_next_success) {
            // COLMAP: a failed registration round triggers one global
            // refinement pass, then the loop retries registration once more.
            // Never run it on a two-camera model (degenerate, see above).
            if (options.refine_bundle && count_registered_images(reconstruction_state) >= 3) {
                run_iterative_global_refinement(reconstruction_state,
                                                triangulator,
                                                tri_options,
                                                initial_pair_i,
                                                initial_pair_j,
                                                options);
                result.stage_stats.push_back(make_stage_stat("global_refinement_retry", reconstruction_state));
            }
        }
    } while (reg_next_success || prev_reg_next_success);

    if (options.refine_bundle) {
        run_iterative_global_refinement(reconstruction_state,
                                        triangulator,
                                        tri_options,
                                        initial_pair_i,
                                        initial_pair_j,
                                        options);
        result.stage_stats.push_back(make_stage_stat("final_global_refinement", reconstruction_state));
    }

    // Optional staged final bundle: release only the shared SIMPLE_PINHOLE
    // focal length (principal point and distortion stay fixed), mirroring the
    // PyCOLMAP reference run for CylinderDIC.
    if (options.final_refine_focal_length) {
        MultiviewCalibrationOptions final_options = options;
        final_options.refine_focal_length = true;
        final_options.refine_principal_point = false;
        final_options.refine_extra_params = false;
        final_options.share_intrinsics = true;
        std::vector<SparsePointDiagnostic> diag_before =
            collect_point_diagnostics(reconstruction_state);
        if (run_state_bundle_adjustment(reconstruction_state,
                                        initial_pair_i,
                                        initial_pair_j,
                                        final_options,
                                        false)) {
            filter_state_points(reconstruction_state, options);
            result.pipeline_log.push_back("Final global BA with focal-length-only refinement (shared SIMPLE_PINHOLE)");
            result.stage_stats.push_back(make_stage_stat("final_focal_only_ba", reconstruction_state));
        }
        // General final geometric filter (never radius/ground-truth based):
        // minimum track length, per-observation reprojection error, positive
        // depth, multi-view depth consistency, min triangulation angle.
        std::vector<SparsePointDiagnostic> diag_after =
            collect_point_diagnostics(reconstruction_state);
        const size_t final_filtered_observations =
            final_filter_state_points(reconstruction_state, options);
        result.pipeline_log.push_back("Final general geometric filter: removed " +
                                      std::to_string(final_filtered_observations) +
                                      " observations (min_track_length=" +
                                      std::to_string(std::max(2, options.final_min_track_length)) +
                                      ", max_depth_ratio=" +
                                      std::to_string(std::max(1.0, options.final_max_depth_ratio)) + ")");
        result.stage_stats.push_back(make_stage_stat("final_filter", reconstruction_state));
        // Merge diagnostics: BA-before state from diag_before, BA-after state
        // from diag_after, and final-filter outcome from the surviving points.
        std::unordered_set<int> kept_ids;
        for (size_t pid = 0; pid < reconstruction_state.points3d.size(); ++pid) {
            const auto& point3d = reconstruction_state.points3d[pid];
            if (point3d.valid && point3d.track.size() >= 2) {
                kept_ids.insert(static_cast<int>(pid));
            }
        }
        result.point_diagnostics.resize(diag_after.size());
        for (size_t pid = 0; pid < diag_after.size(); ++pid) {
            if (diag_after[pid].point_id < 0) {
                result.point_diagnostics[pid].kept_by_final_filter = false;
                continue;
            }
            SparsePointDiagnostic merged = diag_after[pid];
            if (pid < diag_before.size()) {
                merged.xyz_before_final_ba = diag_before[pid].xyz_after_final_ba;
                merged.rms_before_final_ba = diag_before[pid].rms_after_final_ba;
            }
            merged.kept_by_final_filter = kept_ids.count(merged.point_id) > 0;
            result.point_diagnostics[pid] = merged;
        }
    }

    result.sparse_points = export_sparse_points_from_state(reconstruction_state);

    result.cameras.reserve(reconstruction_state.images.size());
    for (size_t i = 0; i < reconstruction_state.images.size(); ++i) {
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
    result.stage_stats.push_back(make_stage_stat("final", reconstruction_state));
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
