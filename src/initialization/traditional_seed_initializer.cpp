#include "neurodic/initialization/traditional_seed_initializer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "neurodic/core/exceptions.hpp"

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace neurodic {
namespace {

#ifdef NEURODIC_HAS_OPENCV
struct Match {
    cv::Point2f reference;
    cv::Point2f displacement;
    float confidence{0.0F};
};

float median(std::vector<float> values) {
    if (values.empty()) return 0.0F;
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), mid, values.end());
    float result = *mid;
    if (values.size() % 2U == 0U) result = 0.5F * (result + *std::max_element(values.begin(), mid));
    return result;
}

cv::Mat tensor_to_u8(const torch::Tensor& image) {
    if (!image.defined() || image.dim() != 2) throw ValidationError("Seed images must have shape [H,W]");
    auto cpu = image.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    cv::Mat source(static_cast<int>(cpu.size(0)), static_cast<int>(cpu.size(1)), CV_32F, cpu.data_ptr<float>());
    cv::Mat result;
    source.convertTo(result, CV_8U);
    return result;
}

cv::Mat tensor_to_mask(const torch::Tensor& mask, int rows, int cols) {
    if (!mask.defined() || mask.dim() != 2 || mask.size(0) != rows || mask.size(1) != cols)
        throw ValidationError("Seed ROI mask must have image shape [H,W]");
    auto cpu = mask.detach().to(torch::kCPU).to(torch::kBool).contiguous();
    cv::Mat result(rows, cols, CV_8U, cv::Scalar(0));
    auto accessor = cpu.accessor<bool, 2>();
    for (int y = 0; y < rows; ++y) for (int x = 0; x < cols; ++x)
        result.at<unsigned char>(y, x) = accessor[y][x] ? 255U : 0U;
    return result;
}

std::vector<Match> sift_matches(const cv::Mat& reference, const cv::Mat& deformed,
                                const TraditionalSeedOptions& options) {
    auto sift = cv::SIFT::create(std::max(1, options.sift_max_features));
    std::vector<cv::KeyPoint> ref_keypoints, def_keypoints;
    cv::Mat ref_descriptors, def_descriptors;
    sift->detectAndCompute(reference, cv::noArray(), ref_keypoints, ref_descriptors);
    sift->detectAndCompute(deformed, cv::noArray(), def_keypoints, def_descriptors);
    if (ref_descriptors.empty() || def_descriptors.empty()) return {};

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> forward, reverse;
    matcher.knnMatch(ref_descriptors, def_descriptors, forward, 2);
    matcher.knnMatch(def_descriptors, ref_descriptors, reverse, 2);
    std::vector<int> reverse_best(static_cast<std::size_t>(def_descriptors.rows), -1);
    for (const auto& pair : reverse) {
        if (pair.size() >= 2U && pair[0].distance < options.sift_ratio_threshold * pair[1].distance)
            reverse_best[static_cast<std::size_t>(pair[0].queryIdx)] = pair[0].trainIdx;
    }

    std::vector<Match> matches;
    for (const auto& pair : forward) {
        if (pair.size() < 2U || !(pair[0].distance < options.sift_ratio_threshold * pair[1].distance)) continue;
        const auto& best = pair[0];
        if (best.trainIdx < 0 || best.trainIdx >= static_cast<int>(reverse_best.size()) ||
            reverse_best[static_cast<std::size_t>(best.trainIdx)] != best.queryIdx) continue;
        const auto p = ref_keypoints[static_cast<std::size_t>(best.queryIdx)].pt;
        const auto q = def_keypoints[static_cast<std::size_t>(best.trainIdx)].pt;
        matches.push_back({p, q - p, 1.0F / (1.0F + best.distance)});
    }
    if (matches.size() < 3U) return matches;
    std::vector<float> us, vs, du, dv;
    for (const auto& match : matches) { us.push_back(match.displacement.x); vs.push_back(match.displacement.y); }
    const float mu = median(us), mv = median(vs);
    for (std::size_t i = 0; i < matches.size(); ++i) { du.push_back(std::abs(us[i] - mu)); dv.push_back(std::abs(vs[i] - mv)); }
    const float gate_u = std::max(3.0F, static_cast<float>(options.sift_robust_mad_factor * 1.4826) * median(du));
    const float gate_v = std::max(3.0F, static_cast<float>(options.sift_robust_mad_factor * 1.4826) * median(dv));
    matches.erase(std::remove_if(matches.begin(), matches.end(), [&](const Match& match) {
        return std::abs(match.displacement.x - mu) > gate_u || std::abs(match.displacement.y - mv) > gate_v;
    }), matches.end());
    return matches;
}

bool sift_prior_at(const std::vector<Match>& matches, cv::Point2f point,
                   const TraditionalSeedOptions& options, cv::Point2f* prior) {
    std::vector<std::pair<float, const Match*>> nearby;
    for (const auto& match : matches) {
        const float distance = cv::norm(point - match.reference);
        if (distance <= options.sift_interpolation_radius) nearby.emplace_back(distance, &match);
    }
    std::sort(nearby.begin(), nearby.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    if (nearby.empty()) return false;
    nearby.resize(std::min(nearby.size(), static_cast<std::size_t>(std::max(1, options.sift_interpolation_neighbors))));
    float weight_sum = 0.0F;
    *prior = {0.0F, 0.0F};
    for (const auto& item : nearby) {
        const float weight = 1.0F / (item.first * item.first + 1.0F);
        *prior += weight * item.second->displacement;
        weight_sum += weight;
    }
    *prior *= 1.0F / weight_sum;
    return true;
}

std::vector<cv::Point2f> kmeans_candidates(const cv::Mat& mask, const TraditionalSeedOptions& options) {
    std::vector<cv::Point> points;
    const int margin = std::max(options.subset_radius, options.subpixel_enabled ? options.subpixel_subset_radius : 0) +
                       options.search_radius;
    for (int y = margin; y < mask.rows - margin; ++y) for (int x = margin; x < mask.cols - margin; ++x)
        if (mask.at<unsigned char>(y, x)) points.emplace_back(x, y);
    if (points.empty()) {
        for (int y = 0; y < mask.rows; ++y) for (int x = 0; x < mask.cols; ++x)
            if (mask.at<unsigned char>(y, x)) points.emplace_back(x, y);
    }
    if (points.empty()) return {};
    const int requested = std::min<int>(std::max(1, options.target_seed_count), static_cast<int>(points.size()));
    const int sample_count = std::min<int>(std::max(requested, options.kmeans_sample_limit), static_cast<int>(points.size()));
    cv::Mat samples(sample_count, 2, CV_32F);
    for (int i = 0; i < sample_count; ++i) {
        const auto index = static_cast<std::size_t>(std::llround(static_cast<double>(i) * (points.size() - 1U) /
                                                                   std::max(1, sample_count - 1)));
        samples.at<float>(i, 0) = static_cast<float>(points[index].x);
        samples.at<float>(i, 1) = static_cast<float>(points[index].y);
    }
    cv::Mat labels, centers;
    cv::kmeans(samples, requested, labels,
               cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                std::max(1, options.kmeans_iterations), 0.1),
               1, cv::KMEANS_PP_CENTERS, centers);
    std::vector<cv::Point2f> result;
    result.reserve(requested);
    for (int i = 0; i < centers.rows; ++i) result.emplace_back(centers.at<float>(i, 0), centers.at<float>(i, 1));
    return result;
}

bool integer_search(const cv::Mat& reference, const cv::Mat& deformed, cv::Point2f point,
                    const TraditionalSeedOptions& options, const cv::Point2f* prior,
                    cv::Point2f* displacement, float* confidence) {
    const int r = options.subset_radius;
    const int x = static_cast<int>(std::lround(point.x)), y = static_cast<int>(std::lround(point.y));
    if (x - r < 0 || y - r < 0 || x + r >= reference.cols || y + r >= reference.rows) return false;
    const cv::Mat patch = reference(cv::Rect(x - r, y - r, 2 * r + 1, 2 * r + 1));
    const int base_u = prior ? static_cast<int>(std::lround(prior->x)) : 0;
    const int base_v = prior ? static_cast<int>(std::lround(prior->y)) : 0;
    const int sr = std::max(0, options.search_radius);
    const int min_x = x + base_u - sr - r, min_y = y + base_v - sr - r;
    const int max_x = x + base_u + sr + r, max_y = y + base_v + sr + r;
    if (min_x < 0 || min_y < 0 || max_x >= deformed.cols || max_y >= deformed.rows) return false;
    cv::Mat scores;
    cv::matchTemplate(deformed(cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)), patch,
                      scores, cv::TM_CCOEFF_NORMED);
    double max_score = -1.0;
    cv::Point location;
    cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &location);
    *displacement = {static_cast<float>(min_x + location.x + r - x), static_cast<float>(min_y + location.y + r - y)};
    *confidence = static_cast<float>(max_score);
    return std::isfinite(max_score);
}

bool icgn_refine(const cv::Mat& reference, const cv::Mat& deformed, cv::Point2f point,
                 const TraditionalSeedOptions& options, cv::Point2f* displacement, float* confidence) {
    if (!options.subpixel_enabled) return true;
    const int r = options.subpixel_subset_radius;
    const int x = static_cast<int>(std::lround(point.x)), y = static_cast<int>(std::lround(point.y));
    if (x-r < 1 || y-r < 1 || x+r >= reference.cols-1 || y+r >= reference.rows-1) return false;

    // First-order affine inverse-compositional Gauss-Newton: the reference
    // gradient and Hessian are fixed for every iteration, as in Traditional-DIC.
    cv::Mat reference_f;
    reference.convertTo(reference_f, CV_32F);
    cv::Mat deformed_f;
    deformed.convertTo(deformed_f, CV_32F);
    const int integer_u = static_cast<int>(std::lround(displacement->x));
    const int integer_v = static_cast<int>(std::lround(displacement->y));
    if (x + integer_u-r < 0 || y + integer_v-r < 0 ||
        x + integer_u+r >= deformed.cols || y + integer_v+r >= deformed.rows) return false;
    const cv::Size patch_size(2*r + 1, 2*r + 1);
    cv::Mat hanning;
    cv::createHanningWindow(hanning, patch_size, CV_32F);
    double phase_response = 0.0;
    const auto phase_shift = cv::phaseCorrelate(
        reference_f(cv::Rect(x-r, y-r, patch_size.width, patch_size.height)),
        deformed_f(cv::Rect(x+integer_u-r, y+integer_v-r, patch_size.width, patch_size.height)),
        hanning, &phase_response);
    cv::Mat grad_x, grad_y;
    cv::Sobel(reference_f, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(reference_f, grad_y, CV_32F, 0, 1, 3);
    cv::Matx<double, 6, 6> hessian = cv::Matx<double, 6, 6>::zeros();
    std::vector<cv::Vec<float, 7>> samples;
    samples.reserve(static_cast<std::size_t>((2*r+1) * (2*r+1)));
    for (int local_y = -r; local_y <= r; ++local_y) for (int local_x = -r; local_x <= r; ++local_x) {
        const int px = x + local_x, py = y + local_y;
        const float gx = grad_x.at<float>(py, px), gy = grad_y.at<float>(py, px);
        cv::Vec<float, 7> sample{static_cast<float>(local_x), static_cast<float>(local_y),
                                  reference_f.at<float>(py, px), gx, gy, 0.0F, 0.0F};
        const double sd[6] = {gx, gx*local_x, gx*local_y, gy, gy*local_x, gy*local_y};
        for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) hessian(row, col) += sd[row] * sd[col];
        samples.push_back(sample);
    }
    cv::Mat hessian_mat(hessian), hessian_inverse;
    if (!cv::invert(hessian_mat, hessian_inverse, cv::DECOMP_SVD)) return false;
    cv::Matx<double, 6, 1> parameters(
        displacement->x + (phase_response > 0.05 ? phase_shift.x : 0.0), 0.0, 0.0,
        displacement->y + (phase_response > 0.05 ? phase_shift.y : 0.0), 0.0, 0.0);
    double final_error = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < std::max(1, options.subpixel_max_iterations); ++iteration) {
        cv::Matx<double, 6, 1> gradient = cv::Matx<double, 6, 1>::zeros();
        double squared_error = 0.0;
        int valid = 0;
        for (const auto& sample : samples) {
            const float dx = static_cast<float>(parameters(0) + parameters(1)*sample[0] + parameters(2)*sample[1]);
            const float dy = static_cast<float>(parameters(3) + parameters(4)*sample[0] + parameters(5)*sample[1]);
            const float wx = static_cast<float>(x) + sample[0] + dx;
            const float wy = static_cast<float>(y) + sample[1] + dy;
            if (wx < 0.0F || wy < 0.0F || wx >= deformed.cols-1.0F || wy >= deformed.rows-1.0F) continue;
            const int x0 = static_cast<int>(std::floor(wx)), y0 = static_cast<int>(std::floor(wy));
            const float fx = wx - x0, fy = wy - y0;
            const float value = (1.0F-fx)*(1.0F-fy)*deformed.at<unsigned char>(y0, x0) +
                                fx*(1.0F-fy)*deformed.at<unsigned char>(y0, x0+1) +
                                (1.0F-fx)*fy*deformed.at<unsigned char>(y0+1, x0) +
                                fx*fy*deformed.at<unsigned char>(y0+1, x0+1);
            const double residual = static_cast<double>(value - sample[2]);
            const double sd[6] = {sample[3], sample[3]*sample[0], sample[3]*sample[1],
                                  sample[4], sample[4]*sample[0], sample[4]*sample[1]};
            for (int index = 0; index < 6; ++index) gradient(index) += sd[index] * residual;
            squared_error += residual * residual;
            ++valid;
        }
        if (valid < 12) return false;
        const cv::Mat update_mat = hessian_inverse * cv::Mat(gradient);
        cv::Matx<double, 6, 1> update;
        update_mat.copyTo(cv::Mat(update));
        // Keep the first-order model inside its local linearization basin.
        // The integer search already supplies the large translation.
        const double limits[6] = {0.25, 0.01, 0.01, 0.25, 0.01, 0.01};
        for (int index = 0; index < 6; ++index)
            update(index) = std::max(-limits[index], std::min(limits[index], update(index)));
        parameters -= update;
        final_error = squared_error / valid;
        if (cv::norm(update) < options.subpixel_convergence_threshold) break;
    }
    displacement->x = static_cast<float>(parameters(0));
    displacement->y = static_cast<float>(parameters(3));
    *confidence = static_cast<float>(1.0 / (1.0 + final_error));
    return std::isfinite(displacement->x) && std::isfinite(displacement->y);
}
#endif
}  // namespace

TraditionalSeedInitializer::TraditionalSeedInitializer(TraditionalSeedOptions options) : options_(options) {
    if (options_.target_seed_count <= 0 || options_.subset_radius <= 0 || options_.search_radius < 0 ||
        options_.sift_ratio_threshold <= 0.0 || options_.sift_ratio_threshold >= 1.0 ||
        options_.subpixel_subset_radius <= 0 || options_.cleanup.mad_threshold <= 0.0) {
        throw ValidationError("Invalid Traditional seed initialization options");
    }
}

SeedSet TraditionalSeedInitializer::initialize(const torch::Tensor& reference,
                                               const torch::Tensor& deformed,
                                               const torch::Tensor& roi_mask) const {
#ifndef NEURODIC_HAS_OPENCV
    (void)reference; (void)deformed; (void)roi_mask;
    return SeedSet::empty();
#else
    const cv::Mat reference_u8 = tensor_to_u8(reference);
    const cv::Mat deformed_u8 = tensor_to_u8(deformed);
    if (reference_u8.size() != deformed_u8.size()) throw ValidationError("Seed images must have equal shape");
    const cv::Mat mask = tensor_to_mask(roi_mask, reference_u8.rows, reference_u8.cols);
    const auto candidates = kmeans_candidates(mask, options_);
    const auto matches = options_.sift_prior_enabled ? sift_matches(reference_u8, deformed_u8, options_) : std::vector<Match>{};
    std::vector<float> positions, displacement;
    positions.reserve(candidates.size() * 2U); displacement.reserve(candidates.size() * 2U);
    for (const auto& point : candidates) {
        cv::Point2f prior, uv;
        const bool has_prior = options_.sift_prior_enabled && sift_prior_at(matches, point, options_, &prior);
        float confidence = 0.0F;
        if (!integer_search(reference_u8, deformed_u8, point, options_, has_prior ? &prior : nullptr, &uv, &confidence)) continue;
        (void)icgn_refine(reference_u8, deformed_u8, point, options_, &uv, &confidence);
        positions.insert(positions.end(), {point.x, point.y});
        displacement.insert(displacement.end(), {uv.x, uv.y});
    }
    if (positions.empty()) return SeedSet::empty();
    const auto shape = std::vector<int64_t>{static_cast<int64_t>(positions.size() / 2U), 2};
    auto pos = torch::from_blob(positions.data(), shape, torch::kFloat32).clone();
    auto uv = torch::from_blob(displacement.data(), shape, torch::kFloat32).clone();
    return clean_seed_set(pos, uv, options_.cleanup);
#endif
}

}  // namespace neurodic
