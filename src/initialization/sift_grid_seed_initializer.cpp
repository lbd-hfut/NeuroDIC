#include "neurodic/initialization/sift_grid_seed_initializer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "neurodic/core/exceptions.hpp"

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace neurodic {

SiftGridSeedInitializer::SiftGridSeedInitializer(SiftGridSeedOptions options) : options_(options) {
    if (options_.target_seed_count <= 0 || options_.lowe_ratio <= 0.0 || options_.lowe_ratio >= 1.0 ||
        options_.min_seeds_per_roi < 0 || options_.mad_threshold <= 0.0)
        throw ValidationError("Invalid SIFT grid seed options");
}

SeedSet SiftGridSeedInitializer::initialize(const torch::Tensor& reference,
                                             const torch::Tensor& deformed,
                                             const torch::Tensor& roi_mask) const {
    if (!reference.defined() || !deformed.defined() || !roi_mask.defined() ||
        reference.dim() != 2 || deformed.sizes() != reference.sizes() || roi_mask.sizes() != reference.sizes())
        throw ValidationError("SIFT inputs must be equally shaped [H,W] tensors");
#ifndef NEURODIC_HAS_OPENCV
    return SeedSet::empty();
#else
    auto ref_u8 = reference.detach().to(torch::kCPU).clamp(0, 255).to(torch::kUInt8).contiguous();
    auto def_u8 = deformed.detach().to(torch::kCPU).clamp(0, 255).to(torch::kUInt8).contiguous();
    auto mask = roi_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous();
    cv::Mat ref(static_cast<int>(ref_u8.size(0)), static_cast<int>(ref_u8.size(1)), CV_8UC1, ref_u8.data_ptr());
    cv::Mat def(static_cast<int>(def_u8.size(0)), static_cast<int>(def_u8.size(1)), CV_8UC1, def_u8.data_ptr());
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp_ref, kp_def;
    cv::Mat desc_ref, desc_def;
    sift->detectAndCompute(ref, cv::noArray(), kp_ref, desc_ref);
    sift->detectAndCompute(def, cv::noArray(), kp_def, desc_def);
    if (desc_ref.empty() || desc_def.empty() || kp_ref.size() < 2 || kp_def.size() < 2)
        return SeedSet::empty();

    cv::FlannBasedMatcher matcher(
        cv::makePtr<cv::flann::KDTreeIndexParams>(options_.flann_trees),
        cv::makePtr<cv::flann::SearchParams>(options_.flann_checks));
    std::vector<std::vector<cv::DMatch>> raw;
    matcher.knnMatch(desc_ref, desc_def, raw, 2);

    struct Candidate { cv::Point2f pos; cv::Point2f uv; float quality; };
    std::vector<Candidate> candidates;
    auto mask_a = mask.accessor<bool, 2>();
    int xmin = static_cast<int>(mask.size(1)), xmax = -1, ymin = static_cast<int>(mask.size(0)), ymax = -1;
    for (int y = 0; y < mask.size(0); ++y) for (int x = 0; x < mask.size(1); ++x) if (mask_a[y][x]) {
        xmin = std::min(xmin, x); xmax = std::max(xmax, x); ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    if (xmax < xmin || ymax < ymin) return SeedSet::empty();
    for (const auto& pair : raw) {
        if (pair.size() < 2 || !(pair[0].distance < options_.lowe_ratio * pair[1].distance)) continue;
        const auto& m = pair[0];
        const auto p = kp_ref[static_cast<std::size_t>(m.queryIdx)].pt;
        const int x = static_cast<int>(std::lround(p.x)), y = static_cast<int>(std::lround(p.y));
        if (x < 0 || y < 0 || x >= mask.size(1) || y >= mask.size(0) || !mask_a[y][x]) continue;
        const auto q = kp_def[static_cast<std::size_t>(m.trainIdx)].pt;
        candidates.push_back({p, q - p, kp_ref[static_cast<std::size_t>(m.queryIdx)].response / (m.distance + 1e-6F)});
    }
    if (candidates.empty()) return SeedSet::empty();
    const double aspect = static_cast<double>(xmax - xmin + 1) / (ymax - ymin + 1);
    const int cols = std::max(1, static_cast<int>(std::lround(std::sqrt(options_.target_seed_count * aspect))));
    const int rows = std::max(1, static_cast<int>(std::lround(static_cast<double>(options_.target_seed_count) / cols)));
    std::map<std::pair<int, int>, Candidate> best;
    for (const auto& c : candidates) {
        const int col = std::min(cols - 1, static_cast<int>((c.pos.x - xmin) * cols / (xmax - xmin + 1)));
        const int row = std::min(rows - 1, static_cast<int>((c.pos.y - ymin) * rows / (ymax - ymin + 1)));
        auto key = std::make_pair(row, col);
        auto it = best.find(key);
        if (it == best.end() || c.quality > it->second.quality) best[key] = c;
    }
    if (static_cast<int>(best.size()) < options_.min_seeds_per_roi) return SeedSet::empty();
    std::vector<float> positions, displacements;
    for (const auto& item : best) {
        const auto& c = item.second;
        positions.insert(positions.end(), {c.pos.x,c.pos.y});
        displacements.insert(displacements.end(), {c.uv.x,c.uv.y});
    }
    if (positions.empty()) return SeedSet::empty();
    auto p = torch::from_blob(positions.data(), {static_cast<int64_t>(positions.size()/2),2}, torch::kFloat32).clone();
    auto uv = torch::from_blob(displacements.data(), {static_cast<int64_t>(displacements.size()/2),2}, torch::kFloat32).clone();
    return clean_seed_set(p, uv, {options_.mad_threshold, options_.min_seeds_per_roi});
#endif
}

}  // namespace neurodic
