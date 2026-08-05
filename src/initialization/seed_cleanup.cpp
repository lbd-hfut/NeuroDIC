#include "neurodic/initialization/seed_cleanup.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace {

float median(std::vector<float> values) {
    if (values.empty()) return 0.0F;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), middle, values.end());
    float value = *middle;
    if (values.size() % 2U == 0U) {
        const auto lower = std::max_element(values.begin(), middle);
        value = 0.5F * (value + *lower);
    }
    return value;
}

}  // namespace

SeedSet clean_seed_set(torch::Tensor positions, torch::Tensor displacement,
                       const SeedCleanupOptions& options) {
    positions = positions.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    displacement = displacement.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (options.mad_threshold <= 0.0 || options.min_seed_count < 1 || positions.dim() != 2 ||
        positions.size(1) != 2 || displacement.sizes() != positions.sizes()) {
        throw ValidationError("Invalid seed cleanup inputs or options");
    }
    if (positions.size(0) < options.min_seed_count) return SeedSet::empty();

    const auto count = static_cast<std::size_t>(positions.size(0));
    auto uv = displacement.accessor<float, 2>();
    std::vector<float> us, vs;
    us.reserve(count); vs.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { us.push_back(uv[i][0]); vs.push_back(uv[i][1]); }
    const float median_u = median(us), median_v = median(vs);
    std::vector<float> du, dv;
    du.reserve(count); dv.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        du.push_back(std::abs(us[i] - median_u)); dv.push_back(std::abs(vs[i] - median_v));
    }
    const float mad_u = median(du), mad_v = median(dv);
    std::vector<int64_t> keep;
    keep.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::abs(us[i] - median_u) < options.mad_threshold * mad_u + 1e-6F &&
            std::abs(vs[i] - median_v) < options.mad_threshold * mad_v + 1e-6F) {
            keep.push_back(static_cast<int64_t>(i));
        }
    }
    if (static_cast<int>(keep.size()) < options.min_seed_count) return SeedSet::empty();
    const auto indices = torch::tensor(keep, torch::TensorOptions().dtype(torch::kInt64));
    return SeedSet::from_tensors(positions.index_select(0, indices), displacement.index_select(0, indices));
}

}  // namespace neurodic
