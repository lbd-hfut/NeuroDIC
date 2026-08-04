#include "neurodic/initialization/seed_set.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

SeedSet SeedSet::empty(torch::ScalarType dtype) {
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return {torch::empty({0, 2}, options), torch::empty({0, 2}, options), torch::tensor({0., 0., 1., 1.}, options)};
}

SeedSet SeedSet::constant(const torch::Tensor& positions, double u, double v) {
    if (!positions.defined() || positions.dim() != 2 || positions.size(1) != 2)
        throw ValidationError("Seed positions must have shape [N,2]");
    auto pos = positions.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto uv = torch::empty_like(pos);
    uv.select(1, 0).fill_(u);
    uv.select(1, 1).fill_(v);
    return from_tensors(pos, uv);
}

SeedSet SeedSet::from_tensors(torch::Tensor positions, torch::Tensor displacement) {
    positions = positions.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    displacement = displacement.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (positions.dim() != 2 || positions.size(1) != 2 || displacement.sizes() != positions.sizes())
        throw ValidationError("Seed positions/displacements must have matching [N,2] shapes");
    if (positions.size(0) == 0) return empty(torch::kFloat32);
    auto means = displacement.mean(0);
    auto half_ranges = (std::get<0>(displacement.max(0)) - std::get<0>(displacement.min(0))) / 2;
    half_ranges = torch::clamp_min(half_ranges, 1e-6);
    SeedSet result{positions, displacement, torch::cat({means, half_ranges})};
    result.validate();
    return result;
}

void SeedSet::validate() const {
    if (!seed_pos.defined() || !seed_uv.defined() || !scale_uv.defined() ||
        seed_pos.dim() != 2 || seed_pos.size(1) != 2 || seed_uv.sizes() != seed_pos.sizes() ||
        scale_uv.dim() != 1 || scale_uv.size(0) != 4 || seed_pos.device().is_cuda() ||
        seed_uv.device().is_cuda() || scale_uv.device().is_cuda())
        throw ValidationError("Invalid SeedSet; coordinates must be original-image CPU [N,2]");
}

}  // namespace neurodic
