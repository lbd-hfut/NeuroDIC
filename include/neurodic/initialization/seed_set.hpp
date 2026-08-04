/** Unified PIN-DIC seed output in original image coordinates (x, y). */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct SeedSet {
    torch::Tensor seed_pos;  // [N,2], original coordinates, x,y
    torch::Tensor seed_uv;   // [N,2], displacement u,v
    torch::Tensor scale_uv;  // [4], mean_u,mean_v,halfrange_u,halfrange_v

    static SeedSet empty(torch::ScalarType dtype = torch::kFloat32);
    static SeedSet constant(const torch::Tensor& positions, double u, double v);
    static SeedSet from_tensors(torch::Tensor positions, torch::Tensor displacement);
    void validate() const;
};

}  // namespace neurodic
