/** Differentiable pinhole projection for NDeF and geometry-aware losses. */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct ProjectionResult {
    torch::Tensor uv;     // [N,2]
    torch::Tensor depth;  // [N]
};

struct MultiViewProjectionResult {
    torch::Tensor uv;     // [N,V,2]
    torch::Tensor depth;  // [N,V]
};

torch::Tensor world_to_camera(const torch::Tensor& points, const torch::Tensor& rotation,
                              const torch::Tensor& translation);
ProjectionResult project_points_with_depth(const torch::Tensor& points, const torch::Tensor& intrinsics,
                                           const torch::Tensor& rotation, const torch::Tensor& translation,
                                           const torch::Tensor& distortion = {});
torch::Tensor project_points(const torch::Tensor& points, const torch::Tensor& intrinsics,
                             const torch::Tensor& rotation, const torch::Tensor& translation,
                             const torch::Tensor& distortion = {});
torch::Tensor project_points(const torch::Tensor& points, const torch::Tensor& projection_matrix);
MultiViewProjectionResult project_points_multi_view(const torch::Tensor& points,
                                                     const torch::Tensor& intrinsics,
                                                     const torch::Tensor& rotations,
                                                     const torch::Tensor& translations,
                                                     const torch::Tensor& distortions = {});

}  // namespace neurodic
