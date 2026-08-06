#pragma once
#include <torch/torch.h>
#include "neurodic/model/ndef_depth_model.hpp"
namespace neurodic {
class NDeFSurfaceProblem {
public:
    NDeFSurfaceProblem(torch::Tensor sparse_uv, torch::Tensor sparse_cameras, torch::Tensor sparse_depth,
                       torch::Tensor image_sizes, torch::Tensor roi_masks, torch::Tensor query_uv, torch::Tensor query_cameras);
    void validate() const;
    torch::Tensor sparse_uv, sparse_cameras, sparse_depth, image_sizes, roi_masks, query_uv, query_cameras;
    NDeFDepthModelOptions model_options;
    int pretrain_iterations{3000}; double pretrain_learning_rate{1e-3}, weight_decay{1e-6}, smoothness_weight{1e-4};
    int smooth_samples_per_camera{256}; torch::Device device{torch::kCPU};
    // Dense reference-image refinement inputs.  Images/masks are [V,H,W],
    // camera tensors are K/R[V,3,3], t[V,3], distortion[V,D], and neighbours
    // is [V,2] with -1 for a missing topology neighbour.
    torch::Tensor reference_images, intrinsics, rotations, translations, distortions, dense_neighbors;
    int dense_iterations{0};
    int dense_samples_per_camera{10000};
    int dense_spacing_px{4};
    int dense_patch_radius{2};
    double dense_learning_rate{1e-4};
    double dense_anchor_weight{0.1};
    double dense_min_valid_patch_ratio{1.0};
    int dense_seed{20260806};
    void set_dense_inputs(torch::Tensor images, torch::Tensor k, torch::Tensor r, torch::Tensor t,
                          torch::Tensor distortion, torch::Tensor neighbours);
};
} // namespace neurodic
