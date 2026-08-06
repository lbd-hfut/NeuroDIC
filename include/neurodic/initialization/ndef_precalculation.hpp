/** Robust NDeF-DIC sparse displacement-scale preprocessing. */
#pragma once

#include <torch/torch.h>

#include <vector>

#include "neurodic/calibration/camera_model.hpp"

namespace neurodic {

struct NDeFDisplacementScale {
    torch::Tensor inlier_mask;  // CPU bool [N]
    double median{0.0};
    double mean{0.0};
    double p75{0.0};
    double p90{0.0};
    double maximum{0.0};
};

// Exact NDeF-DIC convention: MAD is computed from |dX| and an all-inlier
// fallback is used when MAD is effectively zero.
NDeFDisplacementScale estimate_ndef_displacement_scale(const torch::Tensor& displacement,
                                                        double mad_threshold = 5.0);

// Sparse stage of the original NDeF-DIC patch-DIC precalculation.  Images and
// masks are [V,H,W].  projected_uv/visibility are the reference-surface
// dataset used only to obtain robust cross-camera search centres.
struct NDeFSparsePrecalculationOptions {
    int points_per_camera{300};
    int neighbors_per_camera{2};
    int patch_radius{10};
    int cross_search_radius{40};
    int temporal_search_radius{8};
    double cross_ncc_threshold{0.45};
    double temporal_ncc_threshold{0.55};
    double min_texture_std{0.02};
    double max_reprojection_error{3.0};
    double displacement_mad_threshold{5.0};
};

struct NDeFSparsePrecalculationResult {
    torch::Tensor source_camera;              // CPU int64 [N]
    torch::Tensor source_uv;                  // CPU float64 [N,2]
    torch::Tensor reference_points;           // CPU float64 [N,3]
    torch::Tensor current_points;             // CPU float64 [N,3]
    torch::Tensor displacement;               // CPU float64 [N,3]
    torch::Tensor displacement_magnitude;     // CPU float64 [N]
    torch::Tensor camera_count;               // CPU int64 [N]
    torch::Tensor reference_reprojection_error; // CPU float64 [N]
    torch::Tensor current_reprojection_error; // CPU float64 [N]
    torch::Tensor mean_match_score;           // CPU float64 [N]
    torch::Tensor inlier_mask;                // CPU bool [N]
    NDeFDisplacementScale scale;
};

// Dense reference-surface sampler.  It consumes the already reconstructed
// dense cloud (the original Stage-1 product), voxel-uniformly subsamples it,
// and materializes the exact NDeFProblem observation tensors.
struct NDeFDenseSurfaceSampleOptions { double voxel_size{0.0}; int max_points{100000}; int min_visible_views{2}; };
struct NDeFDenseSurfaceSampleResult {
    torch::Tensor points;           // CPU float64 [N,3]
    torch::Tensor visibility_mask;  // CPU bool [N,V]
    torch::Tensor projected_uv;     // CPU float64 [N,V,2]
    torch::Tensor visible_counts;   // CPU float32 [N]
};
class NDeFDenseSurfaceSampler {
public:
    explicit NDeFDenseSurfaceSampler(NDeFDenseSurfaceSampleOptions options = {}) : options_(options) {}
    NDeFDenseSurfaceSampleResult sample(const torch::Tensor& dense_points, const torch::Tensor& roi_masks,
                                        const std::vector<CameraModel>& cameras) const;
private: NDeFDenseSurfaceSampleOptions options_;
};

class NDeFSparsePrecalculator {
public:
    explicit NDeFSparsePrecalculator(NDeFSparsePrecalculationOptions options = {}) : options_(options) {}

    NDeFSparsePrecalculationResult solve(const torch::Tensor& reference_images,
                                         const torch::Tensor& current_images,
                                         const torch::Tensor& roi_masks,
                                         const torch::Tensor& reference_visibility,
                                         const torch::Tensor& reference_projected_uv,
                                         const std::vector<CameraModel>& cameras) const;

private:
    NDeFSparsePrecalculationOptions options_;
};

}  // namespace neurodic
