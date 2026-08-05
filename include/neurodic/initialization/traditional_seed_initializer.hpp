/** Traditional-DIC sparse seed chain for PIN 2D warm starts. */
#pragma once

#include <torch/torch.h>

#include "neurodic/initialization/seed_cleanup.hpp"

namespace neurodic {

struct TraditionalSeedOptions {
    int target_seed_count{256};
    int kmeans_iterations{20};
    int kmeans_sample_limit{20000};
    int subset_radius{10};
    int search_radius{30};
    bool pyramid_enabled{true};
    int pyramid_scale{4};
    int pyramid_refinement_radius{4};
    bool sift_prior_enabled{true};
    int sift_max_features{4000};
    double sift_ratio_threshold{0.75};
    double sift_robust_mad_factor{5.0};
    int sift_interpolation_neighbors{8};
    double sift_interpolation_radius{180.0};
    bool subpixel_enabled{true};
    int subpixel_subset_radius{15};
    int subpixel_max_iterations{30};
    double subpixel_convergence_threshold{1e-3};
    SeedCleanupOptions cleanup{};
};

class TraditionalSeedInitializer {
public:
    explicit TraditionalSeedInitializer(TraditionalSeedOptions options = {});
    SeedSet initialize(const torch::Tensor& reference,
                       const torch::Tensor& deformed,
                       const torch::Tensor& roi_mask) const;

private:
    TraditionalSeedOptions options_;
};

}  // namespace neurodic
