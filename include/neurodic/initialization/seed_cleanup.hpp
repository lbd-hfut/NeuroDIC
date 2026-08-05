/** Shared post-processing for sparse PIN seed sets. */
#pragma once

#include <torch/torch.h>

#include "neurodic/initialization/seed_set.hpp"

namespace neurodic {

struct SeedCleanupOptions {
    double mad_threshold{4.5};
    int min_seed_count{3};
};

// Removes displacement outliers and derives scale_uv from the retained seeds.
SeedSet clean_seed_set(torch::Tensor positions,
                       torch::Tensor displacement,
                       const SeedCleanupOptions& options = {});

}  // namespace neurodic
