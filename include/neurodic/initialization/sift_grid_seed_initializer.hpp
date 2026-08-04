/** Optional OpenCV SIFT/FLANN preprocessing seed source. */
#pragma once

#include <torch/torch.h>

#include "neurodic/initialization/seed_set.hpp"

namespace neurodic {

struct SiftGridSeedOptions {
    int target_seed_count{128};
    double lowe_ratio{0.75};
    int flann_trees{5};
    int flann_checks{50};
    double mad_threshold{4.5};
    int min_seeds_per_roi{3};
};

class SiftGridSeedInitializer {
public:
    explicit SiftGridSeedInitializer(SiftGridSeedOptions options = {});
    SeedSet initialize(const torch::Tensor& reference,
                       const torch::Tensor& deformed,
                       const torch::Tensor& roi_mask) const;
private:
    SiftGridSeedOptions options_;
};

}  // namespace neurodic
