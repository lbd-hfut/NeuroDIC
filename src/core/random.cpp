#include "neurodic/core/random.hpp"

#include <limits>

#include <torch/torch.h>

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/core.hpp>
#endif

namespace neurodic {

void set_random_seed(const std::uint64_t seed) {
    // torch::manual_seed covers CPU and CUDA generators exposed by LibTorch.
    torch::manual_seed(seed);
#ifdef NEURODIC_HAS_OPENCV
    // OpenCV APIs such as KMEANS_PP_CENTERS and RANSAC consume this process-global RNG.
    cv::setRNGSeed(static_cast<int>(seed % static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
#endif
}

}  // namespace neurodic
