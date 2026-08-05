/** Process-wide deterministic random-state control for NeuroDIC runs. */
#pragma once

#include <cstdint>

namespace neurodic {

/// Seed LibTorch and, when enabled, OpenCV's global RNG before a run starts.
void set_random_seed(std::uint64_t seed);

}  // namespace neurodic
