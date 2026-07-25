/**
 * ROI sampling interfaces.
 *
 * Responsibilities: prepare coordinates inside a single ROI.
 * Inputs: ROI and requested sample count.
 * Outputs: coordinate tensors.
 * Ownership: returned tensors are owned by PyTorch.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement deterministic uniform/random sampling policies.
 */
#pragma once

#include <cstdint>
#include <torch/torch.h>

#include "neurodic/data/roi.hpp"

namespace neurodic {

torch::Tensor sample_roi_uniform(const ROI& roi, std::int64_t count);

}  // namespace neurodic
