/**
 * Initialization result.
 *
 * Responsibilities: carry sparse warm-start samples for neural-field training.
 * Inputs: sampled coordinates, displacement, confidence, normalization tensors.
 * Outputs: standardized warm-start container.
 * Ownership: torch tensors use reference-counted ownership.
 * Differentiable: NO. Initialization is preprocessing.
 * TODO(NeuroDIC): define coordinate frame and confidence semantics.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct InitializationResult {
    torch::Tensor coordinates;
    torch::Tensor displacement;
    torch::Tensor confidence;
    torch::Tensor displacement_mean;
    torch::Tensor displacement_scale;
};

}  // namespace neurodic
