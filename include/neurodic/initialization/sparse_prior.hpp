/**
 * Sparse prior container.
 *
 * Responsibilities: carry sampled sparse displacement priors.
 * Inputs: coordinates, displacement, confidence.
 * Outputs: validated prior for normalization/warm start.
 * Ownership: tensors use PyTorch ownership.
 * Differentiable: NO.
 * TODO(NeuroDIC): define outlier filtering and confidence calibration.
 */
#pragma once

#include "neurodic/initialization/initialization_result.hpp"

namespace neurodic {

struct SparsePrior {
    InitializationResult result;
    void validate() const {}
};

}  // namespace neurodic
