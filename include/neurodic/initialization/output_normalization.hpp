/**
 * Output normalization for warm start.
 *
 * Responsibilities: estimate displacement mean and scale from sparse priors.
 * Inputs: SparsePrior.
 * Outputs: InitializationResult with normalization tensors.
 * Ownership: tensors use PyTorch ownership.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement robust mean/scale estimation.
 */
#pragma once

#include "neurodic/initialization/sparse_prior.hpp"

namespace neurodic {

InitializationResult estimate_output_normalization(const SparsePrior& prior);

}  // namespace neurodic
