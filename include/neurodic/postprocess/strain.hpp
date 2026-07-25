/**
 * Strain postprocessing.
 *
 * Responsibilities: derive strain from solved displacement fields.
 * Inputs: displacement field tensors.
 * Outputs: strain tensors.
 * Ownership: tensor references only.
 * Differentiable: NO for post-solve analysis.
 * TODO(NeuroDIC): validate strain definitions before implementation.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic { torch::Tensor compute_strain(const torch::Tensor& displacement); }
