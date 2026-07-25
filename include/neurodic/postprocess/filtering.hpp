/**
 * Result filtering.
 *
 * Responsibilities: remove invalid/outlier values after solving.
 * Inputs: result tensors and masks.
 * Outputs: filtered tensors.
 * Ownership: tensor references only.
 * Differentiable: NO for exported analysis.
 * TODO(NeuroDIC): define robust filtering policy.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic { torch::Tensor filter_result(const torch::Tensor& values); }
