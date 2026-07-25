/**
 * Visibility helpers.
 *
 * Responsibilities: define future visibility masks for multi-view sampling.
 * Inputs: surface and camera tensors.
 * Outputs: visibility tensor.
 * Ownership: tensor references only.
 * Differentiable: PARTIAL. Soft visibility may be differentiable; hard masks are not.
 * TODO(NeuroDIC): choose hard/soft visibility strategy after NDeF validation.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

torch::Tensor compute_visibility(const torch::Tensor& surface);

}  // namespace neurodic
