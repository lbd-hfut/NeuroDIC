/**
 * Regularization shell.
 *
 * Responsibilities: define smoothness/physical priors for neural fields.
 * Inputs: field tensors and derivatives.
 * Outputs: scalar regularization tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): add validated regularization terms after base pipeline works.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

torch::Tensor regularization_term(const torch::Tensor& field);

}  // namespace neurodic
