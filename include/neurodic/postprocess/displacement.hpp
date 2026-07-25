/**
 * Displacement postprocessing.
 *
 * Responsibilities: compute derived displacement quantities after optimization.
 * Inputs: solved field tensors.
 * Outputs: derived tensors.
 * Ownership: tensor references only.
 * Differentiable: NO for exported analysis workflows.
 * TODO(NeuroDIC): define units and component conventions.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic { torch::Tensor displacement_magnitude(const torch::Tensor& displacement); }
