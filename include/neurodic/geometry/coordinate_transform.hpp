/**
 * Coordinate transform helpers.
 *
 * Responsibilities: transform coordinates between image, normalized, camera, and world spaces.
 * Inputs: coordinate tensors and transform tensors.
 * Outputs: transformed coordinates.
 * Ownership: tensor references only.
 * Differentiable: YES when used between model output and loss.
 * TODO(NeuroDIC): implement transform conventions with torch::Tensor operations.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

torch::Tensor transform_coordinates(const torch::Tensor& coordinates, const torch::Tensor& transform);

}  // namespace neurodic
