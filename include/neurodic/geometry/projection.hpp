/**
 * Differentiable projection interface.
 *
 * Responsibilities: project points with fixed camera parameters.
 * Inputs: 3D points and camera tensors.
 * Outputs: image coordinates.
 * Ownership: tensor references only.
 * Differentiable: YES with respect to points and any optimized camera tensors.
 * TODO(NeuroDIC): implement projection using only torch::Tensor operations.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

torch::Tensor project_points(const torch::Tensor& points, const torch::Tensor& camera);

}  // namespace neurodic
