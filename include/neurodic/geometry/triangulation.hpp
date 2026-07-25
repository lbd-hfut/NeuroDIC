/**
 * Stereo triangulation interface.
 *
 * Responsibilities: reconstruct 3D points for stereo PIN-DIC.
 * Inputs: left/right coordinates and calibration tensors.
 * Outputs: 3D point tensor.
 * Ownership: tensor references only.
 * Differentiable: PARTIAL. Use torch::Tensor if triangulation participates in loss.
 * TODO(NeuroDIC): avoid inventing equations before stereo convention validation.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

torch::Tensor triangulate_points(
    const torch::Tensor& left_coordinates,
    const torch::Tensor& right_coordinates,
    const torch::Tensor& calibration
);

}  // namespace neurodic
