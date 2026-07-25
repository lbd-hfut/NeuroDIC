/**
 * Stereo geometry engine.
 *
 * Responsibilities: handle stereo projection, reconstruction, and 3D displacement.
 * Inputs: stereo correspondence tensors and calibration result.
 * Outputs: 3D reference/current/displacement tensors.
 * Ownership: future calibration state.
 * Differentiable: PARTIAL. Tensor operations are required when inside loss path.
 * TODO(NeuroDIC): validate stereo reconstruction formulas before implementation.
 */
#pragma once

#include "neurodic/geometry/geometry.hpp"

namespace neurodic {

class StereoGeometry : public Geometry {
public:
    torch::Tensor reconstruct_reference(const torch::Tensor& coordinates) const;
    torch::Tensor reconstruct_current(const torch::Tensor& coordinates) const;
    torch::Tensor displacement_3d(const torch::Tensor& reference, const torch::Tensor& current) const;
};

}  // namespace neurodic
