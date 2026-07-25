/**
 * NDeF multi-view geometry engine.
 *
 * Responsibilities: project reference/deformed surfaces and handle visibility.
 * Inputs: surface/deformation tensors, camera tensors, view metadata.
 * Outputs: projected coordinates and visibility masks.
 * Ownership: future calibration/view state.
 * Differentiable: YES for geometry used in photometric loss.
 * TODO(NeuroDIC): design NDeF-specific geometry without forcing stereo triangulation.
 */
#pragma once

#include "neurodic/geometry/geometry.hpp"

namespace neurodic {

class NDeFGeometry : public Geometry {
public:
    torch::Tensor project_reference_surface(const torch::Tensor& surface) const;
    torch::Tensor project_deformed_surface(const torch::Tensor& surface, const torch::Tensor& deformation) const;
    torch::Tensor visibility(const torch::Tensor& surface) const;
};

}  // namespace neurodic
