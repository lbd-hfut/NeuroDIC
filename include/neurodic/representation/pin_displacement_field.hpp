/**
 * PIN 2D displacement representation.
 *
 * Responsibilities: decode `(u, v)` displacement fields.
 * Inputs: ROI coordinates and model outputs.
 * Outputs: displacement tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): define output normalization inversion and channel ordering.
 */
#pragma once

#include "neurodic/representation/representation.hpp"

namespace neurodic {

class PINDisplacementField : public FieldRepresentation {
public:
    torch::Tensor decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const override;
};

}  // namespace neurodic
