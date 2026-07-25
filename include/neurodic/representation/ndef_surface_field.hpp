/**
 * NDeF reference surface representation.
 *
 * Responsibilities: decode reference surface fields for multi-view DIC.
 * Inputs: surface coordinates and model outputs.
 * Outputs: reference surface tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): define surface coordinate domain and scale.
 */
#pragma once

#include "neurodic/representation/representation.hpp"

namespace neurodic {

class NDeFSurfaceField : public FieldRepresentation {
public:
    torch::Tensor decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const override;
};

}  // namespace neurodic
