/**
 * NDeF deformation representation.
 *
 * Responsibilities: decode 3D deformation fields for NDeF.
 * Inputs: surface coordinates and model outputs.
 * Outputs: deformation tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): define deformation basis and regularization hooks.
 */
#pragma once

#include "neurodic/representation/representation.hpp"

namespace neurodic {

class NDeFDeformationField : public FieldRepresentation {
public:
    torch::Tensor decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const override;
};

}  // namespace neurodic
