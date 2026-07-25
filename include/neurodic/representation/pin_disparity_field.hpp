/**
 * PIN stereo disparity/correspondence representation.
 *
 * Responsibilities: decode stereo image-space correspondence for PINSolver.
 * Inputs: coordinates and model output.
 * Outputs: disparity/correspondence tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): do not finalize stereo parameterization before validation.
 */
#pragma once

#include "neurodic/representation/representation.hpp"

namespace neurodic {

class PINDisparityField : public FieldRepresentation {
public:
    torch::Tensor decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const override;
};

}  // namespace neurodic
