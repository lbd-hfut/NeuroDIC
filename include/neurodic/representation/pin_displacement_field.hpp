/**
 * PIN 2D displacement representation.
 *
 * Responsibilities: decode `(u, v)` displacement fields.
 * Inputs: ROI coordinates and model outputs.
 * Outputs: displacement tensor.
 * Ownership: owns fixed output-normalization tensors.
 * Differentiable: YES.
 * TODO(NeuroDIC): define output normalization inversion and channel ordering.
 */
#pragma once

#include "neurodic/representation/representation.hpp"

namespace neurodic {

struct PINDisplacementFieldOptions {
    torch::Tensor mean{torch::zeros({2}, torch::kFloat32)};
    torch::Tensor scale{torch::ones({2}, torch::kFloat32)};
};

class PINDisplacementField : public FieldRepresentation {
public:
    explicit PINDisplacementField(PINDisplacementFieldOptions options = {});
    torch::Tensor decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const override;

private:
    torch::Tensor mean_;
    torch::Tensor scale_;
};

}  // namespace neurodic
