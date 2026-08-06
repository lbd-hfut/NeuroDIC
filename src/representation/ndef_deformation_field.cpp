#include "neurodic/representation/ndef_deformation_field.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
torch::Tensor NDeFDeformationField::decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const {
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != 3 ||
        !model_output.defined() || model_output.sizes() != coordinates.sizes())
        throw ValidationError("NDeF deformation field expects matching [N,3] coordinates and model output");
    return model_output;
}
}  // namespace neurodic
