#include "neurodic/representation/ndef_surface_field.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
torch::Tensor NDeFSurfaceField::decode(const torch::Tensor& coordinates, const torch::Tensor& model_output) const {
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != 3 ||
        !model_output.defined() || model_output.sizes() != coordinates.sizes())
        throw ValidationError("NDeF surface field expects matching [N,3] coordinates and deformation");
    return coordinates + model_output;
}
}  // namespace neurodic
