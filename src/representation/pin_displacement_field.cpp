#include "neurodic/representation/pin_displacement_field.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

PINDisplacementField::PINDisplacementField(PINDisplacementFieldOptions options)
    : mean_(options.mean.detach().to(torch::kCPU).to(torch::kFloat32).reshape({2})),
      scale_(options.scale.detach().to(torch::kCPU).to(torch::kFloat32).reshape({2}).clamp_min(1e-8)) {}

torch::Tensor PINDisplacementField::decode(const torch::Tensor& coordinates,
                                           const torch::Tensor& model_output) const {
    (void)coordinates;
    if (!model_output.defined() || model_output.dim() != 2 || model_output.size(1) != 2)
        throw ValidationError("PIN displacement field expects model output [N,2]");
    return model_output * scale_.to(model_output.device(), model_output.scalar_type()) +
           mean_.to(model_output.device(), model_output.scalar_type());
}

}  // namespace neurodic
