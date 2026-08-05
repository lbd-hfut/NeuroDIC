#include "neurodic/model/fourier.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

FourierEncodingImpl::FourierEncodingImpl(int input_dim, FourierEncodingOptions options)
    : input_dim_(input_dim), options_(options) {
    if (input_dim_ < 1 || options_.num_frequencies < 1 || options_.angular_scale <= 0.0)
        throw ValidationError("Invalid fixed Fourier encoding options");
    output_dim_ = options_.enabled
        ? input_dim_ * (2 * options_.num_frequencies + (options_.include_input ? 1 : 0))
        : input_dim_;
    frequencies_ = register_buffer(
        "frequencies",
        torch::pow(2.0, torch::arange(options_.num_frequencies, torch::TensorOptions().dtype(torch::kFloat32))) *
            options_.angular_scale);
}

torch::Tensor FourierEncodingImpl::forward(const torch::Tensor& coordinates) {
    if (!coordinates.defined() || coordinates.dim() < 2 || coordinates.size(-1) != input_dim_)
        throw ValidationError("Fourier encoding expects [..., input_dim] coordinates");
    if (!options_.enabled) return coordinates;
    const auto frequencies = frequencies_.to(coordinates.device(), coordinates.scalar_type());
    const auto angles = coordinates.unsqueeze(-1) * frequencies;
    const auto encoded = torch::cat({torch::sin(angles), torch::cos(angles)}, -1).flatten(-2);
    return options_.include_input ? torch::cat({coordinates, encoded}, -1) : encoded;
}

}  // namespace neurodic
