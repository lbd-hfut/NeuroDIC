#include "neurodic/model/ndef_internal_model.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

NDeFInternalModel::NDeFInternalModel(NDeFModelOptions options,
                                     torch::Tensor coordinate_center,
                                     torch::Tensor coordinate_scale)
    : options_(options) {
    if (options_.output_scale <= 0.0 || !coordinate_center.defined() || !coordinate_scale.defined() ||
        coordinate_center.numel() != 3 || coordinate_scale.numel() != 3)
        throw ValidationError("NDeF model requires positive output scale and [3] coordinate normalization");
    coordinate_center_ = register_buffer("coordinate_center", coordinate_center.detach().to(torch::kFloat32).reshape({1, 3}));
    coordinate_scale_ = register_buffer("coordinate_scale", coordinate_scale.detach().to(torch::kFloat32).reshape({1, 3}).clamp_min(1e-8));
    encoder_ = register_module("fourier_encoding", FourierEncoding(3, options_.fourier_encoding));
    int input_dim = encoder_->output_dim();
    for (int index = 0; index <= 5; ++index) {
        const int output_dim = index == 5 ? 3 : 32;
        auto layer = register_module("linear_" + std::to_string(index), torch::nn::Linear(input_dim, output_dim));
        torch::NoGradGuard no_grad;
        if (index == 5) {
            layer->weight.normal_(0.0, 1e-5);
        } else {
            torch::nn::init::xavier_uniform_(layer->weight, 5.0 / 3.0);
        }
        layer->bias.zero_();
        layers_.push_back(layer);
        input_dim = output_dim;
    }
}

torch::Tensor NDeFInternalModel::normalize(const torch::Tensor& points_world) const {
    if (!points_world.defined() || points_world.dim() != 2 || points_world.size(1) != 3)
        throw ValidationError("NDeF model expects [N,3] world coordinates");
    return (points_world - coordinate_center_.to(points_world.device(), points_world.scalar_type())) /
           coordinate_scale_.to(points_world.device(), points_world.scalar_type());
}

torch::Tensor NDeFInternalModel::forward_normalized(const torch::Tensor& points_normalized) {
    auto values = encoder_->forward(points_normalized);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        values = layers_[index]->forward(values);
        if (index + 1U < layers_.size()) values = torch::tanh(values);
    }
    return values * options_.output_scale;
}

torch::Tensor NDeFInternalModel::forward(const torch::Tensor& points_world) {
    return forward_normalized(normalize(points_world));
}

}  // namespace neurodic
