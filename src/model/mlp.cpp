#include "neurodic/model/mlp.hpp"

#include <cmath>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

MLPModel::MLPModel(PINModelOptions options) : options_(options) {
    if (options_.input_dim < 1 || options_.output_dim < 1 || options_.hidden_dim < 1 || options_.hidden_layers < 1)
        throw ValidationError("Invalid PIN MLP options");
    encoder_ = register_module("fourier_encoding", FourierEncoding(options_.input_dim, options_.fourier_encoding));
    int input_dim = encoder_->output_dim();
    for (int index = 0; index <= options_.hidden_layers; ++index) {
        const int output_dim = index == options_.hidden_layers ? options_.output_dim : options_.hidden_dim;
        auto layer = register_module("linear_" + std::to_string(index), torch::nn::Linear(input_dim, output_dim));
        torch::NoGradGuard no_grad;
        const double bound = 1.0 / std::sqrt(static_cast<double>(input_dim));
        layer->weight.uniform_(-bound, bound);
        layer->bias.uniform_(-bound, bound);
        layers_.push_back(layer);
        input_dim = output_dim;
    }
}

torch::Tensor MLPModel::forward(const torch::Tensor& coordinates) {
    auto values = encoder_->forward(coordinates);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        values = layers_[index]->forward(values);
        if (index + 1U < layers_.size()) values = torch::tanh(values);
    }
    return values;
}

}  // namespace neurodic
