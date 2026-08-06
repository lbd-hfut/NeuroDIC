#include "neurodic/model/ndef_depth_model.hpp"
#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace { torch::nn::Sequential mlp(int in, int hidden, int out, int layers) {
    torch::nn::Sequential net; for (int i=0;i<layers;++i) { net->push_back(torch::nn::Linear(in, hidden)); net->push_back(torch::nn::Tanh()); in=hidden; }
    net->push_back(torch::nn::Linear(in,out)); return net; } }
NDeFDepthModel::NDeFDepthModel(int cameras, NDeFDepthModelOptions options) : options_(options) {
    if (cameras < 1 || options_.hidden_dim < 1 || options_.pixel_layers < 1 || options_.camera_layers < 1 || options_.trunk_layers < 1 || options_.camera_embedding_dim < 1)
        throw ValidationError("NDeF depth model options are invalid");
    if (options_.positional_encoding_enabled) { frequencies_ = register_buffer("frequencies", torch::pow(2.0, torch::arange(options_.positional_encoding_num_frequencies, torch::kFloat32))); encoded_dim_=4*options_.positional_encoding_num_frequencies; }
    pixel_head_ = register_module("pixel_head", mlp(encoded_dim_,options_.hidden_dim,options_.hidden_dim,options_.pixel_layers));
    camera_embedding_ = register_module("camera_embedding", torch::nn::Embedding(cameras,options_.camera_embedding_dim));
    camera_head_ = register_module("camera_head", mlp(options_.camera_embedding_dim,options_.hidden_dim,2*options_.hidden_dim,options_.camera_layers));
    depth_head_ = register_module("depth_head", mlp(options_.hidden_dim,options_.hidden_dim,1,options_.trunk_layers));
}
torch::Tensor NDeFDepthModel::encode(const torch::Tensor& uv) const {
    if (!options_.positional_encoding_enabled) return uv;
    constexpr double pi = 3.14159265358979323846;
    // Match Python FourierPixelEncoding exactly: [N,2] -> [N,F,2], then
    // flatten frequency-major [sin(x_f), sin(y_f), cos(x_f), cos(y_f)].
    auto angles = uv.unsqueeze(1) * frequencies_.to(uv.device(),uv.scalar_type()).view({1,-1,1}) * pi;
    return torch::cat(torch::TensorList{torch::sin(angles), torch::cos(angles)}, -1).flatten(1);
}
torch::Tensor NDeFDepthModel::forward(torch::Tensor uv, torch::Tensor camera_indices) {
    auto pixel = pixel_head_->forward(encode(uv)); auto modulation = camera_head_->forward(camera_embedding_->forward(camera_indices.to(torch::kLong)));
    auto chunks=modulation.chunk(2,1); return depth_head_->forward((1.0+0.1*chunks[0])*pixel+chunks[1]).squeeze(1);
}
} // namespace neurodic
