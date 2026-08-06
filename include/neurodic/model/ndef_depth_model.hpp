#pragma once

#include <torch/torch.h>

namespace neurodic {

struct NDeFDepthModelOptions {
    int hidden_dim{32};
    int pixel_layers{3};
    int camera_layers{2};
    int trunk_layers{3};
    int camera_embedding_dim{16};
    bool positional_encoding_enabled{false};
    int positional_encoding_num_frequencies{4};
};

// C++/LibTorch counterpart of NDeF-DIC SfMDepthFiLMNet.
class NDeFDepthModel : public torch::nn::Module {
public:
    NDeFDepthModel(int cameras, NDeFDepthModelOptions options);
    torch::Tensor forward(torch::Tensor normalized_uv, torch::Tensor camera_indices);
private:
    NDeFDepthModelOptions options_;
    torch::nn::Sequential pixel_head_{nullptr}, camera_head_{nullptr}, depth_head_{nullptr};
    torch::nn::Embedding camera_embedding_{nullptr};
    torch::Tensor frequencies_;
    int encoded_dim_{2};
    torch::Tensor encode(const torch::Tensor& uv) const;
};
} // namespace neurodic
