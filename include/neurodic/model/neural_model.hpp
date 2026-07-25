/**
 * Neural model abstraction.
 *
 * Responsibilities: wrap LibTorch-compatible neural networks.
 * Inputs: coordinate tensors.
 * Outputs: model output tensors.
 * Ownership: model implementations own torch::nn modules.
 * Differentiable: YES. Forward passes must preserve parameter and input autograd.
 * TODO(NeuroDIC): decide torch::nn::Module inheritance vs wrapper policy.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

class NeuralModel {
public:
    virtual ~NeuralModel() = default;
    virtual torch::Tensor forward(const torch::Tensor& coordinates) = 0;
};

}  // namespace neurodic
