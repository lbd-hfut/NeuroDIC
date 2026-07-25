/**
 * ResNet-like model shell.
 *
 * Responsibilities: future residual PIN model.
 * Inputs: coordinates.
 * Outputs: model tensor.
 * Ownership: future torch::nn modules.
 * Differentiable: YES.
 * TODO(NeuroDIC): define residual block design after baseline MLP validation.
 */
#pragma once

#include "neurodic/model/neural_model.hpp"

namespace neurodic { class ResNetModel : public NeuralModel { public: torch::Tensor forward(const torch::Tensor& coordinates) override; }; }
