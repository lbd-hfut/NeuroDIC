/**
 * MLP model shell.
 *
 * Responsibilities: future fully connected PIN model.
 * Inputs: coordinates.
 * Outputs: model tensor.
 * Ownership: future torch::nn modules.
 * Differentiable: YES.
 * TODO(NeuroDIC): implement LibTorch MLP after field contracts are fixed.
 */
#pragma once

#include "neurodic/model/neural_model.hpp"

namespace neurodic { class MLPModel : public NeuralModel { public: torch::Tensor forward(const torch::Tensor& coordinates) override; }; }
