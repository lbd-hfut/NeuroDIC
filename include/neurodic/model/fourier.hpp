/**
 * Fourier-feature model shell.
 *
 * Responsibilities: future Fourier encoded PIN model.
 * Inputs: coordinates.
 * Outputs: model tensor.
 * Ownership: future torch::nn modules.
 * Differentiable: YES.
 * TODO(NeuroDIC): define feature matrix ownership and scaling.
 */
#pragma once

#include "neurodic/model/neural_model.hpp"

namespace neurodic { class FourierModel : public NeuralModel { public: torch::Tensor forward(const torch::Tensor& coordinates) override; }; }
