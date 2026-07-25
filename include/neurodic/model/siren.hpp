/**
 * SIREN model shell.
 *
 * Responsibilities: future sinusoidal PIN model.
 * Inputs: coordinates.
 * Outputs: model tensor.
 * Ownership: future torch::nn modules.
 * Differentiable: YES.
 * TODO(NeuroDIC): implement initialization and frequency scaling.
 */
#pragma once

#include "neurodic/model/neural_model.hpp"

namespace neurodic { class SIRENModel : public NeuralModel { public: torch::Tensor forward(const torch::Tensor& coordinates) override; }; }
