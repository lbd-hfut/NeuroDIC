/**
 * Internal NDeF model shell.
 *
 * Responsibilities: own NDeF topology internally without public topology exposure.
 * Inputs: NDeF coordinates/internal features.
 * Outputs: surface/deformation model tensors.
 * Ownership: solver/internal factory controls construction.
 * Differentiable: YES.
 * TODO(NeuroDIC): design internal topology privately after NDeF math is validated.
 */
#pragma once

#include "neurodic/model/neural_model.hpp"

namespace neurodic { class NDeFInternalModel : public NeuralModel { public: torch::Tensor forward(const torch::Tensor& coordinates) override; }; }
