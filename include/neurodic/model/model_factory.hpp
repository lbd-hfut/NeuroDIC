/**
 * PIN model factory.
 *
 * Responsibilities: create user-selectable PIN models.
 * Inputs: model type and future architecture options.
 * Outputs: neural model instance.
 * Ownership: caller receives unique ownership.
 * Differentiable: YES for created models.
 * TODO(NeuroDIC): add MLP/SIREN/Fourier/ResNet construction after validation.
 */
#pragma once

#include <memory>
#include <string>

#include "neurodic/model/neural_model.hpp"

namespace neurodic {

class ModelFactory {
public:
    std::unique_ptr<NeuralModel> create_pin_model(const std::string& model_type) const;
};

}  // namespace neurodic
