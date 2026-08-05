#include "neurodic/model/model_factory.hpp"

#include <algorithm>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

std::shared_ptr<NeuralModel> ModelFactory::create_pin_model(const std::string& model_type,
                                                             const PINModelOptions& options) const {
    if (model_type == "mlp") return std::make_shared<MLPModel>(options);
    throw ValidationError("PIN model type must be 'mlp'; Fourier encoding is an MLP input option");
}

}  // namespace neurodic
