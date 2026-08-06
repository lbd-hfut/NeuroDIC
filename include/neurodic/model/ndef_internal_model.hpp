/** Internal NDeF 3D reference-surface deformation model. */
#pragma once

#include <vector>

#include "neurodic/model/fourier.hpp"
#include "neurodic/model/neural_model.hpp"

namespace neurodic {

struct NDeFModelOptions {
    int hidden_dim{32};
    int hidden_layers{5};
    double output_scale{1.0};
    FourierEncodingOptions fourier_encoding{};
};

class NDeFInternalModel : public NeuralModel {
public:
    NDeFInternalModel(NDeFModelOptions options,
                      torch::Tensor coordinate_center,
                      torch::Tensor coordinate_scale);
    torch::Tensor forward(const torch::Tensor& points_world) override;
    torch::Tensor forward_normalized(const torch::Tensor& points_normalized);
    torch::Tensor normalize(const torch::Tensor& points_world) const;

private:
    NDeFModelOptions options_;
    FourierEncoding encoder_{nullptr};
    std::vector<torch::nn::Linear> layers_;
    torch::Tensor coordinate_center_;
    torch::Tensor coordinate_scale_;
};

}  // namespace neurodic
