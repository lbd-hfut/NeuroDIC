/** MSPINN-compatible tanh MLP used by the single-network PIN branch. */
#pragma once

#include <vector>

#include "neurodic/model/fourier.hpp"
#include "neurodic/model/neural_model.hpp"

namespace neurodic {

struct PINModelOptions {
    int input_dim{2};
    int output_dim{2};
    int hidden_dim{64};
    int hidden_layers{5};
    FourierEncodingOptions fourier_encoding{false, 6, true, 3.14159265358979323846};
};

class MLPModel : public NeuralModel {
public:
    explicit MLPModel(PINModelOptions options = {});
    torch::Tensor forward(const torch::Tensor& coordinates) override;
    [[nodiscard]] const PINModelOptions& options() const noexcept { return options_; }

private:
    PINModelOptions options_;
    FourierEncoding encoder_{nullptr};
    std::vector<torch::nn::Linear> layers_;
};

}  // namespace neurodic
