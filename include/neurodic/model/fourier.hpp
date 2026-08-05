/** Fixed dyadic Fourier positional encoding shared by PIN and NDeF models. */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct FourierEncodingOptions {
    bool enabled{true};
    int num_frequencies{6};
    bool include_input{true};
    double angular_scale{3.14159265358979323846};
};

class FourierEncodingImpl : public torch::nn::Module {
public:
    FourierEncodingImpl(int input_dim, FourierEncodingOptions options = {});
    torch::Tensor forward(const torch::Tensor& coordinates);
    [[nodiscard]] int output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] const FourierEncodingOptions& options() const noexcept { return options_; }

private:
    int input_dim_;
    int output_dim_;
    FourierEncodingOptions options_;
    torch::Tensor frequencies_;
};
TORCH_MODULE(FourierEncoding);

}  // namespace neurodic
