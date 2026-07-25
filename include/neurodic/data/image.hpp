/**
 * Image data container.
 *
 * Responsibilities: represent image observations for C++ problem construction.
 * Inputs: image tensor or future CPU image storage.
 * Outputs: validated image dimensions.
 * Ownership: torch::Tensor uses PyTorch reference-counted tensor ownership.
 * Differentiable: NO for normal observed images. Images are fixed observations.
 * TODO(NeuroDIC): decide owned storage vs view semantics and channel layout.
 */
#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace neurodic {

class Image {
public:
    Image() = default;
    explicit Image(torch::Tensor tensor);

    [[nodiscard]] std::int64_t width() const;
    [[nodiscard]] std::int64_t height() const;
    [[nodiscard]] std::int64_t channels() const;
    [[nodiscard]] const torch::Tensor& tensor() const noexcept { return tensor_; }
    void validate() const;

private:
    torch::Tensor tensor_;
};

}  // namespace neurodic
