/**
 * ZNSSD loss shell.
 *
 * Responsibilities: future zero-normalized SSD loss.
 * Inputs: paired sampled intensity tensors.
 * Outputs: scalar loss tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic {

struct ZNSSDLossOptions {
    double epsilon{1e-6};
    int kernel_size{7};
};

class ZNSSDLoss : public Loss {
public:
    explicit ZNSSDLoss(ZNSSDLossOptions options = {});
    torch::Tensor compute(const torch::Tensor& residual) override;
    torch::Tensor compute(const torch::Tensor& reference, const torch::Tensor& deformed) const;
    torch::Tensor compute_windows(const torch::Tensor& reference, const torch::Tensor& deformed,
                                  const torch::Tensor& mask) const;
    torch::Tensor compute_image(const torch::Tensor& reference, const torch::Tensor& deformed,
                                const torch::Tensor& roi_mask) const;

private:
    ZNSSDLossOptions options_;
};

}  // namespace neurodic
