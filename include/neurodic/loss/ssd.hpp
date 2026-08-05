/** Differentiable photometric sum-of-squared-differences loss. */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic {

class SSDLoss : public Loss {
public:
    torch::Tensor compute(const torch::Tensor& residual) override;
    torch::Tensor compute(const torch::Tensor& reference, const torch::Tensor& deformed) const;
    torch::Tensor compute_masked(const torch::Tensor& reference, const torch::Tensor& deformed,
                                 const torch::Tensor& mask) const;
};

}  // namespace neurodic
