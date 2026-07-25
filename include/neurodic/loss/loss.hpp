/**
 * Loss interface.
 *
 * Responsibilities: compute differentiable scalar objectives.
 * Inputs: future residual tensors and problem context.
 * Outputs: scalar loss tensor.
 * Ownership: implementations own no tensors by default.
 * Differentiable: YES. Loss implementations must not detach tensors.
 * TODO(NeuroDIC): finalize argument contracts after photometric path is wired.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

class Loss {
public:
    virtual ~Loss() = default;
    virtual torch::Tensor compute(const torch::Tensor& residual) = 0;
};

}  // namespace neurodic
