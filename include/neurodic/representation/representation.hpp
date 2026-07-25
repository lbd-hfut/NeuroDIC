/**
 * Field representation interface.
 *
 * Responsibilities: decode physical fields from neural model outputs.
 * Inputs: coordinates and model output tensors.
 * Outputs: physical field tensors.
 * Ownership: implementations own no tensors by default.
 * Differentiable: YES. Decoding is between model output and loss and must
 * preserve the PyTorch autograd graph.
 * TODO(NeuroDIC): define coordinate conventions and output channel contracts.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

class FieldRepresentation {
public:
    virtual ~FieldRepresentation() = default;
    virtual torch::Tensor decode(
        const torch::Tensor& coordinates,
        const torch::Tensor& model_output
    ) const = 0;
};

}  // namespace neurodic
