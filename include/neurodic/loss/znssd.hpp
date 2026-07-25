/**
 * ZNSSD loss shell.
 *
 * Responsibilities: future zero-normalized SSD loss.
 * Inputs: paired sampled intensity tensors.
 * Outputs: scalar loss tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 * TODO(NeuroDIC): define stable normalization windows and epsilon policy.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

namespace neurodic { class ZNSSDLoss : public Loss { public: torch::Tensor compute(const torch::Tensor& residual) override; }; }
