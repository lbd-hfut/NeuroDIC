/**
 * Runtime context shell.
 *
 * Responsibilities: carry minimal execution choices without over-engineered
 * backend dispatch.
 * Inputs: device name, dtype, seed, debug flag.
 * Outputs: immutable/read-write configuration value.
 * Ownership: context owns its strings and scalar values.
 * Differentiable: NO.
 * TODO(NeuroDIC): map device strings to torch::Device after backend policy is validated.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <torch/torch.h>

namespace neurodic {

struct RuntimeContext {
    std::string device = "auto";
    torch::Dtype dtype = torch::kFloat32;
    std::optional<std::uint64_t> random_seed;
    bool debug = false;

    void validate() const;
};

}  // namespace neurodic
