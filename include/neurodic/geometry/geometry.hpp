/**
 * Base geometry interface.
 *
 * Responsibilities: define geometry engines consumed by loss construction.
 * Inputs: tensors from representation/model outputs plus calibration.
 * Outputs: projected or reconstructed tensors.
 * Ownership: implementations own geometry state.
 * Differentiable: YES when called between model output and loss.
 * TODO(NeuroDIC): define common geometry contracts without forcing stereo/NDeF together.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

class Geometry {
public:
    virtual ~Geometry() = default;
};

}  // namespace neurodic
