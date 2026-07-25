/**
 * SIFT initializer adapter.
 *
 * Responsibilities: provide feature-based sparse warm starts.
 * Inputs: image observations and optional calibration.
 * Outputs: InitializationResult.
 * Ownership: future OpenCV state remains outside differentiable path.
 * Differentiable: NO.
 * TODO(NeuroDIC): add OpenCV-backed feature detection behind optional dependency.
 */
#pragma once

#include "neurodic/initialization/initializer.hpp"

namespace neurodic {

class SIFTInitializer : public Initializer {
public:
    InitializationResult run() const override;
};

}  // namespace neurodic
