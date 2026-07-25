/**
 * Integer-pixel initialization search.
 *
 * Responsibilities: estimate coarse sparse correspondences.
 * Inputs: fixed observations and sampled ROI coordinates.
 * Outputs: InitializationResult.
 * Ownership: implementation-defined preprocessing state.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement robust search without entering solver autograd path.
 */
#pragma once

#include "neurodic/initialization/initializer.hpp"

namespace neurodic {

class IntegerSearchInitializer : public Initializer {
public:
    InitializationResult run() const override;
};

}  // namespace neurodic
