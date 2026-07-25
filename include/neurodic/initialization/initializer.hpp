/**
 * Initializer interface.
 *
 * Responsibilities: define non-differentiable warm-start strategies.
 * Inputs: future dataset/ROI/calibration objects.
 * Outputs: InitializationResult.
 * Ownership: implementations own preprocessing state.
 * Differentiable: NO.
 * TODO(NeuroDIC): add concrete arguments after data flow is validated.
 */
#pragma once

#include "neurodic/initialization/initialization_result.hpp"

namespace neurodic {

class Initializer {
public:
    virtual ~Initializer() = default;
    virtual InitializationResult run() const = 0;
};

}  // namespace neurodic
