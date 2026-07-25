/**
 * Subpixel initialization refinement.
 *
 * Responsibilities: refine coarse correspondences before neural training.
 * Inputs: sparse prior and image observations.
 * Outputs: refined InitializationResult.
 * Ownership: implementation-defined.
 * Differentiable: NO.
 * TODO(NeuroDIC): define interpolation and confidence models for refinement.
 */
#pragma once

#include "neurodic/initialization/initializer.hpp"

namespace neurodic {

class SubpixelSearchInitializer : public Initializer {
public:
    InitializationResult run() const override;
};

}  // namespace neurodic
