/**
 * Base DIC dataset container.
 *
 * Responsibilities: group reference/current images with a single ROI.
 * Inputs: observed images and ROI.
 * Outputs: validated problem-construction data.
 * Ownership: value object with reference-counted tensors.
 * Differentiable: NO. Dataset preparation is outside the model-to-loss path.
 * TODO(NeuroDIC): finalize temporal naming and image sequence support.
 */
#pragma once

#include "neurodic/data/image.hpp"
#include "neurodic/data/roi.hpp"

namespace neurodic {

struct DICDataset {
    Image reference;
    Image deformed;
    ROI roi;
    void validate() const {
        reference.validate();
        deformed.validate();
        roi.validate();
    }
};

}  // namespace neurodic
