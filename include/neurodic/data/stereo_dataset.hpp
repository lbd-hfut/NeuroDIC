/**
 * Stereo dataset container.
 *
 * Responsibilities: carry left/right observations for stereo PIN-DIC.
 * Inputs: stereo image pairs and one ROI.
 * Outputs: validated stereo dataset.
 * Ownership: value object.
 * Differentiable: NO.
 * TODO(NeuroDIC): finalize pair ordering and synchronization metadata.
 */
#pragma once

#include "neurodic/data/dataset.hpp"

namespace neurodic {

struct StereoDataset {
    DICDataset left;
    DICDataset right;
    void validate() const {
        left.validate();
        right.validate();
    }
};

}  // namespace neurodic
