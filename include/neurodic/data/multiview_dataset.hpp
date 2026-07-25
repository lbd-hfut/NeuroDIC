/**
 * Multi-view dataset container.
 *
 * Responsibilities: carry observations for NDeF multi-view DIC.
 * Inputs: multiple per-view datasets sharing one conceptual ROI/surface domain.
 * Outputs: validated multi-view dataset.
 * Ownership: vector owns dataset values.
 * Differentiable: NO.
 * TODO(NeuroDIC): define view IDs, masks, timestamps, and surface-domain sampling.
 */
#pragma once

#include <vector>

#include "neurodic/data/dataset.hpp"

namespace neurodic {

struct MultiViewDataset {
    std::vector<DICDataset> views;
    void validate() const;
};

}  // namespace neurodic
