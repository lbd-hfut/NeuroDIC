/**
 * Standardized calibration result.
 *
 * Responsibilities: be the only calibration output consumed by problems/solvers.
 * Inputs: one or more CameraModel values.
 * Outputs: validated camera collection.
 * Ownership: vector owns camera values.
 * Differentiable: PARTIAL. Parameters are fixed observations unless explicitly optimized.
 * TODO(NeuroDIC): add stereo baseline and COLMAP sparse-scene metadata.
 */
#pragma once

#include <vector>

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/core/types.hpp"

namespace neurodic {

struct CalibrationResult {
    CalibrationType type = CalibrationType::NONE;
    std::vector<CameraModel> cameras;
    void validate() const {}
};

}  // namespace neurodic
