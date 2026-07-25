/**
 * Mono calibration interface.
 *
 * Responsibilities: estimate single-camera calibration before problem construction.
 * Inputs: future calibration images/config.
 * Outputs: CalibrationResult.
 * Ownership: preprocessing adapter state.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement board detection and reprojection validation.
 */
#pragma once

#include "neurodic/calibration/calibration_result.hpp"

namespace neurodic {

class MonoCalibration {
public:
    CalibrationResult run() const;
};

}  // namespace neurodic
