/**
 * Stereo calibration interface.
 *
 * Responsibilities: estimate synchronized camera calibration before solving.
 * Inputs: future stereo calibration observations.
 * Outputs: CalibrationResult.
 * Ownership: preprocessing adapter state.
 * Differentiable: NO.
 * TODO(NeuroDIC): implement stereo calibration and baseline validation.
 */
#pragma once

#include "neurodic/calibration/calibration_result.hpp"

namespace neurodic {

class StereoCalibration {
public:
    CalibrationResult run() const;
};

}  // namespace neurodic
