/**
 * COLMAP calibration adapter.
 *
 * Responsibilities: parse COLMAP products into CalibrationResult.
 * Inputs: COLMAP sparse reconstruction files.
 * Outputs: standardized calibration result.
 * Ownership: preprocessing adapter state.
 * Differentiable: NO.
 * TODO(NeuroDIC): parse cameras/images/points without leaking COLMAP format downstream.
 */
#pragma once

#include "neurodic/calibration/calibration_result.hpp"

namespace neurodic {

class COLMAPCalibration {
public:
    CalibrationResult run() const;
};

}  // namespace neurodic
