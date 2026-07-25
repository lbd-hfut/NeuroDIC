/**
 * Calibration manager.
 *
 * Responsibilities: dispatch preprocessing calibration adapters.
 * Inputs: calibration type and external data paths/options.
 * Outputs: CalibrationResult.
 * Ownership: stateless shell.
 * Differentiable: NO. Calibration is preprocessing and never runs inside solvers.
 * TODO(NeuroDIC): define typed request objects for mono/stereo/COLMAP workflows.
 */
#pragma once

#include "neurodic/calibration/calibration_result.hpp"

namespace neurodic {

class CalibrationManager {
public:
    CalibrationResult calibrate(CalibrationType type) const;
};

}  // namespace neurodic
