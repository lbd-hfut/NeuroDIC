/** Standardized immutable calibration output consumed by problem builders, never solvers. */
#pragma once

#include <vector>

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/core/types.hpp"

namespace neurodic {

struct CalibrationResult {
    CalibrationType type{CalibrationType::NONE};
    std::vector<CameraModel> cameras;
    torch::Tensor stereo_rotation;    // optional [3,3], left -> right
    torch::Tensor stereo_translation; // optional [3], left -> right
    double rms_error{0.0};

    void validate() const;
};

}  // namespace neurodic

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
namespace neurodic::calibration {

struct CameraModel;
struct MultiviewCalibrationResult;

// Explicit conversion boundary from Eigen/OpenCV SfM to Torch core geometry.
::neurodic::CameraModel to_core_camera_model(const CameraModel& camera);
::neurodic::CalibrationResult to_core_calibration_result(
    const MultiviewCalibrationResult& reconstruction);

}  // namespace neurodic::calibration
#endif
