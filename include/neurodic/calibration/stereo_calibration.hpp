/** Zhang stereo calibration preprocessing, translated from Traditional-DIC. */
#pragma once

#include "neurodic/calibration/mono_calibration.hpp"

namespace neurodic {

struct StereoCalibrationOptions : MonoCalibrationOptions {
    bool fix_intrinsics{false};
};

class StereoCalibration {
public:
    // All point tensors are CPU [views, points, dim], with shared board layout.
    CalibrationResult run_from_points(const torch::Tensor& object_points,
                                      const torch::Tensor& left_image_points,
                                      const torch::Tensor& right_image_points,
                                      int image_width, int image_height,
                                      const StereoCalibrationOptions& options = {}) const;
};

}  // namespace neurodic
