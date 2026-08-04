/** Zhang mono calibration preprocessing, translated from Traditional-DIC. */
#pragma once

#include "neurodic/calibration/calibration_result.hpp"

namespace neurodic {

struct MonoCalibrationOptions {
    bool estimate_tangential_distortion{true};
    bool estimate_k3{true};
    int max_iterations{100};
    double epsilon{1e-9};
};

class MonoCalibration {
public:
    // object_points [views, points, 3], image_points [views, points, 2], CPU.
    CalibrationResult run_from_points(const torch::Tensor& object_points,
                                      const torch::Tensor& image_points,
                                      int image_width, int image_height,
                                      const MonoCalibrationOptions& options = {}) const;
};

}  // namespace neurodic
