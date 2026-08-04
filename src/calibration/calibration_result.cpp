#include "neurodic/calibration/calibration_result.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void CalibrationResult::validate() const {
    if (type == CalibrationType::NONE) {
        if (!cameras.empty()) throw ValidationError("CalibrationType::NONE cannot contain cameras");
        return;
    }
    if (cameras.empty()) throw ValidationError("Calibration result must contain at least one camera");
    for (const auto& camera : cameras) camera.validate();
    if (type == CalibrationType::STEREO) {
        if (cameras.size() != 2 || !stereo_rotation.defined() || !stereo_translation.defined() ||
            stereo_rotation.sizes() != torch::IntArrayRef({3, 3}) || stereo_translation.sizes() != torch::IntArrayRef({3}) ||
            stereo_rotation.device().is_cuda() || stereo_translation.device().is_cuda()) {
            throw ValidationError("Stereo calibration requires two cameras and CPU R_lr[3,3], t_lr[3]");
        }
    }
}

}  // namespace neurodic
