#include "neurodic/calibration/calibration_result.hpp"

#include "neurodic/core/exceptions.hpp"

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
#include "neurodic/calibration/multiview_calibration.hpp"
#endif

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

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
namespace neurodic::calibration {
namespace {

torch::Tensor matrix_tensor(const Eigen::Matrix3d& matrix) {
    auto tensor = torch::empty({3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto values = tensor.accessor<double, 2>();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) values[row][col] = matrix(row, col);
    }
    return tensor;
}

torch::Tensor vector_tensor(const Eigen::Vector3d& vector) {
    auto tensor = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto values = tensor.accessor<double, 1>();
    for (int index = 0; index < 3; ++index) values[index] = vector(index);
    return tensor;
}

torch::Tensor distortion_tensor(const std::vector<double>& distortion) {
    auto tensor = torch::empty({static_cast<int64_t>(distortion.size())},
                               torch::TensorOptions().dtype(torch::kFloat64));
    auto values = tensor.accessor<double, 1>();
    for (size_t index = 0; index < distortion.size(); ++index) values[index] = distortion[index];
    return tensor;
}

}  // namespace

::neurodic::CameraModel to_core_camera_model(const CameraModel& camera) {
    ::neurodic::CameraModel converted;
    converted.intrinsics = matrix_tensor(camera.K);
    converted.rotation = matrix_tensor(camera.R);
    converted.translation = vector_tensor(camera.t);
    converted.distortion = distortion_tensor(camera.distortion);
    converted.image_width = camera.image_width;
    converted.image_height = camera.image_height;
    converted.rms_error = camera.rms_error;
    converted.label = camera.label;
    converted.validate();
    return converted;
}

::neurodic::CalibrationResult to_core_calibration_result(
    const MultiviewCalibrationResult& reconstruction) {
    ::neurodic::CalibrationResult converted;
    converted.type = ::neurodic::CalibrationType::COLMAP;
    converted.rms_error = reconstruction.mean_reprojection_error;
    converted.cameras.reserve(reconstruction.cameras.size());
    for (const auto& camera : reconstruction.cameras) {
        converted.cameras.push_back(to_core_camera_model(camera));
    }
    converted.validate();
    return converted;
}

}  // namespace neurodic::calibration
#endif
