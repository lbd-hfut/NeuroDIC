#include "neurodic/calibration/camera_model.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void CameraModel::validate() const {
    const auto valid_cpu_f64 = [](const torch::Tensor& value) {
        return value.defined() && value.device().is_cpu() && value.scalar_type() == torch::kFloat64;
    };
    if (!valid_cpu_f64(intrinsics) || intrinsics.sizes() != torch::IntArrayRef({3, 3}) ||
        !valid_cpu_f64(rotation) || rotation.sizes() != torch::IntArrayRef({3, 3}) ||
        !valid_cpu_f64(translation) || translation.sizes() != torch::IntArrayRef({3}) ||
        !valid_cpu_f64(distortion) || distortion.dim() != 1 || image_width <= 0 || image_height <= 0) {
        throw ValidationError("CameraModel requires CPU float64 K[3,3], R[3,3], t[3], distortion[N], and image size");
    }
}

torch::Tensor CameraModel::projection_matrix() const {
    validate();
    return torch::matmul(intrinsics, torch::cat({rotation, translation.unsqueeze(1)}, 1));
}

torch::Tensor CameraModel::camera_center() const {
    validate();
    return -torch::matmul(rotation.transpose(0, 1), translation);
}

}  // namespace neurodic

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
namespace neurodic::calibration {

Eigen::Matrix<double, 3, 4> CameraModel::projection_matrix() const {
    Eigen::Matrix<double, 3, 4> extrinsic;
    extrinsic.block<3, 3>(0, 0) = R;
    extrinsic.col(3) = t;
    return K * extrinsic;
}

Eigen::Vector3d CameraModel::camera_center() const { return -R.transpose() * t; }

}  // namespace neurodic::calibration
#endif
