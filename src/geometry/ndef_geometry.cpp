#include "neurodic/geometry/ndef_geometry.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace {
torch::Tensor stack_camera_tensor(const std::vector<CameraModel>& cameras, const char* member,
                                  const torch::Device& device, torch::ScalarType dtype) {
    std::vector<torch::Tensor> values;
    values.reserve(cameras.size());
    for (const auto& camera : cameras) {
        if (std::string(member) == "K") values.push_back(camera.intrinsics.to(device, dtype));
        else if (std::string(member) == "R") values.push_back(camera.rotation.to(device, dtype));
        else if (std::string(member) == "t") values.push_back(camera.translation.to(device, dtype));
        else values.push_back(camera.distortion.to(device, dtype));
    }
    return torch::stack(values, 0);
}
}  // namespace

NDeFGeometry::NDeFGeometry(std::vector<CameraModel> cameras) : cameras_(std::move(cameras)) {
    if (cameras_.empty()) throw ValidationError("NDeF geometry requires at least one camera");
    const auto distortion_count = cameras_.front().distortion.numel();
    for (const auto& camera : cameras_) {
        camera.validate();
        if (camera.distortion.numel() != distortion_count)
            throw ValidationError("NDeF cameras must share one distortion coefficient count");
    }
}

MultiViewProjectionResult NDeFGeometry::project_reference_surface(const torch::Tensor& surface) const {
    auto K = stack_camera_tensor(cameras_, "K", surface.device(), surface.scalar_type());
    auto R = stack_camera_tensor(cameras_, "R", surface.device(), surface.scalar_type());
    auto t = stack_camera_tensor(cameras_, "t", surface.device(), surface.scalar_type());
    auto d = stack_camera_tensor(cameras_, "d", surface.device(), surface.scalar_type());
    return project_points_multi_view(surface, K, R, t, d);
}

MultiViewProjectionResult NDeFGeometry::project_deformed_surface(const torch::Tensor& surface,
                                                                  const torch::Tensor& deformation) const {
    if (surface.sizes() != deformation.sizes()) throw ValidationError("NDeF surface/deformation must have matching [N,3]");
    return project_reference_surface(surface + deformation);
}

torch::Tensor NDeFGeometry::visibility(const torch::Tensor& surface) const {
    auto projected = project_reference_surface(surface);
    auto result = projected.depth > 0.0;
    for (size_t view = 0; view < cameras_.size(); ++view) {
        const auto& camera = cameras_[view];
        auto u = projected.uv.select(1, static_cast<int64_t>(view)).select(1, 0);
        auto v = projected.uv.select(1, static_cast<int64_t>(view)).select(1, 1);
        result.select(1, static_cast<int64_t>(view)).copy_(result.select(1, static_cast<int64_t>(view)) &
            (u >= 0.0) & (u < camera.image_width) & (v >= 0.0) & (v < camera.image_height));
    }
    return result;
}

}  // namespace neurodic
