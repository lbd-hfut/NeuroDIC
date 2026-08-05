#include "neurodic/geometry/projection.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace {
void validate_camera_tensors(const torch::Tensor& points, const torch::Tensor& intrinsics,
                             const torch::Tensor& rotation, const torch::Tensor& translation,
                             const torch::Tensor& distortion) {
    if (!points.defined() || points.dim() != 2 || points.size(1) != 3 || !points.is_floating_point() ||
        intrinsics.sizes() != torch::IntArrayRef({3, 3}) || rotation.sizes() != torch::IntArrayRef({3, 3}) ||
        translation.sizes() != torch::IntArrayRef({3}) || points.device() != intrinsics.device() ||
        points.device() != rotation.device() || points.device() != translation.device() ||
        points.scalar_type() != intrinsics.scalar_type() || points.scalar_type() != rotation.scalar_type() ||
        points.scalar_type() != translation.scalar_type() ||
        (distortion.defined() && (distortion.dim() != 1 || distortion.device() != points.device() ||
                                  distortion.scalar_type() != points.scalar_type())))
        throw ValidationError("Projection expects floating points[N,3], K/R[3,3], t[3] on one device and dtype");
}

torch::Tensor distort_normalized(const torch::Tensor& normalized, const torch::Tensor& distortion) {
    if (!distortion.defined() || distortion.numel() == 0) return normalized;
    auto coefficient = [&](int index) {
        return index < distortion.numel() ? distortion[index] : torch::zeros({}, distortion.options());
    };
    auto x = normalized.select(1, 0);
    auto y = normalized.select(1, 1);
    auto r2 = x * x + y * y;
    auto radial = 1.0 + coefficient(0) * r2 + coefficient(1) * r2 * r2 + coefficient(4) * r2 * r2 * r2;
    auto xd = x * radial + 2.0 * coefficient(2) * x * y + coefficient(3) * (r2 + 2.0 * x * x);
    auto yd = y * radial + coefficient(2) * (r2 + 2.0 * y * y) + 2.0 * coefficient(3) * x * y;
    return torch::stack({xd, yd}, 1);
}
}  // namespace

torch::Tensor world_to_camera(const torch::Tensor& points, const torch::Tensor& rotation,
                              const torch::Tensor& translation) {
    if (!points.defined() || points.dim() != 2 || points.size(1) != 3 || rotation.sizes() != torch::IntArrayRef({3, 3}) ||
        translation.sizes() != torch::IntArrayRef({3}) || points.device() != rotation.device() ||
        points.device() != translation.device() || points.scalar_type() != rotation.scalar_type() ||
        points.scalar_type() != translation.scalar_type())
        throw ValidationError("world_to_camera expects matching points[N,3], R[3,3], and t[3]");
    return torch::matmul(points, rotation.transpose(0, 1)) + translation;
}

ProjectionResult project_points_with_depth(const torch::Tensor& points, const torch::Tensor& intrinsics,
                                           const torch::Tensor& rotation, const torch::Tensor& translation,
                                           const torch::Tensor& distortion) {
    validate_camera_tensors(points, intrinsics, rotation, translation, distortion);
    auto camera_points = world_to_camera(points, rotation, translation);
    auto depth = camera_points.select(1, 2);
    auto normalized = camera_points.slice(1, 0, 2) / depth.unsqueeze(1);
    auto distorted = distort_normalized(normalized, distortion);
    auto homogeneous = torch::cat({distorted, torch::ones({points.size(0), 1}, points.options())}, 1);
    auto pixel = torch::matmul(homogeneous, intrinsics.transpose(0, 1));
    return {pixel.slice(1, 0, 2) / pixel.select(1, 2).unsqueeze(1), depth};
}

torch::Tensor project_points(const torch::Tensor& points, const torch::Tensor& intrinsics,
                             const torch::Tensor& rotation, const torch::Tensor& translation,
                             const torch::Tensor& distortion) {
    return project_points_with_depth(points, intrinsics, rotation, translation, distortion).uv;
}

torch::Tensor project_points(const torch::Tensor& points, const torch::Tensor& projection_matrix) {
    if (!points.defined() || points.dim() != 2 || points.size(1) != 3 || projection_matrix.sizes() != torch::IntArrayRef({3, 4}) ||
        points.device() != projection_matrix.device() || points.scalar_type() != projection_matrix.scalar_type())
        throw ValidationError("Projection matrix path expects matching points[N,3] and P[3,4]");
    auto homogeneous = torch::cat({points, torch::ones({points.size(0), 1}, points.options())}, 1);
    auto pixel = torch::matmul(homogeneous, projection_matrix.transpose(0, 1));
    return pixel.slice(1, 0, 2) / pixel.select(1, 2).unsqueeze(1);
}

MultiViewProjectionResult project_points_multi_view(const torch::Tensor& points, const torch::Tensor& intrinsics,
                                                     const torch::Tensor& rotations, const torch::Tensor& translations,
                                                     const torch::Tensor& distortions) {
    if (!points.defined() || points.dim() != 2 || points.size(1) != 3 || intrinsics.dim() != 3 ||
        intrinsics.size(1) != 3 || intrinsics.size(2) != 3 || rotations.sizes() != intrinsics.sizes() ||
        translations.dim() != 2 || translations.size(0) != intrinsics.size(0) || translations.size(1) != 3 ||
        points.device() != intrinsics.device() || points.device() != rotations.device() || points.device() != translations.device() ||
        points.scalar_type() != intrinsics.scalar_type() || points.scalar_type() != rotations.scalar_type() ||
        points.scalar_type() != translations.scalar_type())
        throw ValidationError("Multi-view projection expects points[N,3], K/R[V,3,3], t[V,3] on one device and dtype");
    const auto views = intrinsics.size(0);
    auto camera_points = torch::einsum("vij,nj->nvi", {rotations, points}) + translations.unsqueeze(0);
    auto depth = camera_points.select(2, 2);
    auto normalized = camera_points.slice(2, 0, 2) / depth.unsqueeze(2);
    torch::Tensor distorted = normalized;
    if (distortions.defined() && distortions.numel() > 0) {
        if (distortions.dim() != 2 || distortions.size(0) != views || distortions.device() != points.device() ||
            distortions.scalar_type() != points.scalar_type()) throw ValidationError("Distortions must be [V,D] on projection device");
        auto x = normalized.select(2, 0), y = normalized.select(2, 1), r2 = x * x + y * y;
        auto c = [&](int index) { return index < distortions.size(1) ? distortions.select(1, index).unsqueeze(0) : torch::zeros({1, views}, points.options()); };
        auto radial = 1.0 + c(0) * r2 + c(1) * r2 * r2 + c(4) * r2 * r2 * r2;
        distorted = torch::stack({x * radial + 2.0 * c(2) * x * y + c(3) * (r2 + 2.0 * x * x),
                                  y * radial + c(2) * (r2 + 2.0 * y * y) + 2.0 * c(3) * x * y}, 2);
    }
    auto homogeneous = torch::cat({distorted, torch::ones({points.size(0), views, 1}, points.options())}, 2);
    auto pixel = torch::einsum("vij,nvj->nvi", {intrinsics, homogeneous});
    return {pixel.slice(2, 0, 2) / pixel.select(2, 2).unsqueeze(2), depth};
}

}  // namespace neurodic
