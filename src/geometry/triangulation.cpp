#include "neurodic/geometry/triangulation.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/geometry/projection.hpp"

namespace neurodic {
namespace {
torch::Tensor undistort_pixels(const torch::Tensor& pixels, const CameraModel& camera, int iterations) {
    auto fx = camera.intrinsics.index({0, 0}), fy = camera.intrinsics.index({1, 1});
    auto cx = camera.intrinsics.index({0, 2}), cy = camera.intrinsics.index({1, 2});
    auto x = (pixels.select(1, 0) - cx) / fx;
    auto y = (pixels.select(1, 1) - cy) / fy;
    auto xu = x.clone(), yu = y.clone();
    auto coefficient = [&](int index) {
        return index < camera.distortion.numel() ? camera.distortion[index] :
            torch::zeros({}, camera.distortion.options());
    };
    for (int step = 0; step < iterations; ++step) {
        auto r2 = xu * xu + yu * yu;
        auto radial = 1.0 + coefficient(0) * r2 + coefficient(1) * r2 * r2 + coefficient(4) * r2 * r2 * r2;
        auto tx = 2.0 * coefficient(2) * xu * yu + coefficient(3) * (r2 + 2.0 * xu * xu);
        auto ty = coefficient(2) * (r2 + 2.0 * yu * yu) + 2.0 * coefficient(3) * xu * yu;
        xu = (x - tx) / radial;
        yu = (y - ty) / radial;
    }
    return torch::stack({xu, yu}, 1);
}
}  // namespace

ReconstructionResult triangulate_multiview(const torch::Tensor& observations, const std::vector<CameraModel>& cameras,
                                           const torch::Tensor& observation_valid, ReconstructionOptions options) {
    if (!observations.defined() || observations.dim() != 3 || observations.size(2) != 2 || !observations.is_floating_point() ||
        observations.device().is_cuda() || observations.scalar_type() != torch::kFloat64 ||
        observations.size(1) != static_cast<int64_t>(cameras.size()) || cameras.size() < 2 ||
        options.max_reprojection_error < 0.0 || options.undistort_iterations < 0)
        throw ValidationError("CPU reconstruction expects observations[N,V,2] float64, matching at least two cameras");
    for (const auto& camera : cameras) camera.validate();
    const auto count = observations.size(0), views = observations.size(1);
    auto valid = observation_valid.defined() ? observation_valid.detach().to(torch::kCPU).to(torch::kBool).contiguous() :
        torch::ones({count, views}, torch::TensorOptions().dtype(torch::kBool));
    if (valid.sizes() != torch::IntArrayRef({count, views})) throw ValidationError("Observation validity must be [N,V]");
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> rows;
    rows.reserve(static_cast<size_t>(views));
    for (int64_t view = 0; view < views; ++view) {
        const auto& camera = cameras[static_cast<size_t>(view)];
        auto normalized = undistort_pixels(observations.select(1, view), camera, options.undistort_iterations);
        auto extrinsic = torch::cat({camera.rotation, camera.translation.unsqueeze(1)}, 1);
        auto row_x = normalized.select(1, 0).unsqueeze(1) * extrinsic.select(0, 2) - extrinsic.select(0, 0);
        auto row_y = normalized.select(1, 1).unsqueeze(1) * extrinsic.select(0, 2) - extrinsic.select(0, 1);
        rows.push_back(torch::stack({row_x, row_y}, 1) * valid.select(1, view).unsqueeze(1).unsqueeze(2));
    }
    auto system = torch::stack(rows, 1).reshape({count, 2 * views, 4});
    auto ata = torch::matmul(system.transpose(1, 2), system);
    auto eigen = torch::linalg_eigh(ata);
    auto homogeneous = std::get<1>(eigen).select(2, 0);
    auto w = homogeneous.select(1, 3);
    auto points = homogeneous.slice(1, 0, 3) / w.unsqueeze(1);
    auto used = valid.sum(1).to(torch::kLong);
    auto point_valid = (used >= 2) & torch::isfinite(points).all(1) & (torch::abs(w) > 1e-12);
    auto errors = torch::zeros({count, views}, observations.options());
    for (int64_t view = 0; view < views; ++view) {
        const auto& camera = cameras[static_cast<size_t>(view)];
        auto projection = project_points_with_depth(points, camera.intrinsics, camera.rotation, camera.translation,
                                                    camera.distortion);
        errors.select(1, view).copy_(torch::linalg_vector_norm(projection.uv - observations.select(1, view), 2, 1));
        if (options.require_positive_depth) point_valid = point_valid &
            torch::where(valid.select(1, view), projection.depth > 0.0, torch::ones_like(point_valid));
    }
    auto weighted_errors = torch::where(valid, errors, torch::zeros_like(errors));
    auto mean_error = weighted_errors.sum(1) / used.to(torch::kFloat64).clamp_min(1.0);
    auto max_error = std::get<0>(torch::where(valid, errors, torch::zeros_like(errors)).max(1));
    point_valid = point_valid & (max_error <= options.max_reprojection_error);
    return {points, point_valid, mean_error, max_error, used};
}

ReconstructionResult triangulate_stereo(const torch::Tensor& left_coordinates, const torch::Tensor& right_coordinates,
                                        const CameraModel& left_camera, const CameraModel& right_camera,
                                        ReconstructionOptions options) {
    if (left_coordinates.dim() != 2 || left_coordinates.size(1) != 2 || right_coordinates.sizes() != left_coordinates.sizes())
        throw ValidationError("Stereo reconstruction expects matching left/right [N,2] coordinates");
    return triangulate_multiview(torch::stack({left_coordinates, right_coordinates}, 1),
                                 {left_camera, right_camera}, {}, options);
}

}  // namespace neurodic
