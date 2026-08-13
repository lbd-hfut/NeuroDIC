#include "neurodic/postprocess/strain.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/postprocess/filtering.hpp"

namespace neurodic {
namespace {

torch::Tensor scales(const torch::Tensor& scale, int64_t dimensions,
                     const torch::TensorOptions& options, const char* name) {
    if (!scale.defined()) return torch::ones({dimensions}, options);
    if (scale.numel() != dimensions)
        throw ValidationError(std::string(name) + " must contain one positive value per component");
    auto result = scale.detach().to(options.device()).to(options.dtype()).reshape({dimensions});
    if (!(result > 0).all().item<bool>())
        throw ValidationError(std::string(name) + " must be strictly positive");
    return result;
}

torch::Tensor green_lagrange(const torch::Tensor& gradient) {
    const auto dimensions = gradient.size(1);
    auto identity = torch::eye(dimensions, gradient.options()).unsqueeze(0);
    auto deformation_gradient = identity + gradient;
    auto strain = 0.5 * (torch::matmul(deformation_gradient.transpose(1, 2), deformation_gradient) - identity);
    if (dimensions == 2)
        return torch::stack({strain.select(1, 0).select(1, 0), strain.select(1, 1).select(1, 1),
                             strain.select(1, 0).select(1, 1)}, 1);
    return torch::stack({strain.select(1, 0).select(1, 0), strain.select(1, 1).select(1, 1),
                         strain.select(1, 2).select(1, 2), strain.select(1, 0).select(1, 1),
                         strain.select(1, 1).select(1, 2), strain.select(1, 0).select(1, 2)}, 1);
}

torch::Tensor neural_strain(const std::function<torch::Tensor(const torch::Tensor&)>& deformation,
                            const torch::Tensor& coordinates, int64_t dimensions,
                            const torch::Tensor& coordinate_scale, const torch::Tensor& displacement_scale) {
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != dimensions)
        throw ValidationError("Neural strain coordinates must have shape [N,D]");
    auto points = coordinates.detach().clone().set_requires_grad(true);
    auto displacement = deformation(points);
    if (!displacement.defined() || displacement.sizes() != points.sizes())
        throw ValidationError("Neural deformation must return a [N,D] tensor matching its coordinates");
    std::vector<torch::Tensor> derivatives;
    derivatives.reserve(static_cast<size_t>(dimensions));
    for (int64_t component = 0; component < dimensions; ++component) {
        auto derivative = torch::autograd::grad({displacement.select(1, component).sum()}, {points}, {}, true, false,
                                                true)[0];
        if (!derivative.defined()) derivative = torch::zeros_like(points);
        derivatives.push_back(derivative);
    }
    // rows are displacement components; columns are coordinate components.
    auto gradient = torch::stack(derivatives, 1);
    auto coordinate_units = scales(coordinate_scale, dimensions, points.options(), "coordinate_scale");
    auto displacement_units = scales(displacement_scale, dimensions, points.options(), "displacement_scale");
    gradient = gradient * displacement_units.reshape({1, dimensions, 1}) /
               coordinate_units.reshape({1, 1, dimensions});
    return green_lagrange(gradient).detach();
}

}  // namespace

torch::Tensor compute_neural_strain_2d(const std::function<torch::Tensor(const torch::Tensor&)>& deformation,
                                       const torch::Tensor& coordinates, const torch::Tensor& coordinate_scale,
                                       const torch::Tensor& displacement_scale) {
    return neural_strain(deformation, coordinates, 2, coordinate_scale, displacement_scale);
}

torch::Tensor compute_neural_strain_3d(const std::function<torch::Tensor(const torch::Tensor&)>& deformation,
                                       const torch::Tensor& coordinates, const torch::Tensor& coordinate_scale,
                                       const torch::Tensor& displacement_scale) {
    return neural_strain(deformation, coordinates, 3, coordinate_scale, displacement_scale);
}

torch::Tensor compute_traditional_strain_3d(const torch::Tensor& coordinates, const torch::Tensor& displacement,
                                            const torch::Tensor& valid, int64_t neighbors,
                                            const torch::Tensor& coordinate_scale,
                                            const torch::Tensor& displacement_scale) {
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != 3 ||
        !displacement.defined() || displacement.sizes() != coordinates.sizes())
        throw ValidationError("Traditional 3D strain expects matching [N,3] coordinates and displacement");
    if (neighbors < 3) throw ValidationError("Traditional 3D strain requires at least three neighbours");
    const auto input_options = coordinates.options();
    auto coordinate_units = scales(coordinate_scale, 3, input_options, "coordinate_scale");
    auto displacement_units = scales(displacement_scale, 3, input_options, "displacement_scale");
    auto points = (coordinates.detach().to(torch::kCPU).to(torch::kFloat64) * coordinate_units.cpu().to(torch::kFloat64));
    auto values = (displacement.detach().to(torch::kCPU).to(torch::kFloat64) * displacement_units.cpu().to(torch::kFloat64));
    auto usable = valid.defined() ? valid.detach().to(torch::kCPU).to(torch::kBool).reshape({-1})
                                  : torch::isfinite(points).all(1) & torch::isfinite(values).all(1);
    if (usable.numel() != points.size(0)) throw ValidationError("Traditional 3D strain valid mask must be [N]");
    usable = usable & torch::isfinite(points).all(1) & torch::isfinite(values).all(1);
    const auto count = points.size(0);
    auto result = torch::full({count, 6}, std::numeric_limits<double>::quiet_NaN(), points.options());
    const auto usable_ids = torch::nonzero(usable).reshape({-1});
    if (usable_ids.numel() < 4) return result.to(input_options.device()).to(input_options.dtype());
    auto usable_points = points.index_select(0, usable_ids);
    auto usable_values = values.index_select(0, usable_ids);
    const auto k = std::min<int64_t>(neighbors, usable_ids.numel() - 1);
    if (k < 3) return result.to(input_options.device()).to(input_options.dtype());
    auto nearest = knn_indices_3d(usable_points, k);
    for (int64_t row = 0; row < usable_ids.numel(); ++row) {
        auto ids = nearest.select(0, row);
        auto dx = usable_points.index_select(0, ids) - usable_points[row];
        auto du = usable_values.index_select(0, ids) - usable_values[row];
        auto d2 = dx.square().sum(1);
        auto weights = 1.0 / d2.clamp_min(1e-16);
        auto weighted_dx = dx * weights.sqrt().unsqueeze(1);
        auto weighted_du = du * weights.sqrt().unsqueeze(1);
        // lstsq returns dU/dX transposed: [coordinate component, displacement component].
        auto solution = std::get<0>(torch::linalg_lstsq(weighted_dx, weighted_du));
        if (!torch::isfinite(solution).all().item<bool>()) continue;
        result.index_put_({usable_ids[row]}, green_lagrange(solution.transpose(0, 1).unsqueeze(0)).squeeze(0));
    }
    return result.to(input_options.device()).to(input_options.dtype());
}

}  // namespace neurodic
