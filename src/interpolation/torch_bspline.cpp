#include "neurodic/interpolation/torch_bspline.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"

namespace neurodic {
namespace {
void validate_inputs(const torch::Tensor& coefficients, const torch::Tensor& coordinates, int degree) {
    if (!coefficients.defined() || coefficients.dim() != 4 ||
        coefficients.size(2) != degree + 1 || coefficients.size(3) != degree + 1)
        throw ValidationError("B-spline coefficients must have shape [H,W,degree+1,degree+1]");
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != 2 ||
        !coordinates.is_floating_point())
        throw ValidationError("B-spline coordinates must be floating [N,2]");
    if (coefficients.device() != coordinates.device())
        throw ValidationError("B-spline coefficients and coordinates must share a device");
    if (coefficients.scalar_type() != coordinates.scalar_type())
        throw ValidationError("B-spline coefficients and coordinates must share a dtype");
}

torch::Tensor powers(const torch::Tensor& values, int degree) {
    auto exponent = torch::arange(degree + 1, values.options());
    return torch::pow(values.unsqueeze(1), exponent.unsqueeze(0));
}
}  // namespace

TorchBSplineInterpolator::TorchBSplineInterpolator(int degree) : degree_(degree) {
    validate_bspline_degree(degree_);
}

torch::Tensor TorchBSplineInterpolator::evaluate(
    const torch::Tensor& coefficients, const torch::Tensor& coordinates) const {
    validate_inputs(coefficients, coordinates, degree_);
    auto floor_xy = torch::floor(coordinates);
    auto ix = floor_xy.select(1, 0).to(torch::kLong).clamp(0, coefficients.size(1) - 1);
    auto iy = floor_xy.select(1, 1).to(torch::kLong).clamp(0, coefficients.size(0) - 1);
    auto block = coefficients.index({iy, ix});
    auto dx = coordinates.select(1, 0) - ix.to(coordinates.scalar_type());
    auto dy = coordinates.select(1, 1) - iy.to(coordinates.scalar_type());
    auto xp = powers(dx, degree_);
    auto yp = powers(dy, degree_);
    return torch::einsum("ni,nij,nj->n", {yp, block, xp});
}

torch::Tensor TorchBSplineInterpolator::gradient(
    const torch::Tensor& coefficients, const torch::Tensor& coordinates) const {
    validate_inputs(coefficients, coordinates, degree_);
    auto floor_xy = torch::floor(coordinates);
    auto ix = floor_xy.select(1, 0).to(torch::kLong).clamp(0, coefficients.size(1) - 1);
    auto iy = floor_xy.select(1, 1).to(torch::kLong).clamp(0, coefficients.size(0) - 1);
    auto block = coefficients.index({iy, ix});
    auto dx = coordinates.select(1, 0) - ix.to(coordinates.scalar_type());
    auto dy = coordinates.select(1, 1) - iy.to(coordinates.scalar_type());
    auto xp = powers(dx, degree_);
    auto yp = powers(dy, degree_);
    auto orders = torch::arange(1, degree_ + 1, coordinates.options());
    auto dxp = powers(dx, degree_ - 1) * orders.unsqueeze(0);
    auto dyp = powers(dy, degree_ - 1) * orders.unsqueeze(0);
    auto gx = torch::einsum("ni,nij,nj->n", {yp, block.index({torch::indexing::Slice(),
        torch::indexing::Slice(), torch::indexing::Slice(1, degree_ + 1)}), dxp});
    auto gy = torch::einsum("ni,nij,nj->n", {dyp, block.index({torch::indexing::Slice(),
        torch::indexing::Slice(1, degree_ + 1), torch::indexing::Slice()}), xp});
    return torch::stack({gx, gy}, 1);
}

}  // namespace neurodic
