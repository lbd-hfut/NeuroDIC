#include "neurodic/interpolation/torch_bspline.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"

namespace neurodic {

TorchBSplineInterpolator::TorchBSplineInterpolator(int degree) : degree_(degree) {
    validate_bspline_degree(degree_);
}

torch::Tensor TorchBSplineInterpolator::evaluate(
    const torch::Tensor& coefficients,
    const torch::Tensor& coordinates
) const {
    if (!coefficients.defined() || !coordinates.defined()) {
        throw ValidationError("TorchBSplineInterpolator inputs must be defined");
    }
    if (coordinates.dim() != 2 || coordinates.size(1) != 2) {
        throw ValidationError("B-spline coordinates must have shape [N, 2]");
    }
    throw NotImplementedScientificError(
        "TODO(NeuroDIC): implement differentiable LibTorch B-spline sampling without graph-breaking conversions"
    );
}

torch::Tensor TorchBSplineInterpolator::gradient(
    const torch::Tensor& coefficients,
    const torch::Tensor& coordinates
) const {
    if (!coefficients.defined() || !coordinates.defined()) {
        throw ValidationError("TorchBSplineInterpolator inputs must be defined");
    }
    if (coordinates.dim() != 2 || coordinates.size(1) != 2) {
        throw ValidationError("B-spline coordinates must have shape [N, 2]");
    }
    throw NotImplementedScientificError(
        "TODO(NeuroDIC): implement differentiable B-spline spatial gradients"
    );
}

}  // namespace neurodic
