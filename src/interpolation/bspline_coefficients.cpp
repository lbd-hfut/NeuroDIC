#include "neurodic/interpolation/bspline_coefficients.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"

namespace neurodic {

torch::Tensor compute_bspline_coefficients(const torch::Tensor& image, int degree) {
    validate_bspline_degree(degree);
    if (!image.defined()) {
        throw ValidationError("B-spline coefficient input image must be defined");
    }
    throw NotImplementedScientificError(
        "TODO(NeuroDIC): implement B-spline coefficient preprocessing under torch::NoGradGuard"
    );
}

}  // namespace neurodic
