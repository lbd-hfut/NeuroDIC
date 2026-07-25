#include "neurodic/interpolation/bspline.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

bool is_supported_bspline_degree(int degree) {
    return degree == 1 || degree == 3 || degree == 5;
}

void validate_bspline_degree(int degree) {
    if (!is_supported_bspline_degree(degree)) {
        throw ValidationError("B-spline degree must be 1, 3, or 5");
    }
}

}  // namespace neurodic
