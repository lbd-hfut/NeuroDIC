#include <cassert>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"
#include "neurodic/interpolation/torch_bspline.hpp"

void test_bspline() {
    assert(neurodic::is_supported_bspline_degree(1));
    assert(neurodic::is_supported_bspline_degree(3));
    assert(neurodic::is_supported_bspline_degree(5));
    assert(!neurodic::is_supported_bspline_degree(2));

    neurodic::TorchBSplineInterpolator sampler(5);
    assert(sampler.degree() == 5);
}
