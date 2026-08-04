#include "neurodic/interpolation/torch_bspline.hpp"

#include <cassert>

void test_autograd_bspline() {
    auto coefficients = torch::randn({5, 6, 4, 4}, torch::kFloat64);
    auto coordinates = torch::tensor({{1.25, 2.4}, {3.6, 1.2}},
        torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    neurodic::TorchBSplineInterpolator sampler(3);
    sampler.evaluate(coefficients, coordinates).sum().backward();
    assert(coordinates.grad().defined());
    assert(torch::allclose(coordinates.grad(), sampler.gradient(coefficients, coordinates), 1e-9, 1e-9));
}
