#include <cassert>

#include "neurodic/postprocess/strain.hpp"
#include "neurodic/postprocess/filtering.hpp"

void test_postprocess() {
    // u_x = 0.1 X: Green--Lagrange E_xx = ((1 + 0.1)^2 - 1) / 2.
    auto points_2d = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}});
    auto strain_2d = neurodic::compute_neural_strain_2d(
        [](const torch::Tensor& x) { return torch::stack({0.1F * x.select(1, 0), x.select(1, 1) * 0.0F}, 1); },
        points_2d);
    assert(torch::allclose(strain_2d.select(1, 0), torch::full({3}, 0.105F), 1e-6, 1e-6));
    assert(strain_2d.select(1, 1).abs().max().item<float>() < 1e-6F);

    // The physical scaling recovery must give the same derivative when both
    // coordinate and displacement normalizations are changed.
    auto scaled = neurodic::compute_neural_strain_2d(
        [](const torch::Tensor& x) { return torch::stack({0.2F * x.select(1, 0), x.select(1, 1) * 0.0F}, 1); },
        points_2d, torch::tensor({2.0F, 1.0F}), torch::tensor({1.0F, 1.0F}));
    assert(torch::allclose(scaled, strain_2d, 1e-6, 1e-6));

    auto axis = torch::tensor({0.0, 1.0}, torch::kFloat64);
    auto grid = torch::cartesian_prod({axis, axis, axis});
    auto displacement = torch::stack({0.1 * grid.select(1, 0), grid.select(1, 1) * 0.0,
                                      grid.select(1, 2) * 0.0}, 1);
    auto strain_3d = neurodic::compute_traditional_strain_3d(grid, displacement);
    auto finite = torch::isfinite(strain_3d).all(1);
    assert(finite.all().item<bool>());
    assert(torch::allclose(strain_3d.index({finite}).select(1, 0),
                           torch::full({finite.sum().item<int64_t>()}, 0.105, torch::kFloat64), 1e-8, 1e-8));

    auto plane_x = torch::arange(5, torch::kFloat64), plane_y = torch::arange(5, torch::kFloat64);
    auto plane = torch::cartesian_prod({plane_x, plane_y});
    plane = torch::cat({plane, torch::zeros({plane.size(0), 1}, torch::kFloat64)}, 1);
    auto contaminated = torch::cat({plane, torch::tensor({{100.0, 100.0, 100.0}}, torch::kFloat64)}, 0);
    auto cleaned = neurodic::clean_pin_multi_surface(contaminated, 4, 5.0);
    assert(cleaned.inlier_mask.slice(0, 0, plane.size(0)).all().item<bool>());
    assert(!cleaned.inlier_mask[-1].item<bool>());
}
