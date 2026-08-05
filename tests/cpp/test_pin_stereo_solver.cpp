#include <cassert>

#include "neurodic/problem/pin_stereo_problem.hpp"
#include "neurodic/solver/pin_stereo_solver.hpp"

namespace {
neurodic::CameraModel camera(double tx) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    neurodic::CameraModel value;
    value.intrinsics = torch::tensor({{800.0, 0.0, 320.0}, {0.0, 800.0, 240.0}, {0.0, 0.0, 1.0}}, options);
    value.rotation = torch::eye(3, options);
    value.translation = torch::tensor({tx, 0.0, 0.0}, options);
    value.distortion = torch::zeros({5}, options);
    value.image_width = 640;
    value.image_height = 480;
    return value;
}

neurodic::PINProblem dummy_problem() {
    auto image = torch::zeros({8, 8}, torch::kFloat32);
    auto mask = torch::ones({8, 8}, torch::kBool);
    auto pos = torch::tensor({{2.0F, 2.0F}, {5.0F, 5.0F}});
    auto uv = torch::zeros({2, 2});
    return neurodic::PINProblem(image, image.clone(), mask, neurodic::SeedSet::from_tensors(pos, uv));
}

neurodic::PINResult field(const torch::Tensor& xy, const torch::Tensor& uv) {
    neurodic::PINResult result;
    result.displacement = {xy.to(torch::kFloat32), uv.to(torch::kFloat32)};
    return result;
}
}  // namespace

void test_pin_stereo_solver() {
    auto left = camera(0.0);
    auto right = camera(-0.2);
    auto p = dummy_problem();
    neurodic::PINStereoProblem problem(p, p, p, left, right);
    problem.require_image_bounds = true;
    problem.reconstruction.max_reprojection_error = 1e-5;
    auto l0 = torch::tensor({{320.0, 240.0}, {360.0, 256.0}}, torch::kFloat64);
    auto r0 = torch::tensor({{288.0, 240.0}, {328.0, 256.0}}, torch::kFloat64);
    auto l1 = l0 + torch::tensor({{8.0, 0.0}, {8.0, 0.0}}, torch::kFloat64);
    auto r1 = r0 + torch::tensor({{8.0, 0.0}, {8.0, 0.0}}, torch::kFloat64);
    auto result = neurodic::PINStereoSolver().reconstruct(
        field(l0, r0 - l0), field(l0, l1 - l0), field(l0, r1 - l0), problem);
    assert(result.valid.all().item<bool>());
    assert(torch::allclose(result.reference_points.select(1, 2), torch::full({2}, 5.0, torch::kFloat64), 1e-8, 1e-8));
    assert(torch::allclose(result.displacement_3d.select(1, 0), torch::full({2}, 0.05, torch::kFloat64), 1e-7, 1e-7));
    assert(torch::allclose(result.displacement_3d.select(1, 2), torch::zeros({2}, torch::kFloat64), 1e-7, 1e-7));
}
