#include <cassert>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/problem/pin_multi_problem.hpp"
#include "neurodic/solver/pin_multi_slover.hpp"

void test_pin_multi_validate();
void test_pin_multi_solve();

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

neurodic::PINMultiPairProblem pair_problem(const std::string& pair_id, double tx,
                                           neurodic::PINProblem problem) {
    neurodic::PINMultiPairProblem pair;
    pair.pair_id = pair_id;
    pair.reference_stereo = problem;
    pair.left_temporal = problem;
    pair.deformed_stereo = problem;
    pair.left_camera = camera(0.0);
    pair.right_camera = camera(tx);
    return pair;
}

neurodic::PINMultiPairProblem fast_pair_problem(const std::string& pair_id, double tx) {
    auto problem = dummy_problem();
    problem.seed_iterations = 2;
    problem.photometric_iterations = 2;
    return pair_problem(pair_id, tx, problem);
}

void expect_validation_error(const std::function<void()>& call) {
    bool thrown = false;
    try {
        call();
    } catch (const neurodic::ValidationError&) {
        thrown = true;
    }
    assert(thrown);
}
}  // namespace

void test_pin_multi_validate() {
    neurodic::PINMultiProblem problem;
    expect_validation_error([&] { problem.validate(); });

    problem.route_id = "ndef";
    expect_validation_error([&] { problem.validate(); });
    problem.route_id = "pin_multi_slover";

    problem.pairs.push_back(pair_problem("cam_0__cam_1", -0.2, dummy_problem()));
    problem.world_scale = 0.0;
    expect_validation_error([&] { problem.validate(); });
    problem.world_scale = -1.0;
    expect_validation_error([&] { problem.validate(); });
    problem.world_scale = 1.0;
    problem.validate();

    problem.pairs.push_back(pair_problem("cam_0__cam_1", -0.2, dummy_problem()));
    expect_validation_error([&] { problem.validate(); });
    problem.pairs[1].pair_id = "cam_2__cam_3";
    problem.validate();

    auto different_roi = dummy_problem();
    different_roi.roi_mask = torch::zeros({8, 8}, torch::kBool);
    problem.pairs[1].reference_stereo = different_roi;
    expect_validation_error([&] { problem.validate(); });
}

void test_pin_multi_solver() {
    test_pin_multi_validate();
    test_pin_multi_solve();
}

void test_pin_multi_solve() {
    neurodic::PINMultiProblem problem;
    problem.world_scale = 1.0;
    problem.require_image_bounds = true;
    problem.reconstruction.max_reprojection_error = 1e-5;
    problem.pairs.push_back(fast_pair_problem("cam_0__cam_1", -0.2));
    problem.pairs.push_back(fast_pair_problem("cam_1__cam_2", -0.25));

    neurodic::PINMultiSolver solver;
    auto result = solver.solve(problem);
    assert(result.pairs.size() == 2);
    for (const auto& pair_result : result.pairs) {
        const auto& stereo = pair_result.result;
        assert(stereo.reference_points.sizes() == torch::IntArrayRef({64, 3}));
        assert(stereo.current_points.sizes() == torch::IntArrayRef({64, 3}));
        assert(stereo.displacement_3d.sizes() == torch::IntArrayRef({64, 3}));
        assert(stereo.valid.sizes() == torch::IntArrayRef({64}));
        assert(stereo.reference_reprojection_error.sizes() == torch::IntArrayRef({64}));
        assert(stereo.current_reprojection_error.sizes() == torch::IntArrayRef({64}));
        assert(stereo.reference_disparity.displacement.coordinates.sizes() == torch::IntArrayRef({64, 2}));
        assert(stereo.left_temporal.displacement.values.sizes() == torch::IntArrayRef({64, 2}));
        assert(stereo.deformed_disparity.displacement.values.sizes() == torch::IntArrayRef({64, 2}));
    }
    assert(result.pairs[0].pair_id == "cam_0__cam_1");
    assert(result.pairs[1].pair_id == "cam_1__cam_2");
}
