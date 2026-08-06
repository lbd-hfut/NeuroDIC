#include <cassert>

#include "neurodic/problem/ndef_problem.hpp"
#include "neurodic/solver/ndef_solver.hpp"

void test_ndef_solver() {
    using namespace neurodic;
    CameraModel left, right;
    for (auto* camera : {&left, &right}) {
        camera->intrinsics = torch::tensor({{20., 0., 16.}, {0., 20., 16.}, {0., 0., 1.}}, torch::kFloat64);
        camera->rotation = torch::eye(3, torch::kFloat64);
        camera->distortion = torch::zeros({5}, torch::kFloat64);
        camera->image_width = 32;
        camera->image_height = 32;
    }
    left.translation = torch::zeros({3}, torch::kFloat64);
    right.translation = torch::tensor({0.2, 0., 0.}, torch::kFloat64);
    auto axis = torch::arange(32, torch::kFloat32);
    auto image = axis.unsqueeze(0).repeat({32, 1});
    auto images = torch::stack({image, image}, 0);
    auto mask = torch::ones({2, 32, 32}, torch::kBool);
    auto surface = torch::tensor({{-0.4F, -0.4F, 4.F}, {0.4F, -0.4F, 4.F},
                                  {-0.4F, 0.4F, 4.F}, {0.4F, 0.4F, 4.F}});
    NDeFProblem problem(surface, images, images, mask, mask, {left, right});
    problem.photometric_iterations = 2;
    problem.photometric_sample_count = 4;
    problem.bspline_degree = 3;
    problem.photometric_loss = PhotometricLossType::SSD;
    // Mirror the NDeF-DIC surface-dataset contract rather than relying on
    // solver-derived reference visibility.
    problem.reference_visibility = torch::ones({4, 2}, torch::kBool);
    problem.reference_projected_uv = torch::tensor({{{14.F, 14.F}, {15.F, 14.F}},
                                                     {{18.F, 14.F}, {19.F, 14.F}},
                                                     {{14.F, 18.F}, {15.F, 18.F}},
                                                     {{18.F, 18.F}, {19.F, 18.F}}});
    problem.visible_counts = torch::full({4}, 2.F);
    problem.patch_radius = 1;
    problem.smoothness_weight = 1e-5;
    problem.sfm_to_world_scale = 2.0;
    auto result = NDeFSolver().solve(problem);
    assert(result.surface.coordinates.sizes() == torch::IntArrayRef({4, 3}));
    assert(result.surface.values.sizes() == torch::IntArrayRef({4, 3}));
    assert(result.deformation.values.sizes() == torch::IntArrayRef({4, 3}));
    assert(result.valid.sizes() == torch::IntArrayRef({4, 2}));
    assert(result.valid.all().item<bool>());
    assert(torch::allclose(result.reference_surface_sfm, surface));
    assert(torch::allclose(result.surface.coordinates, surface * 2.0F));
    assert(result.sfm_to_world_scale == 2.0);
    assert(result.diagnostics.iterations == 2);
    assert(result.training_history.sizes() == torch::IntArrayRef({2, 8}));
    assert(result.training_batch_size == 4);
    assert(result.steps_per_epoch == 1);
    assert(result.training_sample_counts.sum().item<int64_t>() == 8);
    assert(result.training_sample_counts.size(0) == surface.size(0));
    assert(result.coordinate_center.sizes() == torch::IntArrayRef({3}));
    assert(result.coordinate_scale.min().item<float>() > 0.0F);
    assert(!result.model_parameter_names.empty());
    assert(result.model_state.size() == result.model_parameter_names.size());
    assert(result.last_model_state.size() == result.model_parameter_names.size());
}
