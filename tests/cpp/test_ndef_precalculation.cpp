#include <cassert>

#include "neurodic/calibration/camera_model.hpp"

#include "neurodic/initialization/ndef_precalculation.hpp"

void test_ndef_precalculation() {
    auto displacement = torch::tensor({{1.F, 0.F, 0.F}, {1.1F, 0.F, 0.F}, {0.9F, 0.F, 0.F}, {100.F, 0.F, 0.F}});
    auto scale = neurodic::estimate_ndef_displacement_scale(displacement, 3.0);
    assert(scale.inlier_mask.sum().item<int64_t>() == 3);
    assert(scale.mean > 0.9 && scale.mean < 1.1);
    assert(scale.maximum < 1.2);

    auto make_camera = [](double tx) {
        neurodic::CameraModel camera;
        camera.intrinsics = torch::tensor({{80.0, 0.0, 32.0}, {0.0, 80.0, 32.0}, {0.0, 0.0, 1.0}}, torch::kFloat64);
        camera.rotation = torch::eye(3, torch::kFloat64); camera.translation = torch::tensor({tx, 0.0, 0.0}, torch::kFloat64);
        camera.distortion = torch::zeros({5}, torch::kFloat64); camera.image_width = 64; camera.image_height = 64;
        return camera;
    };
    auto base = torch::zeros({64, 64}, torch::kFloat);
    for (int y = 0; y < 64; ++y) for (int x = 0; x < 64; ++x)
        base.index_put_({y, x}, static_cast<float>((17 * x + 31 * y + (x * y) % 23) % 255) / 255.0F);
    auto shifted = torch::zeros_like(base);
    shifted.index({torch::indexing::Slice(), torch::indexing::Slice(1, 64)}).copy_(base.index({torch::indexing::Slice(), torch::indexing::Slice(0, 63)}));
    // Camera 1 is translated by -0.1 at Z=1 (8 px disparity); make its
    // texture obey that epipolar shift before applying the common +1 px time
    // translation.
    auto right_reference = torch::zeros_like(base);
    right_reference.index({torch::indexing::Slice(), torch::indexing::Slice(0, 56)}).copy_(base.index({torch::indexing::Slice(), torch::indexing::Slice(8, 64)}));
    auto right_current = torch::zeros_like(base);
    right_current.index({torch::indexing::Slice(), torch::indexing::Slice(1, 64)}).copy_(right_reference.index({torch::indexing::Slice(), torch::indexing::Slice(0, 63)}));
    auto visible = torch::ones({4, 2}, torch::kBool);
    auto uv = torch::tensor({{{30.0, 30.0}, {22.0, 30.0}}, {{34.0, 30.0}, {26.0, 30.0}},
                             {{30.0, 34.0}, {22.0, 34.0}}, {{34.0, 34.0}, {26.0, 34.0}}}, torch::kFloat64);
    neurodic::NDeFSparsePrecalculationOptions options;
    options.points_per_camera = 4; options.patch_radius = 2; options.cross_search_radius = 3;
    options.temporal_search_radius = 2; options.min_texture_std = 0.0;
    options.cross_ncc_threshold = 0.8; options.temporal_ncc_threshold = 0.8; options.max_reprojection_error = 0.2;
    auto sparse = neurodic::NDeFSparsePrecalculator(options).solve(torch::stack({base, right_reference}), torch::stack({shifted, right_current}),
        torch::ones({2, 64, 64}, torch::kBool), visible, uv, {make_camera(0.0), make_camera(-0.1)});
    assert(sparse.displacement.size(0) > 0 && torch::isfinite(sparse.displacement).all().item<bool>());

    neurodic::NDeFDenseSurfaceSampleOptions dense_options;
    dense_options.voxel_size = 0.01; dense_options.min_visible_views = 2;
    auto sampled = neurodic::NDeFDenseSurfaceSampler(dense_options).sample(
        torch::tensor({{0.0, 0.0, 1.0}, {0.001, 0.001, 1.0}, {0.05, 0.0, 1.0}}, torch::kFloat64),
        torch::ones({2, 64, 64}, torch::kBool), {make_camera(0.0), make_camera(-0.1)});
    assert(sampled.points.size(0) == 2 && sampled.visible_counts.min().item<float>() >= 2.0F);
}
