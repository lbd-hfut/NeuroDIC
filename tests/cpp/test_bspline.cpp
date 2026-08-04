#include <cassert>
#include <cmath>

#include "neurodic/data/image_precompute_context.hpp"
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

    neurodic::ImagePrecomputeOptions options;
    options.integer_search_radius = 7;
    options.coarse_subset_radius = 9;
    options.fine_subset_radius = 5;
    options.subset_radius = 6;
    options.bspline_border = 3;
    options.bspline_degree = 3;
    assert(neurodic::calculate_image_padding(options) == 19);

    auto image = torch::tensor({{1., 2., 3.}, {4., 5., 6.}}, torch::kFloat64);
    auto padded = neurodic::mirror_pad_image(image, 2);
    assert(padded.sizes() == torch::IntArrayRef({6, 7}));
    assert(padded.index({0, 0}).item<double>() == 5.0);
    assert(padded.index({2, 2}).item<double>() == 1.0);
    assert(padded.index({5, 6}).item<double>() == 2.0);

    auto ref = torch::arange(36, torch::kFloat64).reshape({6, 6});
    auto mask = torch::ones({6, 6}, torch::kBool);
    options.integer_search_radius = 1;
    options.coarse_subset_radius = 1;
    options.fine_subset_radius = 1;
    options.subset_radius = 1;
    options.bspline_border = 1;
    auto context = neurodic::ImagePrecomputeContext::create(ref, ref + 2, mask, options);
    assert(context.pad_offset == 3);
    auto original = torch::tensor({{0., 0.}, {5., 4.}}, torch::kFloat64);
    assert(torch::equal(context.padded_to_original(context.original_to_padded(original)), original));
    assert(context.roi_mask_padded.sum().item<int64_t>() == 36);
    const auto& blocks = context.reference_coefficients.cpu();
    assert(blocks.sizes() == torch::IntArrayRef({12, 12, 4, 4}));
    assert(blocks.device().is_cpu());

    auto xy = torch::tensor({{3.25, 3.5}, {6.75, 5.125}}, torch::kFloat64);
    auto sampled = neurodic::TorchBSplineInterpolator(3).evaluate(blocks, xy);
    for (int64_t i = 0; i < xy.size(0); ++i) {
        const int64_t x = static_cast<int64_t>(std::floor(xy[i][0].item<double>()));
        const int64_t y = static_cast<int64_t>(std::floor(xy[i][1].item<double>()));
        const double dx = xy[i][0].item<double>() - x;
        const double dy = xy[i][1].item<double>() - y;
        double expected = 0.0;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c)
            expected += blocks[y][x][r][c].item<double>() * std::pow(dy, r) * std::pow(dx, c);
        assert(std::abs(sampled[i].item<double>() - expected) < 1e-10);
    }
}
