#include "neurodic/initialization/initialization_result.hpp"
#include "neurodic/initialization/seed_set.hpp"
#include "neurodic/initialization/sift_grid_seed_initializer.hpp"

#include <cassert>

void test_initialization() {
    neurodic::InitializationResult result;
    (void)result;
    auto empty = neurodic::SeedSet::empty();
    empty.validate();
    assert(empty.seed_pos.sizes() == torch::IntArrayRef({0, 2}));

    auto blank = torch::zeros({64, 64}, torch::kFloat32);
    auto mask = torch::ones({64, 64}, torch::kBool);
    auto sift_empty = neurodic::SiftGridSeedInitializer().initialize(blank, blank, mask);
    sift_empty.validate();
    assert(sift_empty.seed_pos.size(0) == 0);

    auto sparse = blank.clone();
    sparse.index_put_({32, 32}, 255.0F);
    auto sift_sparse = neurodic::SiftGridSeedInitializer().initialize(sparse, sparse, mask);
    sift_sparse.validate();
    assert(sift_sparse.seed_pos.size(0) == 0);

    auto positions = torch::tensor({{1., 2.}, {3., 4.}});
    auto constant = neurodic::SeedSet::constant(positions, 2.5, -1.0);
    assert(constant.seed_pos.sizes() == torch::IntArrayRef({2, 2}));
    assert(torch::allclose(constant.scale_uv, torch::tensor({2.5F, -1.F, 1e-6F, 1e-6F})));
}
