#include "neurodic/initialization/initialization_result.hpp"
#include "neurodic/initialization/seed_set.hpp"
#include "neurodic/initialization/sift_grid_seed_initializer.hpp"
#include "neurodic/initialization/traditional_seed_initializer.hpp"
#include "neurodic/model/fourier.hpp"
#include "neurodic/model/mlp.hpp"
#include "neurodic/model/ndef_internal_model.hpp"
#include "neurodic/representation/pin_displacement_field.hpp"

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

    auto textured = torch::rand({128, 128}, torch::kFloat32) * 255.0F;
    auto shifted = torch::zeros_like(textured);
    shifted.index_put_({torch::indexing::Slice(5, torch::indexing::None),
                        torch::indexing::Slice(7, torch::indexing::None)},
                       textured.index({torch::indexing::Slice(torch::indexing::None, -5),
                                       torch::indexing::Slice(torch::indexing::None, -7)}));
    neurodic::TraditionalSeedOptions traditional_options;
    traditional_options.target_seed_count = 8;
    traditional_options.kmeans_sample_limit = 1000;
    traditional_options.search_radius = 12;
    traditional_options.sift_prior_enabled = false;
    traditional_options.subpixel_enabled = false;
    auto traditional = neurodic::TraditionalSeedInitializer(traditional_options).initialize(
        textured, shifted, torch::ones({128, 128}, torch::kBool));
    traditional.validate();
    assert(traditional.seed_pos.size(0) >= 3);
    assert(torch::allclose(traditional.seed_uv.mean(0), torch::tensor({7.0F, 5.0F}), 0.25, 0.25));

    neurodic::FourierEncodingOptions encoding_options;
    encoding_options.enabled = true;
    encoding_options.num_frequencies = 6;
    encoding_options.include_input = true;
    encoding_options.angular_scale = 3.14159265358979323846;
    neurodic::FourierEncoding encoder(2, encoding_options);
    auto encoded = encoder->forward(torch::zeros({4, 2}, torch::kFloat32));
    assert(encoded.sizes() == torch::IntArrayRef({4, 26}));

    neurodic::PINModelOptions pin_options;
    pin_options.hidden_dim = 16;
    pin_options.hidden_layers = 2;
    neurodic::MLPModel pin_model(pin_options);
    auto pin_coordinates = torch::randn({5, 2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto pin_raw = pin_model.forward(pin_coordinates);
    assert(pin_raw.sizes() == torch::IntArrayRef({5, 2}));
    pin_raw.square().mean().backward();
    assert(pin_coordinates.grad().defined());
    neurodic::PINDisplacementField pin_field({torch::tensor({2.0F, -1.0F}), torch::tensor({4.0F, 5.0F})});
    assert(torch::allclose(pin_field.decode(pin_coordinates, torch::zeros({5, 2})),
                           torch::tensor({2.0F, -1.0F}).repeat({5, 1})));

    neurodic::NDeFInternalModel ndef_model({}, torch::zeros({3}), torch::ones({3}));
    auto surface = torch::randn({7, 3}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto deformation = ndef_model.forward(surface);
    assert(deformation.sizes() == torch::IntArrayRef({7, 3}));
    assert(deformation.abs().max().item<float>() < 1e-2F);
    deformation.square().mean().backward();
    assert(surface.grad().defined());
}
