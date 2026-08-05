#include <cassert>

#include "neurodic/initialization/seed_set.hpp"
#include "neurodic/loss/photometric.hpp"
#include "neurodic/problem/pin_problem.hpp"
#include "neurodic/solver/pin_solver.hpp"

void test_pin_solver() {
    auto photometric_reference = torch::tensor({1.0, 2.0, 4.0}, torch::kFloat64);
    auto photometric_deformed = photometric_reference.clone();
    photometric_deformed.requires_grad_(true);
    neurodic::PhotometricLoss ssd({neurodic::PhotometricLossType::SSD});
    assert(ssd.compute(photometric_reference, photometric_deformed).item<double>() == 0.0);
    neurodic::PhotometricLoss znssd({neurodic::PhotometricLossType::ZNSSD});
    auto invariant_loss = znssd.compute(photometric_reference, photometric_deformed * 2.0 + 3.0);
    assert(invariant_loss.item<double>() < 1e-12);
    auto local_reference = torch::arange(9, torch::kFloat64).reshape({3, 3});
    auto local_deformed = local_reference * 2.0 + 3.0;
    auto local_mask = torch::ones({3, 3}, torch::kBool);
    assert(znssd.compute_windows(local_reference.reshape({1, 9}), local_deformed.reshape({1, 9}),
                                 local_mask.reshape({1, 9})).item<double>() < 1e-12);
    assert(znssd.compute_image(local_reference, local_deformed, local_mask).item<double>() < 1e-12);
    znssd.compute(photometric_reference, photometric_deformed).backward();
    assert(photometric_deformed.grad().defined());

    auto reference = torch::rand({16, 16}, torch::kFloat32);
    auto mask = torch::ones({16, 16}, torch::kBool);
    auto seed_positions = torch::tensor({{2.F, 2.F}, {12.F, 3.F}, {4.F, 12.F}, {13.F, 13.F}});
    auto seeds = neurodic::SeedSet::constant(seed_positions, 0.0, 0.0);
    neurodic::PINProblem problem(reference, reference.clone(), mask, seeds);
    problem.model_options.hidden_dim = 8;
    problem.model_options.hidden_layers = 1;
    problem.seed_iterations = 2;
    problem.photometric_iterations = 2;
    problem.photometric_sample_count = 32;
    auto result = neurodic::PINSolver().solve(problem);
    assert(result.diagnostics.status == neurodic::SolverStatus::CONVERGED);
    assert(result.diagnostics.iterations == 4);
    assert(result.displacement.coordinates.sizes() == torch::IntArrayRef({256, 2}));
    assert(result.displacement.values.sizes() == torch::IntArrayRef({256, 2}));
}
