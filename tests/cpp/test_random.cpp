#include <cassert>

#include <torch/torch.h>

#include "neurodic/core/random.hpp"

void test_random() {
    neurodic::set_random_seed(314159U);
    const auto first = torch::rand({16}, torch::kFloat32);
    neurodic::set_random_seed(314159U);
    const auto second = torch::rand({16}, torch::kFloat32);
    assert(torch::equal(first, second));
}
