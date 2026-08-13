#include "neurodic/optimizer/adam.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

AdamOptimizer::AdamOptimizer(std::vector<torch::Tensor> parameters, double learning_rate)
    : optimizer_(std::move(parameters), torch::optim::AdamOptions(learning_rate)) {
    if (learning_rate <= 0.0) throw ValidationError("Adam learning rate must be positive");
}

OptimizationResult AdamOptimizer::minimize(int iterations, const LossClosure& closure) {
    if (iterations < 0) throw ValidationError("Adam iteration count must be non-negative");
    OptimizationResult result;
    for (int step = 0; step < iterations; ++step) {
        optimizer_.zero_grad();
        auto loss = closure();
        if (!loss.defined() || loss.numel() != 1) throw ValidationError("Optimization closure must return scalar loss");
        loss.backward();
        optimizer_.step();
        result.final_loss = loss.detach().item<double>();
        result.losses.push_back(result.final_loss);
        ++result.iterations;
    }
    return result;
}

}  // namespace neurodic
