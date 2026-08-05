#include "neurodic/loss/mse.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

torch::Tensor MSELoss::compute(const torch::Tensor& residual) {
    if (!residual.defined() || !residual.is_floating_point())
        throw ValidationError("MSE residual must be a defined floating tensor");
    return torch::mean(torch::square(residual));
}

}  // namespace neurodic
