#include "neurodic/loss/ssd.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

torch::Tensor SSDLoss::compute(const torch::Tensor& residual) {
    if (!residual.defined() || !residual.is_floating_point())
        throw ValidationError("SSD residual must be a defined floating tensor");
    return torch::mean(torch::square(residual));
}

torch::Tensor SSDLoss::compute(const torch::Tensor& reference, const torch::Tensor& deformed) const {
    if (!reference.defined() || !deformed.defined() || reference.sizes() != deformed.sizes() ||
        reference.device() != deformed.device() || reference.scalar_type() != deformed.scalar_type())
        throw ValidationError("SSD observations must have matching shape, dtype, and device");
    return torch::mean(torch::square(reference - deformed));
}

torch::Tensor SSDLoss::compute_masked(const torch::Tensor& reference, const torch::Tensor& deformed,
                                      const torch::Tensor& mask) const {
    if (!mask.defined() || mask.sizes() != reference.sizes() || mask.device() != reference.device())
        throw ValidationError("SSD mask must match observation shape and device");
    auto weights = mask.to(reference.scalar_type());
    return torch::sum(weights * torch::square(reference - deformed)) / weights.sum().clamp_min(1.0);
}

}  // namespace neurodic
