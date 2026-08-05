#include "neurodic/loss/photometric.hpp"

namespace neurodic {

PhotometricLoss::PhotometricLoss(PhotometricLossOptions options)
    : options_(options), znssd_(options.znssd) {}

torch::Tensor PhotometricLoss::compute(const torch::Tensor& residual) {
    return options_.type == PhotometricLossType::SSD ? ssd_.compute(residual) : znssd_.compute(residual);
}

torch::Tensor PhotometricLoss::compute(const torch::Tensor& reference, const torch::Tensor& deformed) const {
    return options_.type == PhotometricLossType::SSD ? ssd_.compute(reference, deformed) :
        znssd_.compute(reference, deformed);
}

torch::Tensor PhotometricLoss::compute_windows(const torch::Tensor& reference, const torch::Tensor& deformed,
                                                const torch::Tensor& mask) const {
    return options_.type == PhotometricLossType::SSD ? ssd_.compute_masked(reference, deformed, mask) :
        znssd_.compute_windows(reference, deformed, mask);
}

torch::Tensor PhotometricLoss::compute_image(const torch::Tensor& reference, const torch::Tensor& deformed,
                                              const torch::Tensor& roi_mask) const {
    return options_.type == PhotometricLossType::SSD ? ssd_.compute_masked(reference, deformed, roi_mask) :
        znssd_.compute_image(reference, deformed, roi_mask);
}

}  // namespace neurodic
