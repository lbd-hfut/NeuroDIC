/**
 * Selectable SSD or zero-normalized SSD photometric loss.
 *
 * Responsibilities: connect warped sampling residuals to loss values.
 * Inputs: sampled reference/current tensors.
 * Outputs: scalar tensor.
 * Ownership: stateless shell.
 * Differentiable: YES.
 */
#pragma once

#include "neurodic/loss/loss.hpp"

#include "neurodic/loss/ssd.hpp"
#include "neurodic/loss/znssd.hpp"

namespace neurodic {

enum class PhotometricLossType { SSD, ZNSSD };

struct PhotometricLossOptions {
    PhotometricLossType type{PhotometricLossType::ZNSSD};
    ZNSSDLossOptions znssd{};
};

class PhotometricLoss : public Loss {
public:
    explicit PhotometricLoss(PhotometricLossOptions options = {});
    torch::Tensor compute(const torch::Tensor& residual) override;
    torch::Tensor compute(const torch::Tensor& reference, const torch::Tensor& deformed) const;
    torch::Tensor compute_windows(const torch::Tensor& reference, const torch::Tensor& deformed,
                                  const torch::Tensor& mask) const;
    torch::Tensor compute_image(const torch::Tensor& reference, const torch::Tensor& deformed,
                                const torch::Tensor& roi_mask) const;

private:
    PhotometricLossOptions options_;
    SSDLoss ssd_;
    ZNSSDLoss znssd_;
};

}  // namespace neurodic
