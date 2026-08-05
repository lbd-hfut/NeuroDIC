#include "neurodic/loss/znssd.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

ZNSSDLoss::ZNSSDLoss(ZNSSDLossOptions options) : options_(options) {
    if (options_.epsilon <= 0.0 || options_.kernel_size < 1 || options_.kernel_size % 2 == 0)
        throw ValidationError("ZNSSD epsilon must be positive and kernel size must be positive odd");
}

torch::Tensor ZNSSDLoss::compute_windows(const torch::Tensor& reference, const torch::Tensor& deformed,
                                         const torch::Tensor& mask) const {
    if (reference.dim() != 2 || deformed.sizes() != reference.sizes() || mask.sizes() != reference.sizes() ||
        reference.device() != deformed.device() || reference.device() != mask.device())
        throw ValidationError("ZNSSD windows must be matching [B,K*K] tensors on one device");
    auto weights = mask.to(reference.scalar_type());
    auto count = weights.sum(1, true).clamp_min(1.0);
    auto reference_mean = (weights * reference).sum(1, true) / count;
    auto deformed_mean = (weights * deformed).sum(1, true) / count;
    auto epsilon = torch::tensor(options_.epsilon, reference.options());
    auto reference_std = torch::sqrt((weights * torch::square(reference - reference_mean)).sum(1, true) / count + epsilon);
    auto deformed_std = torch::sqrt((weights * torch::square(deformed - deformed_mean)).sum(1, true) / count + epsilon);
    // MSPINN-DIC's local ZNSSD convention preserves the deformed local intensity scale.
    auto residual = (reference - reference_mean) / reference_std * deformed_std - (deformed - deformed_mean);
    return torch::mean((weights * torch::square(residual)).sum(1) / count.squeeze(1));
}

torch::Tensor ZNSSDLoss::compute_image(const torch::Tensor& reference, const torch::Tensor& deformed,
                                       const torch::Tensor& roi_mask) const {
    if (reference.dim() != 2 || deformed.sizes() != reference.sizes() || roi_mask.sizes() != reference.sizes() ||
        reference.device() != deformed.device() || reference.device() != roi_mask.device())
        throw ValidationError("ZNSSD images and ROI must be matching [H,W] tensors on one device");
    const int pad = options_.kernel_size / 2;
    auto weights = roi_mask.to(reference.scalar_type());
    auto kernel = torch::ones({1, 1, options_.kernel_size, options_.kernel_size}, reference.options());
    auto convolution = [&](const torch::Tensor& image) {
        return torch::conv2d(torch::constant_pad_nd(image.unsqueeze(0).unsqueeze(0), {pad, pad, pad, pad}, 0.0), kernel)
            .squeeze(0).squeeze(0);
    };
    auto reference_roi = reference * weights;
    auto deformed_roi = deformed * weights;
    auto count = convolution(weights).clamp_min(1.0);
    auto reference_mean = convolution(reference_roi) / count;
    auto deformed_mean = convolution(deformed_roi) / count;
    auto epsilon = torch::tensor(options_.epsilon, reference.options());
    auto reference_std = torch::sqrt(torch::clamp_min(convolution(reference_roi * reference_roi) / count -
        torch::square(reference_mean), epsilon));
    auto deformed_std = torch::sqrt(torch::clamp_min(convolution(deformed_roi * deformed_roi) / count -
        torch::square(deformed_mean), epsilon));
    auto residual = (reference - reference_mean) / reference_std * deformed_std - (deformed - deformed_mean);
    return torch::sum(weights * torch::square(residual)) / weights.sum().clamp_min(1.0);
}

torch::Tensor ZNSSDLoss::compute(const torch::Tensor& residual) {
    if (!residual.defined() || !residual.is_floating_point())
        throw ValidationError("ZNSSD residual must be a defined floating tensor");
    return torch::mean(torch::square(residual));
}

torch::Tensor ZNSSDLoss::compute(const torch::Tensor& reference, const torch::Tensor& deformed) const {
    if (!reference.defined() || !deformed.defined() || reference.sizes() != deformed.sizes() ||
        reference.device() != deformed.device() || reference.scalar_type() != deformed.scalar_type())
        throw ValidationError("ZNSSD observations must have matching shape, dtype, and device");
    const auto epsilon = torch::tensor(options_.epsilon, reference.options());
    auto normalized_reference = (reference - reference.mean()) /
        torch::sqrt(reference.var(false, false) + epsilon);
    auto normalized_deformed = (deformed - deformed.mean()) /
        torch::sqrt(deformed.var(false, false) + epsilon);
    return torch::mean(torch::square(normalized_reference - normalized_deformed));
}

}  // namespace neurodic
