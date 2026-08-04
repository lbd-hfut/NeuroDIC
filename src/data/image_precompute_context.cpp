#include "neurodic/data/image_precompute_context.hpp"

#include <algorithm>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace {
void validate_2d(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined() || tensor.dim() != 2) throw ValidationError(std::string(name) + " must be [H,W]");
}

torch::Tensor reflected_indices(int64_t size, int pad) {
    if (size <= 0) throw ValidationError("Cannot pad an empty image");
    auto idx = torch::arange(-pad, size + pad, torch::TensorOptions().dtype(torch::kLong));
    if (size == 1) return torch::zeros_like(idx);
    const int64_t period = 2 * size;
    idx = torch::remainder(idx, period);
    return torch::where(idx < size, idx, period - 1 - idx);
}
}  // namespace

int calculate_image_padding(const ImagePrecomputeOptions& o) {
    if (o.integer_search_radius < 0 || o.coarse_subset_radius < 0 ||
        o.fine_subset_radius < 0 || o.subset_radius < 0 || o.bspline_border < 0) {
        throw ValidationError("Image precompute radii must be non-negative");
    }
    return o.integer_search_radius +
        std::max({o.coarse_subset_radius, o.fine_subset_radius, o.subset_radius}) +
        o.bspline_border;
}

torch::Tensor mirror_pad_image(const torch::Tensor& image, int pad) {
    validate_2d(image, "image");
    if (pad < 0) throw ValidationError("pad must be non-negative");
    if (pad == 0) return image.clone();
    auto yi = reflected_indices(image.size(0), pad).to(image.device());
    auto xi = reflected_indices(image.size(1), pad).to(image.device());
    return image.index_select(0, yi).index_select(1, xi);
}

torch::Tensor zero_pad_roi_mask(const torch::Tensor& mask, int pad) {
    validate_2d(mask, "ROI mask");
    if (pad < 0) throw ValidationError("pad must be non-negative");
    auto output = torch::zeros({mask.size(0) + 2 * pad, mask.size(1) + 2 * pad}, mask.options());
    output.index_put_({torch::indexing::Slice(pad, pad + mask.size(0)),
                       torch::indexing::Slice(pad, pad + mask.size(1))}, mask);
    return output;
}

ImagePrecomputeContext ImagePrecomputeContext::create(
    const torch::Tensor& reference, const torch::Tensor& deformed,
    const torch::Tensor& mask, const ImagePrecomputeOptions& options) {
    validate_2d(reference, "reference image");
    validate_2d(deformed, "deformed image");
    validate_2d(mask, "ROI mask");
    if (reference.sizes() != deformed.sizes() || reference.sizes() != mask.sizes())
        throw ValidationError("Reference, deformed, and ROI tensors must have equal shapes");
    if (!reference.device().is_cpu() || !deformed.device().is_cpu() || !mask.device().is_cpu())
        throw ValidationError("ImagePrecomputeContext preprocessing inputs must be on CPU");
    ImagePrecomputeContext result;
    result.pad_offset = calculate_image_padding(options);
    result.reference_padded = mirror_pad_image(reference, result.pad_offset).contiguous();
    result.deformed_padded = mirror_pad_image(deformed, result.pad_offset).contiguous();
    result.roi_mask_padded = zero_pad_roi_mask(mask, result.pad_offset).contiguous();
    result.reference_coefficients = make_bspline_coefficient_block(
        result.reference_padded, options.bspline_degree, result.pad_offset);
    result.deformed_coefficients = make_bspline_coefficient_block(
        result.deformed_padded, options.bspline_degree, result.pad_offset);
    return result;
}

torch::Tensor ImagePrecomputeContext::original_to_padded(const torch::Tensor& xy) const {
    if (!xy.defined() || xy.dim() != 2 || xy.size(1) != 2) throw ValidationError("Coordinates must be [N,2]");
    return xy + pad_offset;
}

torch::Tensor ImagePrecomputeContext::padded_to_original(const torch::Tensor& xy) const {
    if (!xy.defined() || xy.dim() != 2 || xy.size(1) != 2) throw ValidationError("Coordinates must be [N,2]");
    return xy - pad_offset;
}

}  // namespace neurodic
