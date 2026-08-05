#include "neurodic/problem/pin_problem.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

PINProblem::PINProblem(GeometryType geometry_type) : geometry_type_(geometry_type) {}

PINProblem::PINProblem(torch::Tensor reference, torch::Tensor deformed, torch::Tensor mask,
                       SeedSet seed_set, PINModelOptions options, ImagePrecomputeOptions precompute_options)
    : geometry_type_(GeometryType::PLANAR_2D),
      reference_image(reference.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()),
      deformed_image(deformed.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()),
      roi_mask(mask.detach().to(torch::kCPU).to(torch::kBool).contiguous()),
      precompute(ImagePrecomputeContext::create(reference_image, deformed_image, roi_mask, precompute_options)),
      seeds(std::move(seed_set)), model_options(options) {}

void PINProblem::validate() const {
    if (geometry_type_ == GeometryType::NDEF_MULTIVIEW) {
        throw ValidationError("PINProblem cannot use NDEF_MULTIVIEW geometry");
    }
    if (geometry_type_ != GeometryType::PLANAR_2D)
        throw ValidationError("The validated PIN solver currently supports planar 2D only");
    if (!reference_image.defined() || reference_image.dim() != 2 || !deformed_image.defined() ||
        deformed_image.sizes() != reference_image.sizes() || !roi_mask.defined() ||
        roi_mask.sizes() != reference_image.sizes())
        throw ValidationError("PINProblem requires matching [H,W] reference, deformed, and ROI tensors");
    seeds.validate();
    if (seeds.seed_pos.size(0) == 0) throw ValidationError("PINProblem requires at least one cleaned seed");
    if (seed_iterations < 0 || seed_pretrain_uv_scale_threshold < 0.0 || photometric_iterations < 0 || photometric_sample_count < 1 ||
        znssd_kernel_size < 1 || znssd_kernel_size % 2 == 0 ||
        seed_learning_rate <= 0.0 || photometric_learning_rate <= 0.0)
        throw ValidationError("PIN training options must be nonnegative (iterations may be zero)");
    precompute.reference_coefficients.validate();
    precompute.deformed_coefficients.validate();
}

}  // namespace neurodic
