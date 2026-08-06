#include "neurodic/problem/ndef_problem.hpp"

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"

namespace neurodic {

NDeFProblem::NDeFProblem(torch::Tensor surface, torch::Tensor reference, torch::Tensor deformed,
                         torch::Tensor reference_mask, torch::Tensor deformed_mask,
                         std::vector<CameraModel> input_cameras)
    : reference_surface(surface.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()),
      reference_images(reference.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()),
      deformed_images(deformed.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()),
      reference_masks(reference_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous()),
      deformed_masks(deformed_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous()),
      cameras(std::move(input_cameras)) {}

void NDeFProblem::validate() const {
    if (!reference_surface.defined() || reference_surface.dim() != 2 || reference_surface.size(0) < 1 ||
        reference_surface.size(1) != 3 || !reference_surface.is_floating_point())
        throw ValidationError("NDeF requires a non-empty floating reference_surface [N,3]");
    if (!reference_images.defined() || reference_images.dim() != 3 || !deformed_images.defined() ||
        deformed_images.sizes() != reference_images.sizes() || !reference_masks.defined() ||
        reference_masks.sizes() != reference_images.sizes() || !deformed_masks.defined() ||
        deformed_masks.sizes() != reference_images.sizes() || !reference_images.is_floating_point())
        throw ValidationError("NDeF observations require matching reference/deformed images and masks [V,H,W]");
    if (reference_images.size(0) < 2 || static_cast<int64_t>(cameras.size()) != reference_images.size(0))
        throw ValidationError("NDeF requires at least two cameras and one observation pair per camera");
    for (const auto& camera : cameras) {
        camera.validate();
        if (camera.image_width != reference_images.size(2) || camera.image_height != reference_images.size(1))
            throw ValidationError("NDeF camera image size must match its observation tensor");
    }
    const auto points = reference_surface.size(0), views = reference_images.size(0);
    if (reference_visibility.defined() && (reference_visibility.dim() != 2 ||
        reference_visibility.size(0) != points || reference_visibility.size(1) != views))
        throw ValidationError("NDeF reference_visibility must be [N,V]");
    if (reference_projected_uv.defined() && (reference_projected_uv.dim() != 3 ||
        reference_projected_uv.size(0) != points || reference_projected_uv.size(1) != views ||
        reference_projected_uv.size(2) != 2 || !reference_projected_uv.is_floating_point()))
        throw ValidationError("NDeF reference_projected_uv must be floating [N,V,2]");
    if (reference_visibility.defined() != reference_projected_uv.defined())
        throw ValidationError("NDeF surface visibility and projected UV must be supplied together");
    if (visible_counts.defined() && (visible_counts.dim() != 1 || visible_counts.size(0) != points ||
        !visible_counts.is_floating_point())) throw ValidationError("NDeF visible_counts must be floating [N]");
    validate_bspline_degree(bspline_degree);
    if (model_options.hidden_dim < 1 || model_options.hidden_layers < 1 || model_options.output_scale <= 0.0 ||
        training_epochs < 0 || batch_size < 0 || auto_batch_start < 1 || auto_batch_max < 0 ||
        memory_fraction <= 0.0 || memory_fraction > 1.0 || max_steps_per_epoch < 0 || prediction_batch_size < 1 ||
        random_seed < 0 || photometric_iterations < 0 || photometric_sample_count < 0 ||
        photometric_learning_rate <= 0.0 || weight_decay < 0.0 || smoothness_weight < 0.0 || patch_radius < 0 ||
        min_valid_patch_ratio <= 0.0 || min_valid_patch_ratio > 1.0 || invalid_patch_penalty < 0.0 ||
        sfm_to_world_scale <= 0.0)
        throw ValidationError("NDeF training options are invalid");
}

}  // namespace neurodic
