#include "neurodic/solver/ndef_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/data/image_precompute_context.hpp"
#include "neurodic/geometry/ndef_geometry.hpp"
#include "neurodic/interpolation/torch_bspline.hpp"
#include "neurodic/loss/photometric.hpp"
#include "neurodic/model/ndef_internal_model.hpp"
#include "neurodic/representation/ndef_deformation_field.hpp"
#include "neurodic/representation/ndef_surface_field.hpp"

namespace neurodic {
namespace {
torch::Tensor patch_offsets(int radius, const torch::TensorOptions& options) {
    auto axis = torch::arange(-radius, radius + 1, options);
    const auto side = 2 * radius + 1;
    return torch::stack({axis.repeat({side}), axis.repeat_interleave(side)}, 1);
}

torch::Tensor patch_valid(const torch::Tensor& patch_uv, const torch::Tensor& mask,
                          int64_t width, int64_t height, const torch::Device& device) {
    auto x = patch_uv.select(2, 0), y = patch_uv.select(2, 1);
    auto inside = (x >= 0.0F) & (x < width) & (y >= 0.0F) & (y < height);
    auto xi = x.clamp(0, width - 1).to(torch::kLong);
    auto yi = y.clamp(0, height - 1).to(torch::kLong);
    return inside & mask.to(device).index({yi, xi});
}

torch::Tensor patch_loss(const torch::Tensor& reference, const torch::Tensor& current,
                         const torch::Tensor& valid, PhotometricLossType type) {
    auto weights = valid.to(reference.scalar_type());
    auto count = weights.sum(1, true).clamp_min(1.0F);
    if (type == PhotometricLossType::SSD)
        return (weights * torch::square(reference - current)).sum(1) / count.squeeze(1);
    auto ref_mean = (weights * reference).sum(1, true) / count;
    auto cur_mean = (weights * current).sum(1, true) / count;
    auto ref_std = torch::sqrt((weights * torch::square(reference - ref_mean)).sum(1, true) / count + 1e-6F);
    auto cur_std = torch::sqrt((weights * torch::square(current - cur_mean)).sum(1, true) / count + 1e-6F);
    return (weights * torch::square((reference - ref_mean) / ref_std - (current - cur_mean) / cur_std)).sum(1) /
           count.squeeze(1);
}

torch::Tensor smoothness_loss(NDeFInternalModel& model, const torch::Tensor& points) {
    auto normalized = model.normalize(points.detach()).set_requires_grad(true);
    auto displacement = model.forward_normalized(normalized);
    std::vector<torch::Tensor> gradients;
    for (int component = 0; component < 3; ++component) {
        auto derivative = torch::autograd::grad({displacement.select(1, component).sum()}, {normalized}, {}, true, true)[0];
        gradients.push_back(derivative);
    }
    return torch::stack(gradients, 1).square().sum({1, 2}).mean();
}
}  // namespace

NDeFResult NDeFSolver::solve(const NDeFProblem& problem) const {
    problem.validate();
    constexpr auto dtype = torch::kFloat32;
    const auto device = problem.device;
    const auto views = problem.reference_images.size(0);
    const auto points_count = problem.reference_surface.size(0);
    auto reference_surface = problem.reference_surface.to(device, dtype);
    auto bounds_min = reference_surface.amin(0);
    auto bounds_max = reference_surface.amax(0);
    auto model = NDeFInternalModel(problem.model_options, (bounds_min + bounds_max) * 0.5F,
                                   ((bounds_max - bounds_min) * 0.5F).clamp_min(1e-6F));
    model.to(device, dtype);
    model.train();
    NDeFGeometry geometry(problem.cameras);
    NDeFDeformationField deformation_field;
    NDeFSurfaceField surface_field;
    TorchBSplineInterpolator sampler(problem.bspline_degree);

    std::vector<ImagePrecomputeContext> precompute;
    precompute.reserve(static_cast<std::size_t>(views));
    ImagePrecomputeOptions options;
    options.bspline_degree = problem.bspline_degree;
    for (int64_t view = 0; view < views; ++view) {
        // The per-view contexts deliberately stay outside the optimizer: images are fixed observations.
        precompute.emplace_back(ImagePrecomputeContext::create(problem.reference_images[view],
            problem.deformed_images[view], problem.reference_masks[view], options));
    }
    auto reference_projection = geometry.project_reference_surface(reference_surface);
    auto projection_valid = [&](const MultiViewProjectionResult& projection, const torch::Tensor& masks) {
        auto valid = projection.depth > 0.0F;
        for (int64_t view = 0; view < views; ++view) {
            auto uv = projection.uv.select(1, view);
            auto x = uv.select(1, 0), y = uv.select(1, 1);
            auto in_bounds = (x >= 0.0F) & (x < problem.cameras[static_cast<std::size_t>(view)].image_width) &
                             (y >= 0.0F) & (y < problem.cameras[static_cast<std::size_t>(view)].image_height);
            auto xi = x.clamp(0, problem.cameras[static_cast<std::size_t>(view)].image_width - 1).to(torch::kLong);
            auto yi = y.clamp(0, problem.cameras[static_cast<std::size_t>(view)].image_height - 1).to(torch::kLong);
            auto roi = masks[view].to(device).index({yi, xi});
            valid.select(1, view).copy_(valid.select(1, view) & in_bounds & roi);
        }
        return valid;
    };
    auto reference_uv = problem.reference_projected_uv.defined()
        ? problem.reference_projected_uv.to(device, dtype) : reference_projection.uv;
    auto reference_for_validity = MultiViewProjectionResult{reference_uv, reference_projection.depth};
    auto reference_valid = projection_valid(reference_for_validity, problem.reference_masks);
    if (problem.reference_visibility.defined()) reference_valid = reference_valid & problem.reference_visibility.to(device);
    auto candidate_any = reference_valid.any(1);
    auto candidates = torch::nonzero(candidate_any).reshape({-1});
    if (candidates.numel() == 0) throw ValidationError("NDeF reference surface has no visible ROI observations");
    const auto selected_count = std::min<int64_t>(problem.photometric_sample_count, candidates.numel());
    auto select_position = torch::linspace(0, candidates.numel() - 1, selected_count,
                                            torch::TensorOptions().device(device).dtype(torch::kLong));
    auto selected = candidates.index_select(0, select_position);
    auto selected_reference_uv = reference_uv.index_select(0, selected);
    auto selected_reference_valid = reference_valid.index_select(0, selected);
    auto selected_visible_counts = problem.visible_counts.defined()
        ? problem.visible_counts.to(device, dtype).index_select(0, selected).clamp_min(1.0F)
        : selected_reference_valid.sum(1).to(dtype).clamp_min(1.0F);
    const auto offsets = patch_offsets(problem.patch_radius, torch::TensorOptions().device(device).dtype(dtype));
    const auto patch_size = offsets.size(0);
    const auto required_patch_pixels = static_cast<double>(patch_size) * problem.min_valid_patch_ratio;
    double final_loss = 0.0;
    int iterations = 0;
    double last_valid_observations = 0.0, last_supervised_observations = 0.0, last_smoothness = 0.0;
    if (problem.photometric_iterations > 0) {
        torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(problem.photometric_learning_rate)
            .weight_decay(problem.weight_decay));
        double best_loss = std::numeric_limits<double>::infinity();
        std::vector<torch::Tensor> best_parameters;
        for (int iteration = 0; iteration < problem.photometric_iterations; ++iteration) {
            optimizer.zero_grad();
            auto selected_surface = reference_surface.index_select(0, selected);
            auto deformation = deformation_field.decode(selected_surface, model.forward(selected_surface));
            auto current_projection = geometry.project_deformed_surface(selected_surface, deformation);
            torch::Tensor weighted_loss = torch::zeros({}, reference_surface.options());
            auto total_weight = torch::zeros({}, reference_surface.options());
            last_valid_observations = 0.0;
            last_supervised_observations = 0.0;
            for (int64_t view = 0; view < views; ++view) {
                auto ref_patch_uv = selected_reference_uv.select(1, view).unsqueeze(1) + offsets.unsqueeze(0);
                auto ref_patch_valid = patch_valid(ref_patch_uv, problem.reference_masks[view],
                    problem.cameras[static_cast<std::size_t>(view)].image_width,
                    problem.cameras[static_cast<std::size_t>(view)].image_height, device);
                auto supervised = selected_reference_valid.select(1, view) &
                    (ref_patch_valid.sum(1).to(dtype) >= required_patch_pixels);
                auto indices = torch::nonzero(supervised).reshape({-1});
                if (indices.numel() == 0) continue;
                last_supervised_observations += static_cast<double>(indices.numel());
                auto cur_patch_uv = current_projection.uv.select(1, view).index_select(0, indices).unsqueeze(1) + offsets.unsqueeze(0);
                auto cur_patch_valid = patch_valid(cur_patch_uv, problem.deformed_masks[view],
                    problem.cameras[static_cast<std::size_t>(view)].image_width,
                    problem.cameras[static_cast<std::size_t>(view)].image_height, device);
                auto current_good = (current_projection.depth.select(1, view).index_select(0, indices) > 1e-8F) &
                    (cur_patch_valid.sum(1).to(dtype) >= required_patch_pixels);
                auto flat_ref = ref_patch_uv.index_select(0, indices).reshape({-1, 2});
                auto flat_cur = cur_patch_uv.reshape({-1, 2});
                auto ref_xy = precompute[static_cast<std::size_t>(view)].original_to_padded(flat_ref);
                auto cur_xy = precompute[static_cast<std::size_t>(view)].original_to_padded(flat_cur);
                auto coefficients = precompute[static_cast<std::size_t>(view)].deformed_coefficients.on(device).to(dtype);
                auto ref_coefficients = precompute[static_cast<std::size_t>(view)].reference_coefficients.on(device).to(dtype);
                auto reference = sampler.evaluate(ref_coefficients, ref_xy).reshape({indices.numel(), patch_size}).detach();
                auto observed = sampler.evaluate(coefficients, cur_xy).reshape({indices.numel(), patch_size});
                auto pair = torch::where(current_good, patch_loss(reference, observed,
                    ref_patch_valid.index_select(0, indices) & cur_patch_valid, problem.photometric_loss),
                    torch::full({indices.numel()}, problem.invalid_patch_penalty, reference.options()));
                auto weights = 1.0F / selected_visible_counts.index_select(0, indices);
                weighted_loss = weighted_loss + (pair * weights).sum();
                total_weight = total_weight + weights.sum();
                last_valid_observations += current_good.sum().item<double>();
            }
            if (last_supervised_observations == 0.0) throw ValidationError("NDeF selected surface has no valid reference patches");
            auto photo = weighted_loss / total_weight.clamp_min(1e-8F);
            auto smooth = problem.smoothness_weight > 0.0 ? smoothness_loss(model, selected_surface) : torch::zeros({}, photo.options());
            auto loss = photo + problem.smoothness_weight * smooth;
            loss.backward(); optimizer.step();
            final_loss = loss.detach().item<double>(); last_smoothness = smooth.detach().item<double>(); ++iterations;
            if (final_loss < best_loss) {
                best_loss = final_loss; best_parameters.clear();
                for (const auto& parameter : model.parameters()) best_parameters.push_back(parameter.detach().clone());
            }
        }
        if (!best_parameters.empty()) {
            torch::NoGradGuard guard;
            auto parameters = model.parameters();
            for (size_t index = 0; index < parameters.size(); ++index) parameters[index].copy_(best_parameters[index]);
            final_loss = best_loss;
        }
    }
    model.eval();
    torch::NoGradGuard no_grad;
    auto deformation = deformation_field.decode(reference_surface, model.forward(reference_surface));
    auto current_surface = surface_field.decode(reference_surface, deformation);
    auto current_projection = geometry.project_reference_surface(current_surface);
    auto valid = reference_valid & projection_valid(current_projection, problem.deformed_masks);
    NDeFResult result;
    const auto scale = static_cast<float>(problem.sfm_to_world_scale);
    result.reference_surface_sfm = reference_surface.detach().to(torch::kCPU);
    result.current_surface_sfm = current_surface.detach().to(torch::kCPU);
    result.deformation_sfm = deformation.detach().to(torch::kCPU);
    result.sfm_to_world_scale = problem.sfm_to_world_scale;
    result.surface.coordinates = result.reference_surface_sfm * scale;
    result.surface.values = result.current_surface_sfm * scale;
    result.deformation.coordinates = result.reference_surface_sfm * scale;
    result.deformation.values = result.deformation_sfm * scale;
    result.reference_uv = reference_uv.detach().to(torch::kCPU);
    result.current_uv = current_projection.uv.detach().to(torch::kCPU);
    result.reference_depth = reference_projection.depth.detach().to(torch::kCPU);
    result.current_depth = current_projection.depth.detach().to(torch::kCPU);
    result.valid = valid.detach().to(torch::kCPU);
    result.diagnostics.status = SolverStatus::CONVERGED;
    result.diagnostics.iterations = iterations;
    result.diagnostics.final_loss = final_loss;
    result.diagnostics.metrics["surface_samples"] = static_cast<double>(points_count);
    result.diagnostics.metrics["photometric_samples"] = static_cast<double>(selected_count);
    result.diagnostics.metrics["reference_valid_observations"] = reference_valid.sum().item<double>();
    result.diagnostics.metrics["current_valid_observations"] = result.valid.sum().item<double>();
    result.diagnostics.metrics["last_training_valid_observations"] = last_valid_observations;
    result.diagnostics.metrics["last_training_supervised_observations"] = last_supervised_observations;
    result.diagnostics.metrics["last_smoothness"] = last_smoothness;
    return result;
}

}  // namespace neurodic
