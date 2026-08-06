#include "neurodic/solver/ndef_solver.hpp"

#include <ATen/Context.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/geometry/ndef_geometry.hpp"
#include "neurodic/model/ndef_internal_model.hpp"
#include "neurodic/representation/ndef_deformation_field.hpp"
#include "neurodic/representation/ndef_surface_field.hpp"

namespace neurodic {
namespace {

using torch::indexing::Slice;

torch::Tensor patch_offsets(int radius, const torch::TensorOptions& options) {
    auto axis = torch::arange(-radius, radius + 1, options);
    const auto side = 2 * radius + 1;
    // Python make_patch_offsets flattens meshgrid(dv,du) and stacks [du,dv].
    return torch::stack({axis.repeat({side}), axis.repeat_interleave(side)}, 1);
}

torch::Tensor in_image_patch_bounds(const torch::Tensor& uv, const torch::Tensor& cameras,
                                    const torch::Tensor& image_sizes) {
    auto sizes = image_sizes.index_select(0, cameras);
    auto width = sizes.select(1, 0).unsqueeze(1), height = sizes.select(1, 1).unsqueeze(1);
    return (uv.select(2, 0) >= 0.0F) & (uv.select(2, 0) <= width - 1.0F) &
           (uv.select(2, 1) >= 0.0F) & (uv.select(2, 1) <= height - 1.0F);
}

torch::Tensor in_image_bounds(const torch::Tensor& uv, const torch::Tensor& cameras,
                              const torch::Tensor& image_sizes) {
    auto sizes = image_sizes.index_select(0, cameras);
    return (uv.select(1, 0) >= 0.0F) & (uv.select(1, 0) <= sizes.select(1, 0) - 1.0F) &
           (uv.select(1, 1) >= 0.0F) & (uv.select(1, 1) <= sizes.select(1, 1) - 1.0F);
}

// Exact deformation_dataset.py bilinear sampler: out-of-bounds neighbours are
// clamped, while patch validity is decided separately from the sampled values.
torch::Tensor sample_per_camera(const torch::Tensor& images, const torch::Tensor& uv,
                                const torch::Tensor& cameras) {
    const auto height = images.size(1), width = images.size(2);
    auto x = uv.select(2, 0), y = uv.select(2, 1);
    auto x0 = torch::floor(x).to(torch::kLong), y0 = torch::floor(y).to(torch::kLong);
    auto x1 = x0 + 1, y1 = y0 + 1;
    auto camera_grid = cameras.unsqueeze(1).expand_as(x0);
    auto q00 = images.index({camera_grid, y0.clamp(0, height - 1), x0.clamp(0, width - 1)});
    auto q10 = images.index({camera_grid, y0.clamp(0, height - 1), x1.clamp(0, width - 1)});
    auto q01 = images.index({camera_grid, y1.clamp(0, height - 1), x0.clamp(0, width - 1)});
    auto q11 = images.index({camera_grid, y1.clamp(0, height - 1), x1.clamp(0, width - 1)});
    auto dx = x - x0.to(x.scalar_type()), dy = y - y0.to(y.scalar_type());
    return q00 * (1.0F - dx) * (1.0F - dy) + q10 * dx * (1.0F - dy) +
           q01 * (1.0F - dx) * dy + q11 * dx * dy;
}

torch::Tensor patch_loss(const torch::Tensor& reference, const torch::Tensor& current,
                         PhotometricLossType type) {
    if (type == PhotometricLossType::SSD) return torch::square(reference - current).mean(1);
    auto ref_centered = reference - reference.mean(1, true);
    auto cur_centered = current - current.mean(1, true);
    auto ref_std = torch::sqrt(ref_centered.square().mean(1, true) + 1e-6F);
    auto cur_std = torch::sqrt(cur_centered.square().mean(1, true) + 1e-6F);
    return torch::square(ref_centered / ref_std - cur_centered / cur_std).mean(1);
}

torch::Tensor smoothness_loss(NDeFInternalModel& model, const torch::Tensor& points) {
    auto normalized = model.normalize(points.detach()).set_requires_grad(true);
    auto displacement = model.forward_normalized(normalized);
    std::vector<torch::Tensor> gradients;
    for (int component = 0; component < 3; ++component) {
        gradients.push_back(torch::autograd::grad(
            {displacement.select(1, component).sum()}, {normalized}, {}, true, true)[0]);
    }
    return torch::stack(gradients, 1).square().sum({1, 2}).mean();
}

struct TrainingObjective {
    torch::Tensor total;
    torch::Tensor photo;
    torch::Tensor smooth;
    torch::Tensor displacement_rms;
    double valid_pairs{0.0};
    double supervised_pairs{0.0};
};

TrainingObjective objective(NDeFInternalModel& model, const NDeFProblem& problem,
                            NDeFGeometry& geometry, const torch::Tensor& reference_surface,
                            const torch::Tensor& reference_uv, const torch::Tensor& visibility,
                            const torch::Tensor& visible_counts, const torch::Tensor& indices,
                            const torch::Tensor& reference_images, const torch::Tensor& current_images,
                            const torch::Tensor& image_sizes) {
    auto points = reference_surface.index_select(0, indices);
    auto batch_visibility = visibility.index_select(0, indices);
    auto batch_reference_uv = reference_uv.index_select(0, indices);
    auto batch_visible_counts = visible_counts.index_select(0, indices).clamp_min(1.0F);
    auto displacement = model.forward(points);
    auto current = geometry.project_reference_surface(points + displacement);
    auto pair_ids = torch::nonzero(batch_visibility);
    auto zero = displacement.sum() * 0.0F;
    if (pair_ids.numel() == 0) return {zero, zero.detach(), zero.detach(),
        displacement.square().sum(1).mean().sqrt().detach(), 0.0, 0.0};
    auto point_ids = pair_ids.select(1, 0), camera_ids = pair_ids.select(1, 1);
    auto ref_uv = batch_reference_uv.index({point_ids, camera_ids});
    auto cur_uv = current.uv.index({point_ids, camera_ids});
    auto depth = current.depth.index({point_ids, camera_ids});
    auto offsets = patch_offsets(problem.patch_radius, ref_uv.options());
    auto ref_patch_uv = ref_uv.unsqueeze(1) + offsets.unsqueeze(0);
    auto cur_patch_uv = cur_uv.unsqueeze(1) + offsets.unsqueeze(0);
    auto ref_bounds = in_image_patch_bounds(ref_patch_uv, camera_ids, image_sizes);
    auto cur_bounds = in_image_patch_bounds(cur_patch_uv, camera_ids, image_sizes);
    const auto minimum = std::max(1.0, static_cast<double>(offsets.size(0)) * problem.min_valid_patch_ratio);
    auto supervised = ref_bounds.to(torch::kFloat).sum(1) >= minimum;
    if (!supervised.any().item<bool>()) return {zero, zero.detach(), zero.detach(),
        displacement.square().sum(1).mean().sqrt().detach(), 0.0, 0.0};
    point_ids = point_ids.index({supervised}); camera_ids = camera_ids.index({supervised});
    ref_patch_uv = ref_patch_uv.index({supervised}); cur_patch_uv = cur_patch_uv.index({supervised});
    cur_bounds = cur_bounds.index({supervised}); depth = depth.index({supervised});
    auto current_valid = in_image_bounds(cur_uv.index({supervised}), camera_ids, image_sizes) &
                         (depth > 1e-8F) & (cur_bounds.to(torch::kFloat).sum(1) >= minimum);
    auto pair = torch::full({point_ids.size(0)}, problem.invalid_patch_penalty, points.options());
    auto valid_ids = torch::nonzero(current_valid).reshape({-1});
    if (valid_ids.numel() > 0) {
        auto valid_cameras = camera_ids.index_select(0, valid_ids);
        auto reference_patch = sample_per_camera(reference_images, ref_patch_uv.index_select(0, valid_ids), valid_cameras);
        auto current_patch = sample_per_camera(current_images, cur_patch_uv.index_select(0, valid_ids), valid_cameras);
        pair.index_put_({valid_ids}, patch_loss(reference_patch, current_patch, problem.photometric_loss));
    }
    auto weights = 1.0F / batch_visible_counts.index_select(0, point_ids);
    auto photo = (pair * weights).sum() / weights.sum().clamp_min(1e-8F);
    auto smooth = problem.smoothness_weight > 0.0 ? smoothness_loss(model, points) : torch::zeros({}, photo.options());
    return {photo + problem.smoothness_weight * smooth, photo.detach(), smooth.detach(),
            displacement.square().sum(1).mean().sqrt().detach(),
            current_valid.sum().item<double>(), static_cast<double>(point_ids.numel())};
}

torch::Tensor predict(NDeFInternalModel& model, const torch::Tensor& points, int64_t batch_size) {
    std::vector<torch::Tensor> chunks;
    torch::NoGradGuard guard;
    for (int64_t start = 0; start < points.size(0); start += batch_size) {
        chunks.push_back(model.forward(points.slice(0, start, std::min(start + batch_size, points.size(0)))).cpu());
    }
    return torch::cat(chunks, 0);
}

}  // namespace

NDeFResult NDeFSolver::solve(const NDeFProblem& problem) const {
    problem.validate();
    constexpr auto dtype = torch::kFloat32;
    const auto device = problem.device;
    const auto views = problem.reference_images.size(0);
    const auto points_count = problem.reference_surface.size(0);
    torch::manual_seed(problem.random_seed);
    if (device.is_cuda()) torch::cuda::manual_seed_all(problem.random_seed);
    at::globalContext().setDeterministicAlgorithms(true, true);
    auto reference_surface = problem.reference_surface.to(device, dtype);
    auto bounds_min = reference_surface.amin(0), bounds_max = reference_surface.amax(0);
    auto center = (bounds_min + bounds_max) * 0.5F;
    auto scale = ((bounds_max - bounds_min) * 0.5F).clamp_min(1e-8F);
    auto model = NDeFInternalModel(problem.model_options, center, scale);
    model.to(device, dtype); model.train();
    NDeFGeometry geometry(problem.cameras);

    auto reference_projection = geometry.project_reference_surface(reference_surface);
    auto image_sizes = torch::empty({views, 2}, reference_surface.options());
    for (int64_t view = 0; view < views; ++view) {
        image_sizes.index_put_({view, 0}, problem.cameras[static_cast<size_t>(view)].image_width);
        image_sizes.index_put_({view, 1}, problem.cameras[static_cast<size_t>(view)].image_height);
    }
    auto reference_uv = problem.reference_projected_uv.defined()
        ? problem.reference_projected_uv.to(device, dtype) : reference_projection.uv;
    torch::Tensor reference_valid;
    if (problem.reference_visibility.defined()) reference_valid = problem.reference_visibility.to(device);
    else {
        reference_valid = reference_projection.depth > 0.0F;
        for (int64_t view = 0; view < views; ++view) {
            auto cams = torch::full({points_count}, view, torch::TensorOptions().device(device).dtype(torch::kLong));
            reference_valid.select(1, view).logical_and_(in_image_bounds(reference_uv.select(1, view), cams, image_sizes));
        }
    }
    auto visible_counts = problem.visible_counts.defined()
        ? problem.visible_counts.to(device, dtype).clamp_min(1.0F)
        : reference_valid.sum(1).to(dtype).clamp_min(1.0F);
    if (!reference_valid.any().item<bool>()) throw ValidationError("NDeF reference surface has no visible observations");
    auto reference_images = problem.reference_images.to(device, dtype);
    auto current_images = problem.deformed_images.to(device, dtype);

    int64_t batch = problem.batch_size > 0 ? problem.batch_size : std::min<int64_t>(problem.auto_batch_start, points_count);
    if (problem.photometric_sample_count > 0) batch = problem.photometric_sample_count;
    const int64_t upper = problem.auto_batch_max > 0 ? std::min<int64_t>(problem.auto_batch_max, points_count) : points_count;
    // Python uses auto_batch_start on CPU. CUDA probing is intentionally kept in
    // Python-equivalent doubling/bisection shape; OOM recovery is delegated to
    // LibTorch's allocator by retrying smaller batches.
    if (problem.batch_size == 0 && device.is_cuda()) {
        int64_t last_good = 0, probe = std::min<int64_t>(batch, upper), high = probe;
        const auto device_index = device.has_index() ? device.index() : c10::cuda::current_device();
        const auto memory = c10::cuda::CUDACachingAllocator::get()->getMemoryInfo(device_index);
        const auto target_bytes = static_cast<size_t>(static_cast<double>(memory.first) * problem.memory_fraction);
        auto fits = [&](int64_t count) {
            try {
                c10::cuda::CUDACachingAllocator::emptyCache();
                c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
                model.zero_grad();
                auto ids = torch::randint(points_count, {count}, torch::TensorOptions().device(device).dtype(torch::kLong));
                auto trial = objective(model, problem, geometry, reference_surface, reference_uv, reference_valid,
                    visible_counts, ids, reference_images, current_images, image_sizes);
                trial.total.backward(); model.zero_grad();
                const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
                return static_cast<size_t>(stats.allocated_bytes[0].peak) <= target_bytes;
            } catch (const c10::Error& error) {
                model.zero_grad();
                if (std::string(error.what()).find("out of memory") != std::string::npos) {
                    c10::cuda::CUDACachingAllocator::emptyCache(); return false;
                }
                throw;
            }
        };
        while (probe <= upper && fits(probe)) {
            last_good = probe; if (probe == upper) break;
            probe = std::min<int64_t>(probe * 2, upper); high = probe;
        }
        if (last_good == 0) throw ValidationError("NDeF automatic batch sizing could not fit one batch");
        if (last_good < upper) {
            int64_t low = last_good;
            while (high - low > 1) { auto mid = (low + high) / 2; if (fits(mid)) low = last_good = mid; else high = mid; }
        }
        batch = last_good;
    }
    batch = std::max<int64_t>(1, batch);
    int steps_per_epoch = static_cast<int>((points_count + batch - 1) / batch);
    if (problem.max_steps_per_epoch > 0) steps_per_epoch = std::min(steps_per_epoch, problem.max_steps_per_epoch);
    int total_steps = problem.photometric_iterations > 0
        ? problem.photometric_iterations : problem.training_epochs * steps_per_epoch;

    torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(problem.photometric_learning_rate)
        .weight_decay(problem.weight_decay));
    std::vector<double> history; history.reserve(static_cast<size_t>(total_steps) * 8);
    auto sample_counts = torch::zeros({points_count}, torch::TensorOptions().device(device).dtype(torch::kLong));
    double best_loss = std::numeric_limits<double>::infinity();
    std::vector<torch::Tensor> best_parameters;
    for (int iteration = 0; iteration < total_steps; ++iteration) {
        auto indices = torch::randint(points_count, {batch}, torch::TensorOptions().device(device).dtype(torch::kLong));
        sample_counts.add_(torch::bincount(indices, {}, points_count));
        optimizer.zero_grad();
        auto metrics = objective(model, problem, geometry, reference_surface, reference_uv, reference_valid,
            visible_counts, indices, reference_images, current_images, image_sizes);
        metrics.total.backward(); optimizer.step();
        const double loss = metrics.total.detach().item<double>();
        const int epoch = iteration / steps_per_epoch + 1, step = iteration % steps_per_epoch + 1;
        history.insert(history.end(), {static_cast<double>(epoch), static_cast<double>(step), loss,
            metrics.photo.item<double>(), metrics.smooth.item<double>(), metrics.valid_pairs,
            metrics.supervised_pairs, metrics.displacement_rms.item<double>()});
        if (loss < best_loss) {
            best_loss = loss; best_parameters.clear();
            for (const auto& parameter : model.parameters()) best_parameters.push_back(parameter.detach().clone());
        }
    }
    std::vector<torch::Tensor> last_parameters;
    for (const auto& parameter : model.parameters()) last_parameters.push_back(parameter.detach().cpu().clone());
    if (!best_parameters.empty()) {
        torch::NoGradGuard guard; auto parameters = model.parameters();
        for (size_t index = 0; index < parameters.size(); ++index) parameters[index].copy_(best_parameters[index]);
    }
    model.eval();
    auto deformation_cpu = predict(model, reference_surface, problem.prediction_batch_size);
    auto reference_cpu = reference_surface.cpu();
    auto current_cpu = reference_cpu + deformation_cpu;
    auto current_projection = geometry.project_reference_surface(current_cpu.to(device));
    auto valid = reference_valid.clone();
    for (int64_t view = 0; view < views; ++view) {
        auto cams = torch::full({points_count}, view, torch::TensorOptions().device(device).dtype(torch::kLong));
        valid.select(1, view).logical_and_(current_projection.depth.select(1, view) > 1e-8F);
        valid.select(1, view).logical_and_(in_image_bounds(current_projection.uv.select(1, view), cams, image_sizes));
    }

    NDeFResult result;
    const auto world_scale = static_cast<float>(problem.sfm_to_world_scale);
    result.reference_surface_sfm = reference_cpu;
    result.current_surface_sfm = current_cpu;
    result.deformation_sfm = deformation_cpu;
    result.sfm_to_world_scale = problem.sfm_to_world_scale;
    result.surface.coordinates = reference_cpu * world_scale;
    result.surface.values = current_cpu * world_scale;
    result.deformation.coordinates = reference_cpu * world_scale;
    result.deformation.values = deformation_cpu * world_scale;
    result.reference_uv = reference_uv.cpu(); result.current_uv = current_projection.uv.cpu();
    result.reference_depth = reference_projection.depth.cpu(); result.current_depth = current_projection.depth.cpu();
    result.valid = valid.cpu();
    result.training_history = history.empty() ? torch::empty({0, 8}, torch::kFloat64)
        : torch::from_blob(history.data(), {total_steps, 8}, torch::kFloat64).clone();
    result.training_sample_counts = sample_counts.cpu();
    result.coordinate_center = center.cpu(); result.coordinate_scale = scale.cpu();
    result.training_batch_size = static_cast<int>(batch); result.steps_per_epoch = steps_per_epoch;
    result.completed_epochs = total_steps == 0 ? 0 : (total_steps + steps_per_epoch - 1) / steps_per_epoch;
    result.random_seed = problem.random_seed; result.output_scale = problem.model_options.output_scale;
    for (const auto& item : model.named_parameters()) result.model_parameter_names.push_back(item.key());
    for (const auto& parameter : model.parameters()) result.model_state.push_back(parameter.detach().cpu().clone());
    result.last_model_state = std::move(last_parameters);
    result.diagnostics.status = SolverStatus::CONVERGED; result.diagnostics.iterations = total_steps;
    result.diagnostics.final_loss = std::isfinite(best_loss) ? best_loss : 0.0;
    result.diagnostics.metrics["surface_samples"] = static_cast<double>(points_count);
    result.diagnostics.metrics["batch_size"] = static_cast<double>(batch);
    result.diagnostics.metrics["steps_per_epoch"] = static_cast<double>(steps_per_epoch);
    result.diagnostics.metrics["epochs"] = static_cast<double>(result.completed_epochs);
    result.diagnostics.metrics["sample_draws"] = sample_counts.sum().item<double>();
    result.diagnostics.metrics["sampled_unique_points"] = (sample_counts > 0).sum().item<double>();
    result.diagnostics.metrics["reference_valid_observations"] = reference_valid.sum().item<double>();
    result.diagnostics.metrics["current_valid_observations"] = valid.sum().item<double>();
    if (!history.empty()) {
        result.diagnostics.metrics["initial_loss"] = history[2];
        result.diagnostics.metrics["last_loss"] = history[history.size() - 6];
    }
    return result;
}

}  // namespace neurodic
