#include "neurodic/solver/pin_solver.hpp"

#include <algorithm>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/torch_bspline.hpp"
#include "neurodic/loss/mse.hpp"
#include "neurodic/loss/photometric.hpp"
#include "neurodic/model/model_factory.hpp"
#include "neurodic/optimizer/adam.hpp"
#include "neurodic/representation/pin_displacement_field.hpp"

namespace neurodic {

PINResult PINSolver::solve(const PINProblem& problem) const {
    problem.validate();
    const auto device = problem.device;
    constexpr auto dtype = torch::kFloat32;
    auto model = ModelFactory().create_pin_model("mlp", problem.model_options);
    model->to(device, dtype);
    model->train();
    const auto height = problem.reference_image.size(0);
    const auto width = problem.reference_image.size(1);
    auto normalize_coordinates = [height, width](const torch::Tensor& xy) {
        auto scale = torch::tensor({static_cast<float>(std::max<int64_t>(width - 1, 1)),
                                    static_cast<float>(std::max<int64_t>(height - 1, 1))}, xy.options());
        return xy / scale * 2.0F - 1.0F;
    };
    auto seed_pos = problem.seeds.seed_pos.to(device, dtype);
    auto seed_uv = problem.seeds.seed_uv.to(device, dtype);
    PINDisplacementField field({problem.seeds.scale_uv.slice(0, 0, 2),
                                problem.seeds.scale_uv.slice(0, 2, 4)});
    auto decode = [&](const torch::Tensor& xy) {
        return field.decode(xy, model->forward(normalize_coordinates(xy)));
    };

    double final_loss = 0.0;
    int iterations = 0;
    PINResult result;
    const auto seed_scales = problem.seeds.scale_uv.slice(0, 2, 4);
    const auto active_seed_components = torch::nonzero(
        seed_scales > problem.seed_pretrain_uv_scale_threshold).reshape({-1}).to(device, torch::kLong);
    if (problem.seed_iterations > 0 && active_seed_components.numel() > 0) {
        MSELoss seed_loss;
        AdamOptimizer optimizer(model->parameters(), problem.seed_learning_rate);
        const auto run = optimizer.minimize(problem.seed_iterations, [&] {
            return seed_loss.compute(decode(seed_pos).index_select(1, active_seed_components) -
                                     seed_uv.index_select(1, active_seed_components));
        });
        final_loss = run.final_loss;
        iterations += run.iterations;
    }

    auto roi_indices = torch::nonzero(problem.roi_mask).to(torch::kCPU);
    if (roi_indices.size(0) == 0) throw ValidationError("PINProblem ROI has no valid pixels");
    auto all_xy = torch::stack({roi_indices.select(1, 1), roi_indices.select(1, 0)}, 1).to(device, dtype);
    TorchBSplineInterpolator sampler(problem.precompute.deformed_coefficients.degree);
    auto reference_coefficients = problem.precompute.reference_coefficients.on(device).to(dtype);
    auto deformed_coefficients = problem.precompute.deformed_coefficients.on(device).to(dtype);
    if (problem.photometric_iterations > 0) {
        PhotometricLossOptions loss_options;
        loss_options.type = problem.photometric_loss;
        loss_options.znssd.kernel_size = problem.znssd_kernel_size;
        PhotometricLoss photometric_loss(loss_options);
        AdamOptimizer optimizer(model->parameters(), problem.photometric_learning_rate);
        OptimizationResult run;
        if (problem.photometric_sampling_enabled) {
            const int64_t center_count = std::min<int64_t>(problem.photometric_sample_count, all_xy.size(0));
            auto selected = torch::linspace(0, all_xy.size(0) - 1, center_count,
                torch::TensorOptions().device(device).dtype(torch::kLong));
            auto centers = all_xy.index_select(0, selected);
            const int radius = problem.znssd_kernel_size / 2;
            const int side = 2 * radius + 1;
            auto offset_options = torch::TensorOptions().device(device).dtype(dtype);
            auto axis = torch::arange(-radius, radius + 1, offset_options);
            auto offsets_y = axis.repeat_interleave(side);
            auto offsets_x = axis.repeat({side});
            auto offsets = torch::stack({offsets_x, offsets_y}, 1);
            auto window_xy = centers.unsqueeze(1) + offsets.unsqueeze(0);
            auto x = window_xy.select(2, 0);
            auto y = window_xy.select(2, 1);
            auto inside = (x >= 0) & (x < width) & (y >= 0) & (y < height);
            auto roi_device = problem.roi_mask.to(device);
            auto roi_window = roi_device.index({y.clamp(0, height - 1).to(torch::kLong),
                                                x.clamp(0, width - 1).to(torch::kLong)});
            auto window_mask = inside & roi_window;
            auto flat_xy = window_xy.reshape({-1, 2});
            auto padded_xy = problem.precompute.original_to_padded(flat_xy);
            auto reference_values = sampler.evaluate(reference_coefficients, padded_xy).detach();
            run = optimizer.minimize(problem.photometric_iterations, [&] {
                auto warped = sampler.evaluate(deformed_coefficients, padded_xy + decode(flat_xy));
                return photometric_loss.compute_windows(reference_values.reshape({center_count, -1}),
                                                          warped.reshape({center_count, -1}), window_mask);
            });
            result.diagnostics.metrics["photometric_centers"] = static_cast<double>(center_count);
        } else {
            auto padded_xy = problem.precompute.original_to_padded(all_xy);
            auto reference_values = sampler.evaluate(reference_coefficients, padded_xy).detach();
            auto roi_indices_device = roi_indices.to(device);
            auto roi_device = problem.roi_mask.to(device);
            run = optimizer.minimize(problem.photometric_iterations, [&] {
                auto warped = sampler.evaluate(deformed_coefficients, padded_xy + decode(all_xy));
                auto reference_image = torch::zeros({height, width}, reference_values.options());
                auto warped_image = torch::zeros({height, width}, warped.options());
                reference_image.index_put_({roi_indices_device.select(1, 0), roi_indices_device.select(1, 1)}, reference_values);
                warped_image.index_put_({roi_indices_device.select(1, 0), roi_indices_device.select(1, 1)}, warped);
                return photometric_loss.compute_image(reference_image, warped_image, roi_device);
            });
            result.diagnostics.metrics["photometric_centers"] = static_cast<double>(all_xy.size(0));
        }
        final_loss = run.final_loss;
        iterations += run.iterations;
    }
    model->eval();
    torch::NoGradGuard no_grad;
    auto values = decode(all_xy).detach().to(torch::kCPU);
    result.displacement = {all_xy.detach().to(torch::kCPU), values};
    result.diagnostics.status = SolverStatus::CONVERGED;
    result.diagnostics.iterations = iterations;
    result.diagnostics.final_loss = final_loss;
    result.diagnostics.metrics["seed_count"] = static_cast<double>(seed_pos.size(0));
    result.diagnostics.metrics["seed_pretraining_enabled"] = active_seed_components.numel() > 0 ? 1.0 : 0.0;
    result.diagnostics.metrics["seed_pretraining_components"] = static_cast<double>(active_seed_components.numel());
    result.diagnostics.metrics["seed_uv_scale_u"] = seed_scales[0].item<double>();
    result.diagnostics.metrics["seed_uv_scale_v"] = seed_scales[1].item<double>();
    result.diagnostics.metrics["photometric_sampling_enabled"] = problem.photometric_sampling_enabled ? 1.0 : 0.0;
    return result;
}

}  // namespace neurodic
