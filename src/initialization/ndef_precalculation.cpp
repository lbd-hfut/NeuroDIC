#include "neurodic/initialization/ndef_precalculation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <torch/nn/functional/fold.h>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/geometry/triangulation.hpp"
#include "neurodic/geometry/projection.hpp"

namespace neurodic {

NDeFDisplacementScale estimate_ndef_displacement_scale(const torch::Tensor& displacement, double mad_threshold) {
    if (!displacement.defined() || displacement.dim() != 2 || displacement.size(1) != 3 ||
        !displacement.is_floating_point() || mad_threshold <= 0.0)
        throw ValidationError("NDeF displacement scale expects floating displacement[N,3] and positive MAD threshold");
    auto values = displacement.detach().to(torch::kCPU).to(torch::kFloat64);
    auto magnitude = torch::linalg_vector_norm(values, 2, 1);
    auto finite = torch::isfinite(magnitude);
    auto finite_values = magnitude.index({finite});
    if (finite_values.numel() == 0) throw ValidationError("NDeF displacement scale requires one finite displacement");
    auto median = torch::median(finite_values);
    auto mad = torch::median(torch::abs(finite_values - median));
    auto inlier = finite.clone();
    if (mad.item<double>() >= 1e-12)
        inlier = finite & (torch::abs(magnitude - median) <= mad_threshold * 1.4826 * mad);
    auto filtered = magnitude.index({inlier});
    if (filtered.numel() == 0) throw ValidationError("NDeF MAD filtering removed every sparse displacement");
    auto quantile = [&](double q) { return torch::quantile(filtered, q).item<double>(); };
    return {inlier, quantile(0.5), filtered.mean().item<double>(), quantile(0.75), quantile(0.90), filtered.max().item<double>()};
}

namespace {
using torch::indexing::Slice;

torch::Tensor normalized_image(const torch::Tensor& image) {
    auto value = image.to(torch::kFloat);
    return value.detach().amax().item<double>() > 1.5 ? value / 255.0F : value;
}

std::pair<torch::Tensor, torch::Tensor> centered_windows(const torch::Tensor& image,
                                                         const torch::Tensor& centers, int radius) {
    const auto height = image.size(0), width = image.size(1);
    auto rounded = torch::round(centers).to(torch::kLong);
    auto x = rounded.select(1, 0), y = rounded.select(1, 1);
    auto valid = (x >= radius) & (x < width - radius) & (y >= radius) & (y < height - radius);
    auto safe_x = x.clamp(radius, width - radius - 1), safe_y = y.clamp(radius, height - radius - 1);
    auto offsets = torch::arange(-radius, radius + 1, torch::TensorOptions().device(image.device()).dtype(torch::kLong));
    auto yy = safe_y.reshape({-1, 1, 1}) + offsets.reshape({1, -1, 1});
    auto xx = safe_x.reshape({-1, 1, 1}) + offsets.reshape({1, 1, -1});
    return {image.index({yy, xx}).unsqueeze(1), valid};
}

struct TensorMatch { torch::Tensor uv, score, valid; };

TensorMatch match_ncc_batch(const torch::Tensor& source_image, const torch::Tensor& target_image,
                            const torch::Tensor& source_centers, const torch::Tensor& target_centers,
                            int patch_radius, int search_radius, int batch_size) {
    std::vector<torch::Tensor> output_uv, output_score, output_valid;
    auto values = torch::arange(-search_radius, search_radius + 1, source_centers.options());
    auto grids = torch::meshgrid({values, values}, "ij");
    auto search_offsets = torch::stack({grids[1].reshape({-1}), grids[0].reshape({-1})}, 1);
    for (int64_t start = 0; start < source_centers.size(0); start += batch_size) {
        const auto stop = std::min<int64_t>(start + batch_size, source_centers.size(0));
        auto ref_centers = source_centers.slice(0, start, stop);
        auto tgt_centers = target_centers.slice(0, start, stop);
        auto [reference_patch, reference_valid] = centered_windows(source_image, ref_centers, patch_radius);
        auto [target_window, target_valid] = centered_windows(target_image, tgt_centers, patch_radius + search_radius);
        auto candidates = torch::nn::functional::unfold(target_window,
            torch::nn::functional::UnfoldFuncOptions(2 * patch_radius + 1));
        auto reference_flat = reference_patch.reshape({reference_patch.size(0), -1});
        auto reference_zero = reference_flat - reference_flat.mean(1, true);
        auto candidate_zero = candidates - candidates.mean(1, true);
        auto numerator = (candidate_zero * reference_zero.unsqueeze(2)).sum(1);
        auto denominator = torch::sqrt(reference_zero.square().sum(1, true) * candidate_zero.square().sum(1) + 1e-6F);
        auto scores = numerator / denominator.clamp_min(1e-6F);
        auto maximum = scores.max(1);
        auto best_score = std::get<0>(maximum), best_index = std::get<1>(maximum);
        output_uv.push_back(tgt_centers + search_offsets.index_select(0, best_index));
        output_score.push_back(best_score);
        output_valid.push_back(reference_valid & target_valid & torch::isfinite(best_score));
    }
    return {torch::cat(output_uv), torch::cat(output_score), torch::cat(output_valid)};
}

std::vector<std::pair<int, int>> sample_roi(const torch::Tensor& image, const torch::Tensor& mask,
                                             int count, int radius, double min_texture_std,
                                             std::mt19937_64& generator) {
    std::vector<std::pair<int, int>> points, selected;
    auto image_cpu = image.to(torch::kCPU).contiguous(), mask_cpu = mask.to(torch::kCPU).to(torch::kBool).contiguous();
    auto mask_values = mask_cpu.accessor<bool, 2>();
    const int height = static_cast<int>(image_cpu.size(0)), width = static_cast<int>(image_cpu.size(1));
    for (int y = radius; y < height - radius; ++y) for (int x = radius; x < width - radius; ++x)
        if (mask_values[y][x]) points.emplace_back(x, y);
    if (points.empty()) return selected;
    int xmin = width, xmax = 0, ymin = height, ymax = 0;
    for (const auto& point : points) { xmin = std::min(xmin, point.first); xmax = std::max(xmax, point.first);
        ymin = std::min(ymin, point.second); ymax = std::max(ymax, point.second); }
    const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count)))));
    for (int gy = 0; gy < grid; ++gy) for (int gx = 0; gx < grid; ++gx) {
        const double x0 = xmin + (xmax + 1.0 - xmin) * gx / grid;
        const double x1 = xmin + (xmax + 1.0 - xmin) * (gx + 1) / grid;
        const double y0 = ymin + (ymax + 1.0 - ymin) * gy / grid;
        const double y1 = ymin + (ymax + 1.0 - ymin) * (gy + 1) / grid;
        std::vector<std::pair<int, int>> cell;
        for (const auto& point : points) if (point.first >= x0 && point.first < x1 && point.second >= y0 && point.second < y1)
            cell.push_back(point);
        std::shuffle(cell.begin(), cell.end(), generator);
        for (size_t index = 0; index < std::min<size_t>(20, cell.size()); ++index) {
            const auto [x, y] = cell[index];
            const auto patch = image_cpu.index({Slice(y - radius, y + radius + 1), Slice(x - radius, x + radius + 1)});
            if (patch.std(false).item<double>() >= min_texture_std) {
                selected.emplace_back(x, y); break;
            }
        }
        if (static_cast<int>(selected.size()) >= count) return selected;
    }
    std::shuffle(points.begin(), points.end(), generator);
    std::set<std::pair<int, int>> existing(selected.begin(), selected.end());
    for (const auto& point : points) {
        if (existing.count(point)) continue;
        const auto [x, y] = point;
        const auto patch = image_cpu.index({Slice(y - radius, y + radius + 1), Slice(x - radius, x + radius + 1)});
        if (patch.std(false).item<double>() >= min_texture_std) { selected.push_back(point); existing.insert(point); }
        if (static_cast<int>(selected.size()) >= count) break;
    }
    return selected;
}

std::vector<int> neighbours(int source, int views, int requested) {
    std::vector<int> result;
    for (int offset = 1; static_cast<int>(result.size()) < requested && offset < views; ++offset) {
        for (int candidate : {(source - offset + views) % views, (source + offset) % views}) {
            if (candidate != source && std::find(result.begin(), result.end(), candidate) == result.end()) result.push_back(candidate);
            if (static_cast<int>(result.size()) == requested) break;
        }
    }
    return result;
}
}  // namespace

NDeFSparsePrecalculationResult NDeFSparsePrecalculator::solve(
    const torch::Tensor& reference_images, const torch::Tensor& current_images, const torch::Tensor& roi_masks,
    const torch::Tensor& reference_visibility, const torch::Tensor& reference_projected_uv,
    const std::vector<CameraModel>& cameras) const {
    const auto valid_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    if (reference_images.dim() != 3 || !reference_images.is_floating_point() || current_images.sizes() != reference_images.sizes() ||
        roi_masks.sizes() != reference_images.sizes() || reference_visibility.dim() != 2 ||
        reference_projected_uv.dim() != 3 || reference_projected_uv.size(2) != 2 ||
        reference_visibility.size(0) != reference_projected_uv.size(0) ||
        reference_visibility.size(1) != reference_images.size(0) || reference_projected_uv.size(1) != reference_images.size(0) ||
        cameras.size() != static_cast<size_t>(reference_images.size(0)) || cameras.size() < 2 || options_.points_per_camera < 1 ||
        options_.patch_radius < 0 || options_.neighbors_per_camera < 1 || options_.match_batch_size < 1 || options_.random_seed < 0)
        throw ValidationError("NDeF sparse precalculation expects matching [V,H,W] images/masks and surface observations [M,V,(2)]");
    for (const auto& camera : cameras) camera.validate();
    const int views = static_cast<int>(reference_images.size(0));
    std::vector<torch::Tensor> refs, curs, masks;
    for (int view = 0; view < views; ++view) {
        refs.push_back(normalized_image(reference_images[view]));
        curs.push_back(normalized_image(current_images[view]));
        masks.push_back(roi_masks[view].to(torch::kBool));
    }
    const auto uv_cpu = reference_projected_uv.detach().to(torch::kCPU).to(torch::kFloat64);
    const auto vis_cpu = reference_visibility.detach().to(torch::kCPU).to(torch::kBool);
    std::vector<std::vector<std::pair<double, double>>> offsets(views, std::vector<std::pair<double, double>>(views));
    for (int source = 0; source < views; ++source) for (int target = 0; target < views; ++target) {
        auto both = vis_cpu.select(1, source) & vis_cpu.select(1, target);
        auto delta = uv_cpu.select(1, target).index({both}) - uv_cpu.select(1, source).index({both});
        if (delta.numel() > 0) {
            auto median = std::get<0>(delta.median(0));
            offsets[source][target] = {median[0].item<double>(), median[1].item<double>()};
        }
    }
    std::vector<int64_t> source_camera, camera_count;
    std::vector<double> source_uv, ref_observations, cur_observations, scores;
    std::vector<bool> observation_valid;
    std::mt19937_64 generator(static_cast<uint64_t>(options_.random_seed));
    for (int source = 0; source < views; ++source) {
        const auto seeds = sample_roi(refs[source], masks[source], options_.points_per_camera, options_.patch_radius,
                                      options_.min_texture_std, generator);
        const auto targets = neighbours(source, views, std::min(options_.neighbors_per_camera, views - 1));
        if (seeds.empty()) continue;
        std::vector<float> seed_values; seed_values.reserve(seeds.size() * 2);
        for (const auto& [x, y] : seeds) seed_values.insert(seed_values.end(), {static_cast<float>(x), static_cast<float>(y)});
        auto seed_tensor = torch::from_blob(seed_values.data(), {static_cast<int64_t>(seeds.size()), 2},
            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(refs[source].device());
        auto temporal = match_ncc_batch(refs[source], curs[source], seed_tensor, seed_tensor,
            options_.patch_radius, options_.temporal_search_radius, options_.match_batch_size);
        temporal.valid.logical_and_(temporal.score >= options_.temporal_ncc_threshold);
        std::vector<std::vector<std::pair<double, double>>> ref_uv(seeds.size(), std::vector<std::pair<double, double>>(views));
        std::vector<std::vector<std::pair<double, double>>> cur_uv(seeds.size(), std::vector<std::pair<double, double>>(views));
        std::vector<std::vector<bool>> observed(seeds.size(), std::vector<bool>(views, false));
        std::vector<double> score_sum(seeds.size(), 0.0); std::vector<int> score_count(seeds.size(), 0);
        auto temporal_uv_cpu = temporal.uv.cpu(), temporal_score_cpu = temporal.score.cpu(), temporal_valid_cpu = temporal.valid.cpu();
        for (size_t index = 0; index < seeds.size(); ++index) {
            const auto [x, y] = seeds[index];
            if (temporal_valid_cpu.index({static_cast<int64_t>(index)}).item<bool>()) {
                ref_uv[index][source] = {static_cast<double>(x), static_cast<double>(y)};
                cur_uv[index][source] = {temporal_uv_cpu.index({static_cast<int64_t>(index), 0}).item<double>(),
                                         temporal_uv_cpu.index({static_cast<int64_t>(index), 1}).item<double>()};
                observed[index][source] = true; score_sum[index] = temporal_score_cpu.index({static_cast<int64_t>(index)}).item<double>();
                score_count[index] = 1;
            }
        }
        for (int target : targets) {
            const auto offset = offsets[source][target];
            auto predicted = seed_tensor + torch::tensor({offset.first, offset.second}, seed_tensor.options());
            auto cross = match_ncc_batch(refs[source], refs[target], seed_tensor, predicted,
                options_.patch_radius, options_.cross_search_radius, options_.match_batch_size);
            cross.valid.logical_and_(cross.score >= options_.cross_ncc_threshold);
            auto target_temporal = match_ncc_batch(refs[target], curs[target], cross.uv, cross.uv,
                options_.patch_radius, options_.temporal_search_radius, options_.match_batch_size);
            auto target_valid = cross.valid & target_temporal.valid &
                                (target_temporal.score >= options_.temporal_ncc_threshold);
            auto cross_uv_cpu = cross.uv.cpu(), cross_score_cpu = cross.score.cpu();
            auto target_uv_cpu = target_temporal.uv.cpu(), target_score_cpu = target_temporal.score.cpu();
            auto target_valid_cpu = target_valid.cpu();
            for (size_t index = 0; index < seeds.size(); ++index) if (target_valid_cpu.index({static_cast<int64_t>(index)}).item<bool>()) {
                ref_uv[index][target] = {cross_uv_cpu.index({static_cast<int64_t>(index), 0}).item<double>(),
                                         cross_uv_cpu.index({static_cast<int64_t>(index), 1}).item<double>()};
                cur_uv[index][target] = {target_uv_cpu.index({static_cast<int64_t>(index), 0}).item<double>(),
                                         target_uv_cpu.index({static_cast<int64_t>(index), 1}).item<double>()};
                observed[index][target] = true;
                score_sum[index] += cross_score_cpu.index({static_cast<int64_t>(index)}).item<double>() +
                                    target_score_cpu.index({static_cast<int64_t>(index)}).item<double>();
                score_count[index] += 2;
            }
        }
        for (size_t index = 0; index < seeds.size(); ++index) {
            const auto count = static_cast<int>(std::count(observed[index].begin(), observed[index].end(), true));
            if (count < 2) continue;
            const auto [x, y] = seeds[index];
            source_camera.push_back(source); source_uv.insert(source_uv.end(), {static_cast<double>(x), static_cast<double>(y)});
            camera_count.push_back(count); scores.push_back(score_sum[index] / score_count[index]);
            for (int view = 0; view < views; ++view) {
                ref_observations.insert(ref_observations.end(), {ref_uv[index][view].first, ref_uv[index][view].second});
                cur_observations.insert(cur_observations.end(), {cur_uv[index][view].first, cur_uv[index][view].second});
                observation_valid.push_back(observed[index][view]);
            }
        }
    }
    const auto count = static_cast<int64_t>(source_camera.size());
    if (count == 0) throw ValidationError("NDeF sparse precalculation found no two-view NCC tracks");
    auto doubles = torch::TensorOptions().dtype(torch::kFloat64);
    auto refs_uv = torch::from_blob(ref_observations.data(), {count, views, 2}, doubles).clone();
    auto curs_uv = torch::from_blob(cur_observations.data(), {count, views, 2}, doubles).clone();
    auto observed = torch::zeros({count, views}, valid_options);
    for (int64_t row = 0; row < count; ++row) for (int view = 0; view < views; ++view) {
        if (observation_valid[static_cast<size_t>(row * views + view)]) observed.index_put_({row, view}, true);
    }
    ReconstructionOptions reconstruction_options; reconstruction_options.max_reprojection_error = options_.max_reprojection_error;
    auto reference = triangulate_multiview(refs_uv, cameras, observed, reconstruction_options);
    auto current = triangulate_multiview(curs_uv, cameras, observed, reconstruction_options);
    auto good = reference.valid & current.valid;
    auto ids = torch::nonzero(good).reshape({-1});
    if (ids.numel() == 0) throw ValidationError("NDeF sparse precalculation rejected all tracks by triangulation/reprojection");
    auto displacement = current.points.index_select(0, ids) - reference.points.index_select(0, ids);
    auto scale = estimate_ndef_displacement_scale(displacement, options_.displacement_mad_threshold);
    return {torch::from_blob(source_camera.data(), {count}, torch::TensorOptions().dtype(torch::kInt64)).clone().index_select(0, ids),
            torch::from_blob(source_uv.data(), {count, 2}, doubles).clone().index_select(0, ids),
            reference.points.index_select(0, ids), current.points.index_select(0, ids), displacement,
            torch::linalg_vector_norm(displacement, 2, 1),
            torch::from_blob(camera_count.data(), {count}, torch::TensorOptions().dtype(torch::kInt64)).clone().index_select(0, ids),
            reference.mean_reprojection_error.index_select(0, ids), current.mean_reprojection_error.index_select(0, ids),
            torch::from_blob(scores.data(), {count}, doubles).clone().index_select(0, ids), scale.inlier_mask, scale};
}

NDeFDenseSurfaceSampleResult NDeFDenseSurfaceSampler::sample(const torch::Tensor& dense_points,
                                                               const torch::Tensor& roi_masks,
                                                               const std::vector<CameraModel>& cameras) const {
    if (dense_points.dim() != 2 || dense_points.size(1) != 3 || !dense_points.is_floating_point() ||
        roi_masks.dim() != 3 || roi_masks.size(0) != static_cast<int64_t>(cameras.size()) || cameras.size() < 2 ||
        options_.max_points < 1 || options_.min_visible_views < 1)
        throw ValidationError("NDeF dense sampler expects dense_points[N,3], ROI masks[V,H,W], and at least two cameras");
    for (const auto& camera : cameras) camera.validate();
    auto candidates = dense_points.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    candidates = candidates.index({torch::isfinite(candidates).all(1)});
    if (candidates.numel() == 0) throw ValidationError("NDeF dense sampler received no finite reconstructed points");
    if (options_.voxel_size > 0.0) {
        std::vector<int64_t> keep; std::set<std::tuple<int64_t, int64_t, int64_t>> voxels;
        auto access = candidates.accessor<double, 2>();
        for (int64_t i = 0; i < candidates.size(0); ++i) {
            auto key = std::make_tuple(static_cast<int64_t>(std::floor(access[i][0] / options_.voxel_size)),
                                       static_cast<int64_t>(std::floor(access[i][1] / options_.voxel_size)),
                                       static_cast<int64_t>(std::floor(access[i][2] / options_.voxel_size)));
            if (voxels.insert(key).second) keep.push_back(i);
        }
        candidates = candidates.index_select(0, torch::tensor(keep, torch::kLong));
    }
    if (candidates.size(0) > options_.max_points) {
        auto ids = torch::linspace(0, candidates.size(0) - 1, options_.max_points,
                                  torch::TensorOptions().dtype(torch::kLong));
        candidates = candidates.index_select(0, ids);
    }
    const int views = static_cast<int>(cameras.size());
    std::vector<torch::Tensor> ks, rs, ts, ds;
    for (const auto& camera : cameras) { ks.push_back(camera.intrinsics); rs.push_back(camera.rotation); ts.push_back(camera.translation); ds.push_back(camera.distortion); }
    auto projected = project_points_multi_view(candidates, torch::stack(ks), torch::stack(rs), torch::stack(ts), torch::stack(ds));
    auto visible = torch::zeros({candidates.size(0), views}, torch::TensorOptions().dtype(torch::kBool));
    auto uv = projected.uv.to(torch::kCPU); auto depth = projected.depth.to(torch::kCPU); auto masks = roi_masks.to(torch::kCPU).to(torch::kBool);
    for (int64_t point = 0; point < candidates.size(0); ++point) for (int view = 0; view < views; ++view) {
        const int x = static_cast<int>(std::lround(uv[point][view][0].item<double>()));
        const int y = static_cast<int>(std::lround(uv[point][view][1].item<double>()));
        if (depth[point][view].item<double>() > 0 && x >= 0 && y >= 0 && x < masks.size(2) && y < masks.size(1) && masks[view][y][x].item<bool>()) visible.index_put_({point, view}, true);
    }
    auto keep = torch::nonzero(visible.sum(1) >= options_.min_visible_views).reshape({-1});
    if (keep.numel() == 0) throw ValidationError("NDeF dense sampler found no ROI-visible multi-view surface points");
    auto kept_visible = visible.index_select(0, keep);
    return {candidates.index_select(0, keep), kept_visible, uv.index_select(0, keep), kept_visible.sum(1).to(torch::kFloat)};
}

}  // namespace neurodic
