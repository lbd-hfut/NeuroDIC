#include <cstdint>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "neurodic/initialization/seed_set.hpp"
#include "neurodic/initialization/sift_grid_seed_initializer.hpp"
#include "neurodic/initialization/traditional_seed_initializer.hpp"
#include "neurodic/initialization/ndef_precalculation.hpp"
#include "neurodic/model/fourier.hpp"
#include "neurodic/model/mlp.hpp"
#include "neurodic/model/ndef_internal_model.hpp"
#include "neurodic/representation/pin_displacement_field.hpp"

namespace py = pybind11;

void bind_initialization(py::module_& module) {
    module.def("_initialization_bindings_ready", [] { return true; });
    py::class_<neurodic::SeedSet>(module, "SeedSet")
        .def_readonly("seed_pos", &neurodic::SeedSet::seed_pos)
        .def_readonly("seed_uv", &neurodic::SeedSet::seed_uv)
        .def_readonly("scale_uv", &neurodic::SeedSet::scale_uv)
        .def("validate", &neurodic::SeedSet::validate);
    module.def("make_seed_set", &neurodic::SeedSet::from_tensors,
               py::arg("seed_pos"), py::arg("seed_uv"));
    py::class_<neurodic::NDeFDisplacementScale>(module, "NDeFDisplacementScale")
        .def_readonly("inlier_mask", &neurodic::NDeFDisplacementScale::inlier_mask)
        .def_readonly("median", &neurodic::NDeFDisplacementScale::median)
        .def_readonly("mean", &neurodic::NDeFDisplacementScale::mean)
        .def_readonly("p75", &neurodic::NDeFDisplacementScale::p75)
        .def_readonly("p90", &neurodic::NDeFDisplacementScale::p90)
        .def_readonly("maximum", &neurodic::NDeFDisplacementScale::maximum);
    module.def("estimate_ndef_displacement_scale", &neurodic::estimate_ndef_displacement_scale,
               py::arg("displacement"), py::arg("mad_threshold") = 5.0);
    py::class_<neurodic::NDeFSparsePrecalculationOptions>(module, "NDeFSparsePrecalculationOptions")
        .def(py::init<>()).def_readwrite("points_per_camera", &neurodic::NDeFSparsePrecalculationOptions::points_per_camera)
        .def_readwrite("neighbors_per_camera", &neurodic::NDeFSparsePrecalculationOptions::neighbors_per_camera)
        .def_readwrite("patch_radius", &neurodic::NDeFSparsePrecalculationOptions::patch_radius)
        .def_readwrite("cross_search_radius", &neurodic::NDeFSparsePrecalculationOptions::cross_search_radius)
        .def_readwrite("temporal_search_radius", &neurodic::NDeFSparsePrecalculationOptions::temporal_search_radius)
        .def_readwrite("cross_ncc_threshold", &neurodic::NDeFSparsePrecalculationOptions::cross_ncc_threshold)
        .def_readwrite("temporal_ncc_threshold", &neurodic::NDeFSparsePrecalculationOptions::temporal_ncc_threshold)
        .def_readwrite("min_texture_std", &neurodic::NDeFSparsePrecalculationOptions::min_texture_std)
        .def_readwrite("max_reprojection_error", &neurodic::NDeFSparsePrecalculationOptions::max_reprojection_error)
        .def_readwrite("displacement_mad_threshold", &neurodic::NDeFSparsePrecalculationOptions::displacement_mad_threshold)
        .def_readwrite("match_batch_size", &neurodic::NDeFSparsePrecalculationOptions::match_batch_size)
        .def_readwrite("random_seed", &neurodic::NDeFSparsePrecalculationOptions::random_seed);
    py::class_<neurodic::NDeFSparsePrecalculationResult>(module, "NDeFSparsePrecalculationResult")
        .def_readonly("source_camera", &neurodic::NDeFSparsePrecalculationResult::source_camera)
        .def_readonly("source_uv", &neurodic::NDeFSparsePrecalculationResult::source_uv)
        .def_readonly("reference_points", &neurodic::NDeFSparsePrecalculationResult::reference_points)
        .def_readonly("current_points", &neurodic::NDeFSparsePrecalculationResult::current_points)
        .def_readonly("displacement", &neurodic::NDeFSparsePrecalculationResult::displacement)
        .def_readonly("displacement_magnitude", &neurodic::NDeFSparsePrecalculationResult::displacement_magnitude)
        .def_readonly("camera_count", &neurodic::NDeFSparsePrecalculationResult::camera_count)
        .def_readonly("reference_reprojection_error", &neurodic::NDeFSparsePrecalculationResult::reference_reprojection_error)
        .def_readonly("current_reprojection_error", &neurodic::NDeFSparsePrecalculationResult::current_reprojection_error)
        .def_readonly("mean_match_score", &neurodic::NDeFSparsePrecalculationResult::mean_match_score)
        .def_readonly("inlier_mask", &neurodic::NDeFSparsePrecalculationResult::inlier_mask)
        .def_readonly("scale", &neurodic::NDeFSparsePrecalculationResult::scale);
    py::class_<neurodic::NDeFSparsePrecalculator>(module, "NDeFSparsePrecalculator")
        .def(py::init<neurodic::NDeFSparsePrecalculationOptions>(), py::arg("options") = neurodic::NDeFSparsePrecalculationOptions{})
        .def("solve", &neurodic::NDeFSparsePrecalculator::solve);
    py::class_<neurodic::NDeFDenseSurfaceSampleOptions>(module, "NDeFDenseSurfaceSampleOptions")
        .def(py::init<>()).def_readwrite("voxel_size", &neurodic::NDeFDenseSurfaceSampleOptions::voxel_size)
        .def_readwrite("max_points", &neurodic::NDeFDenseSurfaceSampleOptions::max_points)
        .def_readwrite("min_visible_views", &neurodic::NDeFDenseSurfaceSampleOptions::min_visible_views);
    py::class_<neurodic::NDeFDenseSurfaceSampleResult>(module, "NDeFDenseSurfaceSampleResult")
        .def_readonly("points", &neurodic::NDeFDenseSurfaceSampleResult::points)
        .def_readonly("visibility_mask", &neurodic::NDeFDenseSurfaceSampleResult::visibility_mask)
        .def_readonly("projected_uv", &neurodic::NDeFDenseSurfaceSampleResult::projected_uv)
        .def_readonly("visible_counts", &neurodic::NDeFDenseSurfaceSampleResult::visible_counts);
    py::class_<neurodic::NDeFDenseSurfaceSampler>(module, "NDeFDenseSurfaceSampler")
        .def(py::init<neurodic::NDeFDenseSurfaceSampleOptions>(), py::arg("options") = neurodic::NDeFDenseSurfaceSampleOptions{})
        .def("sample", &neurodic::NDeFDenseSurfaceSampler::sample);
    py::class_<neurodic::SiftGridSeedOptions>(module, "SiftGridSeedOptions")
        .def(py::init<>())
        .def_readwrite("target_seed_count", &neurodic::SiftGridSeedOptions::target_seed_count)
        .def_readwrite("lowe_ratio", &neurodic::SiftGridSeedOptions::lowe_ratio)
        .def_readwrite("flann_trees", &neurodic::SiftGridSeedOptions::flann_trees)
        .def_readwrite("flann_checks", &neurodic::SiftGridSeedOptions::flann_checks)
        .def_readwrite("mad_threshold", &neurodic::SiftGridSeedOptions::mad_threshold)
        .def_readwrite("min_seeds_per_roi", &neurodic::SiftGridSeedOptions::min_seeds_per_roi);
    py::class_<neurodic::SiftGridSeedInitializer>(module, "SiftGridSeedInitializer")
        .def(py::init<neurodic::SiftGridSeedOptions>(), py::arg("options") = neurodic::SiftGridSeedOptions{})
        .def("initialize", &neurodic::SiftGridSeedInitializer::initialize);
    py::class_<neurodic::SeedCleanupOptions>(module, "SeedCleanupOptions")
        .def(py::init<>())
        .def_readwrite("mad_threshold", &neurodic::SeedCleanupOptions::mad_threshold)
        .def_readwrite("min_seed_count", &neurodic::SeedCleanupOptions::min_seed_count);
    py::class_<neurodic::TraditionalSeedOptions>(module, "TraditionalSeedOptions")
        .def(py::init<>())
        .def_readwrite("target_seed_count", &neurodic::TraditionalSeedOptions::target_seed_count)
        .def_readwrite("kmeans_iterations", &neurodic::TraditionalSeedOptions::kmeans_iterations)
        .def_readwrite("kmeans_sample_limit", &neurodic::TraditionalSeedOptions::kmeans_sample_limit)
        .def_readwrite("subset_radius", &neurodic::TraditionalSeedOptions::subset_radius)
        .def_readwrite("search_radius", &neurodic::TraditionalSeedOptions::search_radius)
        .def_readwrite("pyramid_enabled", &neurodic::TraditionalSeedOptions::pyramid_enabled)
        .def_readwrite("pyramid_scale", &neurodic::TraditionalSeedOptions::pyramid_scale)
        .def_readwrite("pyramid_refinement_radius", &neurodic::TraditionalSeedOptions::pyramid_refinement_radius)
        .def_readwrite("sift_prior_enabled", &neurodic::TraditionalSeedOptions::sift_prior_enabled)
        .def_readwrite("sift_max_features", &neurodic::TraditionalSeedOptions::sift_max_features)
        .def_readwrite("sift_ratio_threshold", &neurodic::TraditionalSeedOptions::sift_ratio_threshold)
        .def_readwrite("sift_robust_mad_factor", &neurodic::TraditionalSeedOptions::sift_robust_mad_factor)
        .def_readwrite("sift_interpolation_neighbors", &neurodic::TraditionalSeedOptions::sift_interpolation_neighbors)
        .def_readwrite("sift_interpolation_radius", &neurodic::TraditionalSeedOptions::sift_interpolation_radius)
        .def_readwrite("subpixel_enabled", &neurodic::TraditionalSeedOptions::subpixel_enabled)
        .def_readwrite("subpixel_subset_radius", &neurodic::TraditionalSeedOptions::subpixel_subset_radius)
        .def_readwrite("subpixel_max_iterations", &neurodic::TraditionalSeedOptions::subpixel_max_iterations)
        .def_readwrite("subpixel_convergence_threshold", &neurodic::TraditionalSeedOptions::subpixel_convergence_threshold)
        .def_readwrite("cleanup", &neurodic::TraditionalSeedOptions::cleanup);
    py::class_<neurodic::TraditionalSeedInitializer>(module, "TraditionalSeedInitializer")
        .def(py::init<neurodic::TraditionalSeedOptions>(), py::arg("options") = neurodic::TraditionalSeedOptions{})
        .def("initialize", &neurodic::TraditionalSeedInitializer::initialize);
    py::class_<neurodic::FourierEncodingOptions>(module, "FourierEncodingOptions")
        .def(py::init<>())
        .def_readwrite("enabled", &neurodic::FourierEncodingOptions::enabled)
        .def_readwrite("num_frequencies", &neurodic::FourierEncodingOptions::num_frequencies)
        .def_readwrite("include_input", &neurodic::FourierEncodingOptions::include_input)
        .def_readwrite("angular_scale", &neurodic::FourierEncodingOptions::angular_scale);
    py::class_<neurodic::PINModelOptions>(module, "PINModelOptions")
        .def(py::init<>())
        .def_readwrite("input_dim", &neurodic::PINModelOptions::input_dim)
        .def_readwrite("output_dim", &neurodic::PINModelOptions::output_dim)
        .def_readwrite("hidden_dim", &neurodic::PINModelOptions::hidden_dim)
        .def_readwrite("hidden_layers", &neurodic::PINModelOptions::hidden_layers)
        .def_readwrite("fourier_encoding", &neurodic::PINModelOptions::fourier_encoding);
    py::class_<neurodic::NDeFModelOptions>(module, "NDeFModelOptions")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &neurodic::NDeFModelOptions::hidden_dim)
        .def_readwrite("hidden_layers", &neurodic::NDeFModelOptions::hidden_layers)
        .def_readwrite("output_scale", &neurodic::NDeFModelOptions::output_scale)
        .def_readwrite("fourier_encoding", &neurodic::NDeFModelOptions::fourier_encoding);
    py::class_<neurodic::MLPModel, std::shared_ptr<neurodic::MLPModel>>(module, "PINMLPModel")
        .def(py::init<neurodic::PINModelOptions>(), py::arg("options") = neurodic::PINModelOptions{})
        .def("forward", &neurodic::MLPModel::forward)
        .def("train_mode", [](neurodic::MLPModel& model, bool enabled) { model.train(enabled); })
        .def("parameter_count", [](const neurodic::MLPModel& model) {
            std::int64_t count = 0;
            for (const auto& parameter : model.parameters()) count += parameter.numel();
            return count;
        });
    module.def("decode_pin_displacement", [](const torch::Tensor& raw, const torch::Tensor& scale_uv) {
        if (scale_uv.numel() != 4) throw std::invalid_argument("scale_uv must have four values");
        neurodic::PINDisplacementField field({scale_uv.slice(0, 0, 2), scale_uv.slice(0, 2, 4)});
        return field.decode(torch::Tensor(), raw);
    });
}
