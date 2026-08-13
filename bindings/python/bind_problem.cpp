#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <string>

#include "neurodic/problem/pin_problem.hpp"
#include "neurodic/problem/pin_stereo_problem.hpp"
#include "neurodic/problem/pin_multi_problem.hpp"
#include "neurodic/problem/ndef_problem.hpp"
#include "neurodic/problem/ndef_surface_problem.hpp"

namespace py = pybind11;

void bind_problem(py::module_& module) {
    py::class_<neurodic::ImagePrecomputeOptions>(module, "ImagePrecomputeOptions")
        .def(py::init<>())
        .def_readwrite("integer_search_radius", &neurodic::ImagePrecomputeOptions::integer_search_radius)
        .def_readwrite("coarse_subset_radius", &neurodic::ImagePrecomputeOptions::coarse_subset_radius)
        .def_readwrite("fine_subset_radius", &neurodic::ImagePrecomputeOptions::fine_subset_radius)
        .def_readwrite("subset_radius", &neurodic::ImagePrecomputeOptions::subset_radius)
        .def_readwrite("bspline_border", &neurodic::ImagePrecomputeOptions::bspline_border)
        .def_readwrite("bspline_degree", &neurodic::ImagePrecomputeOptions::bspline_degree);

    py::class_<neurodic::PINProblem>(module, "PINProblem")
        .def(py::init<neurodic::GeometryType>())
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, neurodic::SeedSet,
                      neurodic::PINModelOptions, neurodic::ImagePrecomputeOptions>(),
             py::arg("reference"), py::arg("deformed"), py::arg("roi_mask"), py::arg("seeds"),
             py::arg("model_options") = neurodic::PINModelOptions{},
             py::arg("precompute_options") = neurodic::ImagePrecomputeOptions{})
        .def_readwrite("seed_iterations", &neurodic::PINProblem::seed_iterations)
        .def_readwrite("seed_pretrain_uv_scale_threshold", &neurodic::PINProblem::seed_pretrain_uv_scale_threshold)
        .def_readwrite("photometric_iterations", &neurodic::PINProblem::photometric_iterations)
        .def_readwrite("photometric_sample_count", &neurodic::PINProblem::photometric_sample_count)
        .def_readwrite("photometric_sampling_enabled", &neurodic::PINProblem::photometric_sampling_enabled)
        .def_readwrite("znssd_kernel_size", &neurodic::PINProblem::znssd_kernel_size)
        .def_readwrite("seed_learning_rate", &neurodic::PINProblem::seed_learning_rate)
        .def_readwrite("photometric_learning_rate", &neurodic::PINProblem::photometric_learning_rate)
        .def_readwrite("photometric_loss", &neurodic::PINProblem::photometric_loss)
        .def_readwrite("compute_neural_strain_2d", &neurodic::PINProblem::compute_neural_strain_2d)
        .def_readwrite("evaluation_enabled", &neurodic::PINProblem::evaluation_enabled)
        .def_readwrite("evaluation_sample_count", &neurodic::PINProblem::evaluation_sample_count)
        .def_readwrite("evaluation_seed", &neurodic::PINProblem::evaluation_seed)
        .def_readwrite("evaluation_patch_radius", &neurodic::PINProblem::evaluation_patch_radius)
        .def("set_device", [](neurodic::PINProblem& problem, const std::string& device) {
            problem.device = torch::Device(device);
        })
        .def("validate", &neurodic::PINProblem::validate);

    py::class_<neurodic::PINStereoProblem>(module, "PINStereoProblem")
        .def(py::init<neurodic::PINProblem, neurodic::PINProblem, neurodic::PINProblem,
                      neurodic::CameraModel, neurodic::CameraModel>(),
             py::arg("reference_disparity"), py::arg("left_temporal"),
             py::arg("deformed_disparity"), py::arg("left_camera"), py::arg("right_camera"))
        .def_readwrite("world_scale", &neurodic::PINStereoProblem::world_scale)
        .def_readwrite("require_image_bounds", &neurodic::PINStereoProblem::require_image_bounds)
        .def_readwrite("compute_traditional_strain", &neurodic::PINStereoProblem::compute_traditional_strain)
        .def_readwrite("traditional_strain_neighbors", &neurodic::PINStereoProblem::traditional_strain_neighbors)
        .def("set_reconstruction_options", [](neurodic::PINStereoProblem& problem, double max_error,
                                                bool positive_depth, int undistort_iterations) {
            problem.reconstruction.max_reprojection_error = max_error;
            problem.reconstruction.require_positive_depth = positive_depth;
            problem.reconstruction.undistort_iterations = undistort_iterations;
        }, py::arg("max_reprojection_error"), py::arg("require_positive_depth") = true,
           py::arg("undistort_iterations") = 12)
        .def("validate", &neurodic::PINStereoProblem::validate);

    py::class_<neurodic::PINMultiProblem>(module, "PINMultiProblem")
        .def(py::init<>())
        .def_readwrite("route_id", &neurodic::PINMultiProblem::route_id)
        .def_readwrite("world_scale", &neurodic::PINMultiProblem::world_scale)
        .def_readwrite("require_image_bounds", &neurodic::PINMultiProblem::require_image_bounds)
        .def_readwrite("remove_rigid_body_motion", &neurodic::PINMultiProblem::remove_rigid_body_motion)
        .def("add_pair", [](neurodic::PINMultiProblem& problem, const std::string& pair_id,
                            const neurodic::PINProblem& reference_stereo, const neurodic::PINProblem& left_temporal,
                            const neurodic::PINProblem& deformed_stereo, const neurodic::CameraModel& left_camera,
                            const neurodic::CameraModel& right_camera) {
            problem.pairs.push_back({pair_id, reference_stereo, left_temporal, deformed_stereo,
                                     left_camera, right_camera});
        }, py::arg("pair_id"), py::arg("reference_stereo"), py::arg("left_temporal"),
           py::arg("deformed_stereo"), py::arg("left_camera"), py::arg("right_camera"))
        .def("set_reconstruction_options", [](neurodic::PINMultiProblem& problem, double max_error,
                                               bool positive_depth, int undistort_iterations) {
            problem.reconstruction.max_reprojection_error = max_error;
            problem.reconstruction.require_positive_depth = positive_depth;
            problem.reconstruction.undistort_iterations = undistort_iterations;
        }, py::arg("max_reprojection_error"), py::arg("require_positive_depth") = true,
           py::arg("undistort_iterations") = 12)
        .def("validate", &neurodic::PINMultiProblem::validate);

    py::class_<neurodic::NDeFProblem>(module, "NDeFProblem")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                      std::vector<neurodic::CameraModel>>(),
             py::arg("reference_surface"), py::arg("reference_images"), py::arg("deformed_images"),
             py::arg("reference_masks"), py::arg("deformed_masks"), py::arg("cameras"))
        .def_readwrite("model_options", &neurodic::NDeFProblem::model_options)
        .def_readwrite("training_epochs", &neurodic::NDeFProblem::training_epochs)
        .def_readwrite("batch_size", &neurodic::NDeFProblem::batch_size)
        .def_readwrite("auto_batch_start", &neurodic::NDeFProblem::auto_batch_start)
        .def_readwrite("auto_batch_max", &neurodic::NDeFProblem::auto_batch_max)
        .def_readwrite("memory_fraction", &neurodic::NDeFProblem::memory_fraction)
        .def_readwrite("max_steps_per_epoch", &neurodic::NDeFProblem::max_steps_per_epoch)
        .def_readwrite("prediction_batch_size", &neurodic::NDeFProblem::prediction_batch_size)
        .def_readwrite("random_seed", &neurodic::NDeFProblem::random_seed)
        .def_readwrite("photometric_iterations", &neurodic::NDeFProblem::photometric_iterations)
        .def_readwrite("photometric_sample_count", &neurodic::NDeFProblem::photometric_sample_count)
        .def_readwrite("bspline_degree", &neurodic::NDeFProblem::bspline_degree)
        .def_readwrite("photometric_learning_rate", &neurodic::NDeFProblem::photometric_learning_rate)
        .def_readwrite("weight_decay", &neurodic::NDeFProblem::weight_decay)
        .def_readwrite("smoothness_weight", &neurodic::NDeFProblem::smoothness_weight)
        .def_readwrite("patch_radius", &neurodic::NDeFProblem::patch_radius)
        .def_readwrite("min_valid_patch_ratio", &neurodic::NDeFProblem::min_valid_patch_ratio)
        .def_readwrite("invalid_patch_penalty", &neurodic::NDeFProblem::invalid_patch_penalty)
        .def_readwrite("sfm_to_world_scale", &neurodic::NDeFProblem::sfm_to_world_scale)
        .def_readwrite("photometric_loss", &neurodic::NDeFProblem::photometric_loss)
        .def_readwrite("evaluation_enabled", &neurodic::NDeFProblem::evaluation_enabled)
        .def_readwrite("evaluation_sample_count", &neurodic::NDeFProblem::evaluation_sample_count)
        .def_readwrite("evaluation_seed", &neurodic::NDeFProblem::evaluation_seed)
        .def("set_device", [](neurodic::NDeFProblem& problem, const std::string& device) {
            problem.device = torch::Device(device);
        })
        .def("set_surface_observations", [](neurodic::NDeFProblem& problem, torch::Tensor visibility,
                                               torch::Tensor projected_uv, torch::Tensor visible_counts) {
            problem.reference_visibility = visibility.detach().to(torch::kCPU).to(torch::kBool).contiguous();
            problem.reference_projected_uv = projected_uv.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            problem.visible_counts = visible_counts.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        }, py::arg("visibility"), py::arg("projected_uv"), py::arg("visible_counts"))
        .def("validate", &neurodic::NDeFProblem::validate);
    py::class_<neurodic::NDeFDepthModelOptions>(module,"NDeFDepthModelOptions").def(py::init<>()).def_readwrite("hidden_dim",&neurodic::NDeFDepthModelOptions::hidden_dim).def_readwrite("pixel_layers",&neurodic::NDeFDepthModelOptions::pixel_layers).def_readwrite("camera_layers",&neurodic::NDeFDepthModelOptions::camera_layers).def_readwrite("trunk_layers",&neurodic::NDeFDepthModelOptions::trunk_layers).def_readwrite("camera_embedding_dim",&neurodic::NDeFDepthModelOptions::camera_embedding_dim).def_readwrite("positional_encoding_enabled",&neurodic::NDeFDepthModelOptions::positional_encoding_enabled).def_readwrite("positional_encoding_num_frequencies",&neurodic::NDeFDepthModelOptions::positional_encoding_num_frequencies);
    py::class_<neurodic::NDeFSurfaceProblem>(module,"NDeFSurfaceProblem").def(py::init<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>()).def_readwrite("model_options",&neurodic::NDeFSurfaceProblem::model_options).def_readwrite("pretrain_iterations",&neurodic::NDeFSurfaceProblem::pretrain_iterations).def_readwrite("pretrain_learning_rate",&neurodic::NDeFSurfaceProblem::pretrain_learning_rate).def_readwrite("weight_decay",&neurodic::NDeFSurfaceProblem::weight_decay).def_readwrite("smoothness_weight",&neurodic::NDeFSurfaceProblem::smoothness_weight).def_readwrite("smooth_samples_per_camera",&neurodic::NDeFSurfaceProblem::smooth_samples_per_camera).def_readwrite("dense_iterations",&neurodic::NDeFSurfaceProblem::dense_iterations).def_readwrite("dense_epochs",&neurodic::NDeFSurfaceProblem::dense_epochs).def_readwrite("dense_samples_per_camera",&neurodic::NDeFSurfaceProblem::dense_samples_per_camera).def_readwrite("dense_auto_batch",&neurodic::NDeFSurfaceProblem::dense_auto_batch).def_readwrite("dense_auto_batch_start",&neurodic::NDeFSurfaceProblem::dense_auto_batch_start).def_readwrite("dense_memory_fraction",&neurodic::NDeFSurfaceProblem::dense_memory_fraction).def_readwrite("dense_spacing_px",&neurodic::NDeFSurfaceProblem::dense_spacing_px).def_readwrite("dense_patch_radius",&neurodic::NDeFSurfaceProblem::dense_patch_radius).def_readwrite("dense_learning_rate",&neurodic::NDeFSurfaceProblem::dense_learning_rate).def_readwrite("dense_anchor_weight",&neurodic::NDeFSurfaceProblem::dense_anchor_weight).def_readwrite("dense_min_valid_patch_ratio",&neurodic::NDeFSurfaceProblem::dense_min_valid_patch_ratio).def_readwrite("dense_seed",&neurodic::NDeFSurfaceProblem::dense_seed).def_readwrite("prediction_batch_size",&neurodic::NDeFSurfaceProblem::prediction_batch_size).def("set_dense_inputs",&neurodic::NDeFSurfaceProblem::set_dense_inputs).def("set_device",[](neurodic::NDeFSurfaceProblem& p,const std::string& v){p.device=torch::Device(v);});
}
