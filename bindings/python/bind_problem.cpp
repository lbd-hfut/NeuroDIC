#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <string>

#include "neurodic/problem/pin_problem.hpp"

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
        .def_readwrite("photometric_iterations", &neurodic::PINProblem::photometric_iterations)
        .def_readwrite("photometric_sample_count", &neurodic::PINProblem::photometric_sample_count)
        .def_readwrite("photometric_sampling_enabled", &neurodic::PINProblem::photometric_sampling_enabled)
        .def_readwrite("znssd_kernel_size", &neurodic::PINProblem::znssd_kernel_size)
        .def_readwrite("seed_learning_rate", &neurodic::PINProblem::seed_learning_rate)
        .def_readwrite("photometric_learning_rate", &neurodic::PINProblem::photometric_learning_rate)
        .def_readwrite("photometric_loss", &neurodic::PINProblem::photometric_loss)
        .def("set_device", [](neurodic::PINProblem& problem, const std::string& device) {
            problem.device = torch::Device(device);
        })
        .def("validate", &neurodic::PINProblem::validate);
}
