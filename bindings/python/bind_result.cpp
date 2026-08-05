#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "neurodic/core/result.hpp"

namespace py = pybind11;

void bind_result(py::module_& module) {
    py::class_<neurodic::SolverDiagnostics>(module, "SolverDiagnostics")
        .def_readonly("iterations", &neurodic::SolverDiagnostics::iterations)
        .def_readonly("final_loss", &neurodic::SolverDiagnostics::final_loss)
        .def_readonly("metrics", &neurodic::SolverDiagnostics::metrics);
    py::class_<neurodic::FieldResult>(module, "FieldResult")
        .def_readonly("coordinates", &neurodic::FieldResult::coordinates)
        .def_readonly("values", &neurodic::FieldResult::values);
    py::class_<neurodic::PINResult>(module, "PINResult")
        .def_readonly("displacement", &neurodic::PINResult::displacement)
        .def_readonly("diagnostics", &neurodic::PINResult::diagnostics);
    py::class_<neurodic::PINStereoResult>(module, "PINStereoResult")
        .def_readonly("reference_disparity", &neurodic::PINStereoResult::reference_disparity)
        .def_readonly("left_temporal", &neurodic::PINStereoResult::left_temporal)
        .def_readonly("deformed_disparity", &neurodic::PINStereoResult::deformed_disparity)
        .def_readonly("left_reference_coordinates", &neurodic::PINStereoResult::left_reference_coordinates)
        .def_readonly("right_reference_coordinates", &neurodic::PINStereoResult::right_reference_coordinates)
        .def_readonly("left_current_coordinates", &neurodic::PINStereoResult::left_current_coordinates)
        .def_readonly("right_current_coordinates", &neurodic::PINStereoResult::right_current_coordinates)
        .def_readonly("reference_points", &neurodic::PINStereoResult::reference_points)
        .def_readonly("current_points", &neurodic::PINStereoResult::current_points)
        .def_readonly("displacement_3d", &neurodic::PINStereoResult::displacement_3d)
        .def_readonly("valid", &neurodic::PINStereoResult::valid)
        .def_readonly("reference_reprojection_error", &neurodic::PINStereoResult::reference_reprojection_error)
        .def_readonly("current_reprojection_error", &neurodic::PINStereoResult::current_reprojection_error);
}
