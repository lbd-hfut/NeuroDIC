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
}
