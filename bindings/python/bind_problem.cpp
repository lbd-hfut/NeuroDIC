#include <pybind11/pybind11.h>

#include "neurodic/problem/pin_problem.hpp"

namespace py = pybind11;

void bind_problem(py::module_& module) {
    py::class_<neurodic::PINProblem>(module, "PINProblem")
        .def(py::init<neurodic::GeometryType>())
        .def("validate", &neurodic::PINProblem::validate);
}
