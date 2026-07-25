#include <pybind11/pybind11.h>

#include "neurodic/solver/ndef_solver.hpp"
#include "neurodic/solver/pin_solver.hpp"

namespace py = pybind11;

void bind_solver(py::module_& module) {
    py::class_<neurodic::PINSolver>(module, "PINSolver").def(py::init<>());
    py::class_<neurodic::NDeFSolver>(module, "NDeFSolver").def(py::init<>());
}
