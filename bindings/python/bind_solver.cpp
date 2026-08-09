#include <pybind11/pybind11.h>

#include "neurodic/solver/ndef_solver.hpp"
#include "neurodic/solver/ndef_surface_solver.hpp"
#include "neurodic/solver/pin_solver.hpp"
#include "neurodic/solver/pin_stereo_solver.hpp"
#include "neurodic/solver/pin_multi_slover.hpp"
#include "neurodic/core/result.hpp"

namespace py = pybind11;

void bind_solver(py::module_& module) {
    py::class_<neurodic::PINSolver>(module, "PINSolver")
        .def(py::init<>())
        .def("solve", [](const neurodic::PINSolver& solver, const neurodic::PINProblem& problem) {
            py::gil_scoped_release release;
            return solver.solve(problem);
        });
    py::class_<neurodic::PINStereoSolver>(module, "PINStereoSolver")
        .def(py::init<>())
        .def("solve", [](const neurodic::PINStereoSolver& solver, const neurodic::PINStereoProblem& problem) {
            py::gil_scoped_release release;
            return solver.solve(problem);
        });
    py::class_<neurodic::PINMultiSolver>(module, "PINMultiSolver")
        .def(py::init<>())
        .def("solve", [](const neurodic::PINMultiSolver& solver, const neurodic::PINMultiProblem& problem) {
            py::gil_scoped_release release;
            return solver.solve(problem);
        });
    py::class_<neurodic::NDeFSolver>(module, "NDeFSolver")
        .def(py::init<>())
        .def("solve", [](const neurodic::NDeFSolver& solver, const neurodic::NDeFProblem& problem) {
            py::gil_scoped_release release;
            return solver.solve(problem);
        });
    py::class_<neurodic::NDeFSurfaceSolver>(module,"NDeFSurfaceSolver").def(py::init<>()).def("solve",[](const neurodic::NDeFSurfaceSolver& s,const neurodic::NDeFSurfaceProblem& p){py::gil_scoped_release r;return s.solve(p);});
}
