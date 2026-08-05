/**
 * pybind11 module entry for neurodic._neurodic.
 *
 * Responsibilities: assemble selected C++ bindings.
 * Inputs: Python import machinery.
 * Outputs: extension module.
 * Differentiability: PARTIAL. Bound tensor APIs must preserve LibTorch autograd.
 * TODO(NeuroDIC): expose only validated interfaces, not every internal class.
 */
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core(py::module_& module);
void bind_data(py::module_& module);
void bind_interpolation(py::module_& module);
void bind_initialization(py::module_& module);
void bind_calibration(py::module_& module);
void bind_traditional_calibration(py::module_& module);
void bind_problem(py::module_& module);
void bind_geometry(py::module_& module);
void bind_solver(py::module_& module);
void bind_result(py::module_& module);

PYBIND11_MODULE(_neurodic, module) {
    module.doc() = "NeuroDIC C++/LibTorch core bindings.";
    bind_core(module);
    bind_data(module);
    bind_interpolation(module);
    bind_initialization(module);
    bind_calibration(module);
    bind_traditional_calibration(module);
    bind_problem(module);
    bind_geometry(module);
    bind_solver(module);
    bind_result(module);
}
