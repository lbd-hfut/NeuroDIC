#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_calibration(py::module_& module) {
    module.def("_calibration_bindings_ready", [] { return true; });
}
