#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_initialization(py::module_& module) {
    module.def("_initialization_bindings_ready", [] { return true; });
}
