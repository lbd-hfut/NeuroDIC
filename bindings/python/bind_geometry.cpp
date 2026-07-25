#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_geometry(py::module_& module) {
    module.def("_geometry_bindings_ready", [] { return true; });
}
