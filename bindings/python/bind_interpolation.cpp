#include <pybind11/pybind11.h>

#include "neurodic/interpolation/bspline.hpp"
#include "neurodic/interpolation/torch_bspline.hpp"

namespace py = pybind11;

void bind_interpolation(py::module_& module) {
    module.def("is_supported_bspline_degree", &neurodic::is_supported_bspline_degree);
    module.def("validate_bspline_degree", &neurodic::validate_bspline_degree);
    py::class_<neurodic::TorchBSplineInterpolator>(module, "TorchBSplineInterpolator")
        .def(py::init<int>())
        .def_property_readonly("degree", &neurodic::TorchBSplineInterpolator::degree);
}
