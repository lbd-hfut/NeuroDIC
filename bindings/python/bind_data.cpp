#include <pybind11/pybind11.h>

#include "neurodic/data/roi.hpp"

namespace py = pybind11;

void bind_data(py::module_& module) {
    py::class_<neurodic::ROI>(module, "ROI")
        .def(py::init<double, double, double, double>())
        .def("contains", &neurodic::ROI::contains)
        .def("validate", &neurodic::ROI::validate);
}
