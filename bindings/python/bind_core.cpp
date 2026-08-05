#include <pybind11/pybind11.h>

#include "neurodic/core/types.hpp"
#include "neurodic/loss/photometric.hpp"

namespace py = pybind11;

void bind_core(py::module_& module) {
    py::enum_<neurodic::SolverType>(module, "SolverType")
        .value("PIN", neurodic::SolverType::PIN)
        .value("NDEF", neurodic::SolverType::NDEF);
    py::enum_<neurodic::GeometryType>(module, "GeometryType")
        .value("PLANAR_2D", neurodic::GeometryType::PLANAR_2D)
        .value("STEREO", neurodic::GeometryType::STEREO)
        .value("NDEF_MULTIVIEW", neurodic::GeometryType::NDEF_MULTIVIEW);
    py::enum_<neurodic::PhotometricLossType>(module, "PhotometricLossType")
        .value("SSD", neurodic::PhotometricLossType::SSD)
        .value("ZNSSD", neurodic::PhotometricLossType::ZNSSD);
}
