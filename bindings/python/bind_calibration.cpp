#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "neurodic/calibration/calibration_result.hpp"
#include "neurodic/calibration/camera_model.hpp"

namespace py = pybind11;

void bind_calibration(py::module_& module) {
    module.def("_calibration_bindings_ready", [] { return true; });
    py::class_<neurodic::CameraModel>(module, "CameraModel")
        .def(py::init<>())
        .def_readwrite("intrinsics", &neurodic::CameraModel::intrinsics)
        .def_readwrite("rotation", &neurodic::CameraModel::rotation)
        .def_readwrite("translation", &neurodic::CameraModel::translation)
        .def_readwrite("distortion", &neurodic::CameraModel::distortion)
        .def_readwrite("image_width", &neurodic::CameraModel::image_width)
        .def_readwrite("image_height", &neurodic::CameraModel::image_height)
        .def_readwrite("rms_error", &neurodic::CameraModel::rms_error)
        .def_readwrite("label", &neurodic::CameraModel::label)
        .def("validate", &neurodic::CameraModel::validate)
        .def("projection_matrix", &neurodic::CameraModel::projection_matrix)
        .def("camera_center", &neurodic::CameraModel::camera_center);
    py::class_<neurodic::CalibrationResult>(module, "CalibrationResult")
        .def(py::init<>())
        .def_readwrite("cameras", &neurodic::CalibrationResult::cameras)
        .def_readwrite("rms_error", &neurodic::CalibrationResult::rms_error)
        .def("validate", &neurodic::CalibrationResult::validate);
}
