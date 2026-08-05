#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "neurodic/geometry/projection.hpp"
#include "neurodic/geometry/triangulation.hpp"

namespace py = pybind11;

void bind_geometry(py::module_& module) {
    module.def("project_points", [](const torch::Tensor& points, const torch::Tensor& intrinsics,
                                    const torch::Tensor& rotation, const torch::Tensor& translation,
                                    const torch::Tensor& distortion) {
        return neurodic::project_points(points, intrinsics, rotation, translation, distortion);
    }, py::arg("points"), py::arg("intrinsics"), py::arg("rotation"), py::arg("translation"), py::arg("distortion"));
    module.def("project_points", [](const torch::Tensor& points, const torch::Tensor& intrinsics,
                                    const torch::Tensor& rotation, const torch::Tensor& translation) {
        return neurodic::project_points(points, intrinsics, rotation, translation);
    }, py::arg("points"), py::arg("intrinsics"), py::arg("rotation"), py::arg("translation"));
    module.def("project_points_multi_view", [](const torch::Tensor& points, const torch::Tensor& intrinsics,
                                               const torch::Tensor& rotations, const torch::Tensor& translations,
                                               const torch::Tensor& distortions) {
        auto result = neurodic::project_points_multi_view(points, intrinsics, rotations, translations, distortions);
        return py::make_tuple(result.uv, result.depth);
    }, py::arg("points"), py::arg("intrinsics"), py::arg("rotations"), py::arg("translations"),
       py::arg("distortions"));
    module.def("project_points_multi_view", [](const torch::Tensor& points, const torch::Tensor& intrinsics,
                                               const torch::Tensor& rotations, const torch::Tensor& translations) {
        auto result = neurodic::project_points_multi_view(points, intrinsics, rotations, translations);
        return py::make_tuple(result.uv, result.depth);
    }, py::arg("points"), py::arg("intrinsics"), py::arg("rotations"), py::arg("translations"));
    module.def("triangulate_stereo", [](const torch::Tensor& left, const torch::Tensor& right,
                                         const neurodic::CameraModel& left_camera,
                                         const neurodic::CameraModel& right_camera) {
        auto result = neurodic::triangulate_stereo(left, right, left_camera, right_camera);
        py::dict output;
        output["points"] = result.points;
        output["valid"] = result.valid;
        output["mean_reprojection_error"] = result.mean_reprojection_error;
        output["max_reprojection_error"] = result.max_reprojection_error;
        output["observations_used"] = result.observations_used;
        return output;
    }, py::arg("left_coordinates"), py::arg("right_coordinates"), py::arg("left_camera"), py::arg("right_camera"));
}
