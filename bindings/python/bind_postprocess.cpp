#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "neurodic/postprocess/strain.hpp"
#include "neurodic/postprocess/filtering.hpp"
#include "neurodic/postprocess/surface_field.hpp"
#include "neurodic/postprocess/surface_mesh.hpp"

namespace py = pybind11;

void bind_postprocess(py::module_& module) {
    py::class_<neurodic::SurfaceFaceField>(module, "SurfaceFaceField")
        .def_readonly("face_centers", &neurodic::SurfaceFaceField::face_centers)
        .def_readonly("face_values", &neurodic::SurfaceFaceField::face_values)
        .def_readonly("valid_faces", &neurodic::SurfaceFaceField::valid_faces);
    module.def("prepare_surface_face_field", &neurodic::prepare_surface_face_field,
               py::arg("points"), py::arg("faces"), py::arg("point_values"),
               "Validate triangular faces and reduce one or more vertex fields to face fields.");
    py::class_<neurodic::SurfaceMeshOptions>(module, "SurfaceMeshOptions")
        .def(py::init<>()).def_readwrite("k_neighbors", &neurodic::SurfaceMeshOptions::k_neighbors)
        .def_readwrite("max_edge_length", &neurodic::SurfaceMeshOptions::max_edge_length)
        .def_readwrite("min_triangle_quality", &neurodic::SurfaceMeshOptions::min_triangle_quality);
    py::class_<neurodic::SurfaceMesh>(module, "SurfaceMesh")
        .def_readonly("vertices", &neurodic::SurfaceMesh::vertices).def_readonly("faces", &neurodic::SurfaceMesh::faces)
        .def_readonly("normals", &neurodic::SurfaceMesh::normals).def_readonly("quality", &neurodic::SurfaceMesh::quality)
        .def_readonly("median_spacing", &neurodic::SurfaceMesh::median_spacing).def_readonly("max_edge_length", &neurodic::SurfaceMesh::max_edge_length);
    module.def("triangulate_pin_multi_surface", &neurodic::triangulate_pin_multi_surface,
               py::arg("points"), py::arg("options") = neurodic::SurfaceMeshOptions{},
               "Triangulate a cleaned multi-view point surface in local tangent planes.");
    py::class_<neurodic::SurfaceCleanupResult>(module, "SurfaceCleanupResult")
        .def_readonly("inlier_mask", &neurodic::SurfaceCleanupResult::inlier_mask)
        .def_readonly("neighbor_distance", &neurodic::SurfaceCleanupResult::neighbor_distance)
        .def_readonly("plane_residual", &neurodic::SurfaceCleanupResult::plane_residual)
        .def_readonly("neighbor_distance_median", &neurodic::SurfaceCleanupResult::neighbor_distance_median)
        .def_readonly("neighbor_distance_mad", &neurodic::SurfaceCleanupResult::neighbor_distance_mad)
        .def_readonly("neighbor_distance_threshold", &neurodic::SurfaceCleanupResult::neighbor_distance_threshold)
        .def_readonly("plane_residual_median", &neurodic::SurfaceCleanupResult::plane_residual_median)
        .def_readonly("plane_residual_mad", &neurodic::SurfaceCleanupResult::plane_residual_mad)
        .def_readonly("plane_residual_threshold", &neurodic::SurfaceCleanupResult::plane_residual_threshold);
    py::class_<neurodic::MeshCleanupResult>(module, "MeshCleanupResult")
        .def_readonly("face_mask", &neurodic::MeshCleanupResult::face_mask)
        .def_readonly("face_quality", &neurodic::MeshCleanupResult::face_quality)
        .def_readonly("mean_edge_length", &neurodic::MeshCleanupResult::mean_edge_length)
        .def_readonly("overlap_distance", &neurodic::MeshCleanupResult::overlap_distance);
    module.def("clean_pin_multi_mesh", &neurodic::clean_pin_multi_mesh,
               py::arg("vertices"), py::arg("faces"), py::arg("quality") = torch::Tensor(),
               py::arg("overlap_distance") = 0.0, py::arg("min_triangle_quality") = 0.20);
    py::class_<neurodic::LocalDisplacementConsistencyResult>(module, "LocalDisplacementConsistencyResult")
        .def_readonly("predicted_displacement", &neurodic::LocalDisplacementConsistencyResult::predicted_displacement)
        .def_readonly("residual", &neurodic::LocalDisplacementConsistencyResult::residual)
        .def_readonly("inlier_mask", &neurodic::LocalDisplacementConsistencyResult::inlier_mask)
        .def_readonly("residual_median", &neurodic::LocalDisplacementConsistencyResult::residual_median)
        .def_readonly("residual_mad", &neurodic::LocalDisplacementConsistencyResult::residual_mad)
        .def_readonly("residual_threshold", &neurodic::LocalDisplacementConsistencyResult::residual_threshold);
    module.def("compute_local_displacement_consistency", &neurodic::compute_local_displacement_consistency,
               py::arg("coordinates"), py::arg("displacement"), py::arg("valid") = torch::Tensor(),
               py::arg("k_neighbors") = 16, py::arg("mad_factor") = 5.0);
    module.def("clean_pin_multi_surface", &neurodic::clean_pin_multi_surface,
               py::arg("points"), py::arg("k_neighbors") = 16, py::arg("mad_factor") = 5.0,
               "Clean voxel-fused multi-view points using k-NN density and local-plane residuals.");
    module.def("compute_traditional_strain_3d", [](py::object coordinates, py::object displacement,
                                                     py::object valid, int64_t neighbors,
                                                     py::object coordinate_scale, py::object displacement_scale) {
        auto optional_tensor = [](const py::object& value) {
            return value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
        };
        return neurodic::compute_traditional_strain_3d(
            coordinates.cast<torch::Tensor>(), displacement.cast<torch::Tensor>(), optional_tensor(valid), neighbors,
            optional_tensor(coordinate_scale), optional_tensor(displacement_scale));
    },
               py::arg("coordinates"), py::arg("displacement"), py::arg("valid") = py::none(),
               py::arg("neighbors") = 12, py::arg("coordinate_scale") = py::none(),
               py::arg("displacement_scale") = py::none(),
               "Estimate packed Green-Lagrange strain [Exx,Eyy,Ezz,Exy,Eyz,Exz] from a scattered 3D field.");
}
