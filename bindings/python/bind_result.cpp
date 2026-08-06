#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "neurodic/core/result.hpp"

namespace py = pybind11;

void bind_result(py::module_& module) {
    py::class_<neurodic::SolverDiagnostics>(module, "SolverDiagnostics")
        .def_readonly("iterations", &neurodic::SolverDiagnostics::iterations)
        .def_readonly("final_loss", &neurodic::SolverDiagnostics::final_loss)
        .def_readonly("metrics", &neurodic::SolverDiagnostics::metrics);
    py::class_<neurodic::FieldResult>(module, "FieldResult")
        .def_readonly("coordinates", &neurodic::FieldResult::coordinates)
        .def_readonly("values", &neurodic::FieldResult::values);
    py::class_<neurodic::PINResult>(module, "PINResult")
        .def_readonly("displacement", &neurodic::PINResult::displacement)
        .def_readonly("diagnostics", &neurodic::PINResult::diagnostics);
    py::class_<neurodic::PINStereoResult>(module, "PINStereoResult")
        .def_readonly("reference_disparity", &neurodic::PINStereoResult::reference_disparity)
        .def_readonly("left_temporal", &neurodic::PINStereoResult::left_temporal)
        .def_readonly("deformed_disparity", &neurodic::PINStereoResult::deformed_disparity)
        .def_readonly("left_reference_coordinates", &neurodic::PINStereoResult::left_reference_coordinates)
        .def_readonly("right_reference_coordinates", &neurodic::PINStereoResult::right_reference_coordinates)
        .def_readonly("left_current_coordinates", &neurodic::PINStereoResult::left_current_coordinates)
        .def_readonly("right_current_coordinates", &neurodic::PINStereoResult::right_current_coordinates)
        .def_readonly("reference_points", &neurodic::PINStereoResult::reference_points)
        .def_readonly("current_points", &neurodic::PINStereoResult::current_points)
        .def_readonly("displacement_3d", &neurodic::PINStereoResult::displacement_3d)
        .def_readonly("valid", &neurodic::PINStereoResult::valid)
        .def_readonly("reference_reprojection_error", &neurodic::PINStereoResult::reference_reprojection_error)
        .def_readonly("current_reprojection_error", &neurodic::PINStereoResult::current_reprojection_error);
    py::class_<neurodic::NDeFResult>(module, "NDeFResult")
        .def_readonly("surface", &neurodic::NDeFResult::surface)
        .def_readonly("deformation", &neurodic::NDeFResult::deformation)
        .def_readonly("reference_uv", &neurodic::NDeFResult::reference_uv)
        .def_readonly("current_uv", &neurodic::NDeFResult::current_uv)
        .def_readonly("reference_depth", &neurodic::NDeFResult::reference_depth)
        .def_readonly("current_depth", &neurodic::NDeFResult::current_depth)
        .def_readonly("valid", &neurodic::NDeFResult::valid)
        .def_readonly("reference_surface_sfm", &neurodic::NDeFResult::reference_surface_sfm)
        .def_readonly("current_surface_sfm", &neurodic::NDeFResult::current_surface_sfm)
        .def_readonly("deformation_sfm", &neurodic::NDeFResult::deformation_sfm)
        .def_readonly("training_history", &neurodic::NDeFResult::training_history)
        .def_readonly("training_sample_counts", &neurodic::NDeFResult::training_sample_counts)
        .def_readonly("coordinate_center", &neurodic::NDeFResult::coordinate_center)
        .def_readonly("coordinate_scale", &neurodic::NDeFResult::coordinate_scale)
        .def_readonly("training_batch_size", &neurodic::NDeFResult::training_batch_size)
        .def_readonly("steps_per_epoch", &neurodic::NDeFResult::steps_per_epoch)
        .def_readonly("completed_epochs", &neurodic::NDeFResult::completed_epochs)
        .def_readonly("random_seed", &neurodic::NDeFResult::random_seed)
        .def_readonly("output_scale", &neurodic::NDeFResult::output_scale)
        .def_readonly("model_parameter_names", &neurodic::NDeFResult::model_parameter_names)
        .def_readonly("model_state", &neurodic::NDeFResult::model_state)
        .def_readonly("last_model_state", &neurodic::NDeFResult::last_model_state)
        .def_readonly("sfm_to_world_scale", &neurodic::NDeFResult::sfm_to_world_scale)
        .def_readonly("diagnostics", &neurodic::NDeFResult::diagnostics);
    py::class_<neurodic::NDeFSurfaceResult>(module,"NDeFSurfaceResult").def_readonly("sparse_prediction",&neurodic::NDeFSurfaceResult::sparse_prediction).def_readonly("query_depth",&neurodic::NDeFSurfaceResult::query_depth).def_readonly("query_uv",&neurodic::NDeFSurfaceResult::query_uv).def_readonly("query_cameras",&neurodic::NDeFSurfaceResult::query_cameras).def_readonly("dense_uv",&neurodic::NDeFSurfaceResult::dense_uv).def_readonly("dense_cameras",&neurodic::NDeFSurfaceResult::dense_cameras).def_readonly("dense_targets",&neurodic::NDeFSurfaceResult::dense_targets).def_readonly("dense_depth",&neurodic::NDeFSurfaceResult::dense_depth).def_readonly("dense_world",&neurodic::NDeFSurfaceResult::dense_world).def_readonly("dense_history",&neurodic::NDeFSurfaceResult::dense_history).def_readonly("dense_field_uv",&neurodic::NDeFSurfaceResult::dense_field_uv).def_readonly("dense_field_cameras",&neurodic::NDeFSurfaceResult::dense_field_cameras).def_readonly("dense_field_depth",&neurodic::NDeFSurfaceResult::dense_field_depth).def_readonly("dense_field_world",&neurodic::NDeFSurfaceResult::dense_field_world).def_readonly("depth_mean",&neurodic::NDeFSurfaceResult::depth_mean).def_readonly("depth_std",&neurodic::NDeFSurfaceResult::depth_std).def_readonly("diagnostics",&neurodic::NDeFSurfaceResult::diagnostics);
}
