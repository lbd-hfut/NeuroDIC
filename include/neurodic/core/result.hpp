/**
 * Solver results and diagnostics.
 *
 * Responsibilities: carry tensors and metadata produced by solvers.
 * Inputs: solver-owned field tensors, scalar losses, status values.
 * Outputs: result structs for C++ and Python binding layers.
 * Ownership: torch::Tensor uses PyTorch reference-counted tensor ownership.
 * Differentiable: PARTIAL. Result tensors may carry autograd history during
 * internal optimization, but exported public results should normally be detached
 * explicitly by the caller once optimization is complete.
 * TODO(NeuroDIC): finalize physical units, coordinate conventions, and metadata.
 */
#pragma once

#include <map>
#include <string>
#include <torch/torch.h>

#include "neurodic/core/types.hpp"

namespace neurodic {

struct SolverDiagnostics {
    SolverStatus status = SolverStatus::NOT_STARTED;
    int iterations = 0;
    double final_loss = 0.0;
    std::map<std::string, double> metrics;
};

struct FieldResult {
    torch::Tensor coordinates;
    torch::Tensor values;
};

struct PINResult {
    FieldResult displacement;
    SolverDiagnostics diagnostics;
};

struct PINStereoResult {
    PINResult reference_disparity;  // L0 -> R0
    PINResult left_temporal;        // L0 -> L1
    PINResult deformed_disparity;   // L0 -> R1
    torch::Tensor left_reference_coordinates;   // CPU float64 [N,2]
    torch::Tensor right_reference_coordinates;  // CPU float64 [N,2]
    torch::Tensor left_current_coordinates;     // CPU float64 [N,2]
    torch::Tensor right_current_coordinates;    // CPU float64 [N,2]
    torch::Tensor reference_points;              // CPU float64 [N,3]
    torch::Tensor current_points;                // CPU float64 [N,3]
    torch::Tensor displacement_3d;               // CPU float64 [N,3]
    torch::Tensor valid;                         // CPU bool [N]
    torch::Tensor reference_reprojection_error;  // CPU float64 [N]
    torch::Tensor current_reprojection_error;    // CPU float64 [N]
};

struct NDeFResult {
    FieldResult surface;      // coordinates=reference surface; values=current surface
    FieldResult deformation;
    torch::Tensor reference_uv;       // CPU float32 [N,V,2]
    torch::Tensor current_uv;         // CPU float32 [N,V,2]
    torch::Tensor reference_depth;    // CPU float32 [N,V]
    torch::Tensor current_depth;      // CPU float32 [N,V]
    torch::Tensor valid;              // CPU bool [N,V], depth/bounds/ROI at both states
    // The NDeF field is optimized in SfM coordinates; preserve those arrays
    // alongside the world-scale public FieldResult export.
    torch::Tensor reference_surface_sfm;
    torch::Tensor current_surface_sfm;
    torch::Tensor deformation_sfm;
    double sfm_to_world_scale{1.0};
    SolverDiagnostics diagnostics;
};
struct NDeFSurfaceResult {
    torch::Tensor sparse_prediction, query_depth, query_uv, query_cameras;
    // Dense stage exports are sampled ROI centres after topology filtering.
    torch::Tensor dense_uv, dense_cameras, dense_targets, dense_depth, dense_world, dense_history;
    // Final network inference on every exported ROI query-grid point.  These
    // fields are the dense charts consumed by visibility/depth fusion.
    torch::Tensor dense_field_uv, dense_field_cameras, dense_field_depth, dense_field_world;
    double depth_mean{0.0}, depth_std{1.0}; SolverDiagnostics diagnostics;
};

}  // namespace neurodic
