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

struct NDeFResult {
    FieldResult surface;
    FieldResult deformation;
    SolverDiagnostics diagnostics;
};

}  // namespace neurodic
