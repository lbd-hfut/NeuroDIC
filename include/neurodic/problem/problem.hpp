/**
 * Base DIC problem.
 *
 * Responsibilities: carry prepared inputs consumed by solvers.
 * Inputs: datasets, ROI, calibration, coefficients, initialization.
 * Outputs: solver-ready problem object.
 * Ownership: value object with reference-counted tensors.
 * Differentiable: PARTIAL. Coefficients may be fixed tensors; solver-created
 * model outputs and coordinates become differentiable downstream.
 * TODO(NeuroDIC): finalize common problem members after the first PIN-DIC path.
 */
#pragma once

#include "neurodic/core/types.hpp"

namespace neurodic {

class DICProblem {
public:
    virtual ~DICProblem() = default;
    [[nodiscard]] virtual SolverType solver_type() const = 0;
    virtual void validate() const = 0;
};

}  // namespace neurodic
