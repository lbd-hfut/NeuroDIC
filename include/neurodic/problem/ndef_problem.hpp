/**
 * NDeF problem.
 *
 * Responsibilities: carry prepared multi-view NDeF data.
 * Inputs: multi-view data, calibration, surface initialization.
 * Outputs: problem consumed by NDeFSolver.
 * Ownership: value shell.
 * Differentiable: PARTIAL. NDeF model-to-loss path is differentiable.
 * TODO(NeuroDIC): define reference surface and visibility inputs.
 */
#pragma once

#include "neurodic/problem/problem.hpp"

namespace neurodic {

class NDeFProblem : public DICProblem {
public:
    [[nodiscard]] SolverType solver_type() const override { return SolverType::NDEF; }
    void validate() const override;
};

}  // namespace neurodic
