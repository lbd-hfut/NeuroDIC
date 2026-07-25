/**
 * PIN problem.
 *
 * Responsibilities: serve both planar 2D and stereo PIN-DIC.
 * Inputs: prepared PIN data/calibration/initialization.
 * Outputs: problem consumed by PINSolver.
 * Ownership: value shell.
 * Differentiable: PARTIAL. Prepared observations are fixed; training path is differentiable.
 * TODO(NeuroDIC): add datasets, coefficients, and initialization members.
 */
#pragma once

#include "neurodic/core/types.hpp"
#include "neurodic/problem/problem.hpp"

namespace neurodic {

class PINProblem : public DICProblem {
public:
    explicit PINProblem(GeometryType geometry_type = GeometryType::PLANAR_2D);
    [[nodiscard]] SolverType solver_type() const override { return SolverType::PIN; }
    [[nodiscard]] GeometryType geometry_type() const noexcept { return geometry_type_; }
    void validate() const override;

private:
    GeometryType geometry_type_;
};

}  // namespace neurodic
