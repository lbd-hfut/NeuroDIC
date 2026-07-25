/**
 * Problem builder.
 *
 * Responsibilities: assemble validated data/calibration/coefficients/init into problems.
 * Inputs: prepared components only.
 * Outputs: PINProblem or NDeFProblem.
 * Ownership: stateless shell for now.
 * Differentiable: NO. Building problems is preprocessing.
 * TODO(NeuroDIC): define typed builder inputs and validation sequence.
 */
#pragma once

#include "neurodic/problem/ndef_problem.hpp"
#include "neurodic/problem/pin_problem.hpp"

namespace neurodic {

class ProblemBuilder {
public:
    PINProblem build_pin_problem(GeometryType geometry_type) const;
    NDeFProblem build_ndef_problem() const;
};

}  // namespace neurodic
