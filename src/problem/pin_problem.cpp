#include "neurodic/problem/pin_problem.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

PINProblem::PINProblem(GeometryType geometry_type) : geometry_type_(geometry_type) {}

void PINProblem::validate() const {
    if (geometry_type_ == GeometryType::NDEF_MULTIVIEW) {
        throw ValidationError("PINProblem cannot use NDEF_MULTIVIEW geometry");
    }
}

}  // namespace neurodic
