#include "neurodic/data/roi.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

ROI::ROI(double x_min, double y_min, double x_max, double y_max)
    : x_min_(x_min), y_min_(y_min), x_max_(x_max), y_max_(y_max) {
    validate();
}

bool ROI::contains(double x, double y) const {
    return x >= x_min_ && x <= x_max_ && y >= y_min_ && y <= y_max_;
}

void ROI::validate() const {
    if (!(x_max_ > x_min_) || !(y_max_ > y_min_)) {
        throw ValidationError("ROI bounds must be positive and ordered");
    }
}

}  // namespace neurodic
