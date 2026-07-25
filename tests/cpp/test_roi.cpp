#include <cassert>

#include "neurodic/data/roi.hpp"

void test_roi() {
    const neurodic::ROI roi(0.0, 0.0, 10.0, 5.0);
    assert(roi.contains(1.0, 1.0));
    assert(!roi.contains(11.0, 1.0));
}
