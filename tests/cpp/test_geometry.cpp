#include <type_traits>

#include "neurodic/geometry/ndef_geometry.hpp"
#include "neurodic/geometry/stereo_geometry.hpp"

void test_geometry() {
    static_assert(!std::is_same_v<neurodic::StereoGeometry, neurodic::NDeFGeometry>);
}
