#include <type_traits>
#include <cassert>

#include "neurodic/calibration/calibration_result.hpp"
#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/geometry/ndef_geometry.hpp"
#include "neurodic/geometry/stereo_geometry.hpp"

void test_geometry() {
    static_assert(!std::is_same_v<neurodic::StereoGeometry, neurodic::NDeFGeometry>);
    neurodic::CameraModel camera;
    camera.intrinsics = torch::tensor({{2., 0., 3.}, {0., 4., 5.}, {0., 0., 1.}}, torch::kFloat64);
    camera.rotation = torch::eye(3, torch::kFloat64);
    camera.translation = torch::tensor({1., 2., 3.}, torch::kFloat64);
    camera.distortion = torch::zeros({5}, torch::kFloat64);
    camera.image_width = 640;
    camera.image_height = 480;
    camera.validate();
    assert(camera.projection_matrix().sizes() == torch::IntArrayRef({3, 4}));
    assert(torch::allclose(camera.camera_center(), torch::tensor({-1., -2., -3.}, torch::kFloat64)));
    neurodic::CalibrationResult result;
    result.type = neurodic::CalibrationType::MONO;
    result.cameras = {camera};
    result.validate();
}
