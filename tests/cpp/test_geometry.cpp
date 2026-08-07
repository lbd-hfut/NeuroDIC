#include <type_traits>
#include <cassert>

#include "neurodic/calibration/calibration_result.hpp"
#include "neurodic/calibration/camera_model.hpp"
#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
#include "neurodic/calibration/multiview_calibration.hpp"
#endif
#include "neurodic/geometry/ndef_geometry.hpp"
#include "neurodic/geometry/projection.hpp"
#include "neurodic/geometry/stereo_geometry.hpp"
#include "neurodic/geometry/triangulation.hpp"

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

    neurodic::CameraModel left = camera;
    left.intrinsics = torch::eye(3, torch::kFloat64);
    left.translation = torch::zeros({3}, torch::kFloat64);
    left.distortion = torch::zeros({5}, torch::kFloat64);
    neurodic::CameraModel right = left;
    right.translation = torch::tensor({1., 0., 0.}, torch::kFloat64);
    auto point = torch::tensor({{0., 0., 5.}}, torch::kFloat64);
    auto left_uv = neurodic::project_points(point, left.intrinsics, left.rotation, left.translation, left.distortion);
    auto right_uv = neurodic::project_points(point, right.intrinsics, right.rotation, right.translation, right.distortion);
    auto reconstruction = neurodic::triangulate_stereo(left_uv, right_uv, left, right);
    assert(reconstruction.valid.item<bool>());
    assert(torch::allclose(reconstruction.points, point, 1e-9, 1e-9));

    auto differentiable_point = point.clone();
    differentiable_point.requires_grad_(true);
    neurodic::NDeFGeometry ndef({left, right});
    auto projected = ndef.project_reference_surface(differentiable_point);
    assert(projected.uv.sizes() == torch::IntArrayRef({1, 2, 2}));
    assert(torch::allclose(projected.uv.select(1, 0), left_uv, 1e-12, 1e-12));
    projected.uv.sum().backward();
    assert(differentiable_point.grad().defined());
    assert(ndef.visibility(point).all().item<bool>());

#ifdef NEURODIC_HAS_OPENCV_CALIBRATION
    neurodic::calibration::CameraModel sfm_camera;
    sfm_camera.K = Eigen::Matrix3d::Identity();
    sfm_camera.R = Eigen::Matrix3d::Identity();
    sfm_camera.t = Eigen::Vector3d::Zero();
    sfm_camera.image_width = 640;
    sfm_camera.image_height = 480;
    neurodic::calibration::MultiviewCalibrationResult sfm_result;
    sfm_result.cameras = {sfm_camera};
    auto converted_result = neurodic::calibration::to_core_calibration_result(sfm_result);
    assert(converted_result.type == neurodic::CalibrationType::COLMAP);
    assert(converted_result.cameras.size() == 1);
    assert(torch::allclose(converted_result.cameras.front().intrinsics,
                           torch::eye(3, torch::kFloat64)));
#endif
}
