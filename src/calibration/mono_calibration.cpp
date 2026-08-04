#include "neurodic/calibration/mono_calibration.hpp"

#include "neurodic/core/exceptions.hpp"

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#endif

namespace neurodic {
namespace {
void validate_points(const torch::Tensor& object_points, const torch::Tensor& image_points,
                     int width, int height) {
    if (!object_points.defined() || !image_points.defined() || object_points.device().is_cuda() || image_points.device().is_cuda() ||
        object_points.dim() != 3 || image_points.dim() != 3 || object_points.size(2) != 3 || image_points.size(2) != 2 ||
        object_points.size(0) != image_points.size(0) || object_points.size(1) != image_points.size(1) ||
        object_points.size(0) == 0 || object_points.size(1) < 4 || width <= 0 || height <= 0) {
        throw ValidationError("Calibration points require CPU object[views,points,3], image[views,points,2], and positive image size");
    }
}

#ifdef NEURODIC_HAS_OPENCV
std::vector<std::vector<cv::Point3f>> object_cv(const torch::Tensor& tensor) {
    auto cpu = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto data = cpu.accessor<float, 3>();
    std::vector<std::vector<cv::Point3f>> result(cpu.size(0), std::vector<cv::Point3f>(cpu.size(1)));
    for (int64_t v=0; v<cpu.size(0); ++v) for (int64_t p=0; p<cpu.size(1); ++p) result[v][p] = {data[v][p][0],data[v][p][1],data[v][p][2]};
    return result;
}
std::vector<std::vector<cv::Point2f>> image_cv(const torch::Tensor& tensor) {
    auto cpu = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto data = cpu.accessor<float, 3>();
    std::vector<std::vector<cv::Point2f>> result(cpu.size(0), std::vector<cv::Point2f>(cpu.size(1)));
    for (int64_t v=0; v<cpu.size(0); ++v) for (int64_t p=0; p<cpu.size(1); ++p) result[v][p] = {data[v][p][0],data[v][p][1]};
    return result;
}
torch::Tensor tensor64(const cv::Mat& mat, std::initializer_list<int64_t> shape) {
    cv::Mat converted; mat.convertTo(converted, CV_64F);
    return torch::from_blob(converted.ptr<double>(), shape, torch::kFloat64).clone();
}
CameraModel make_camera(const cv::Mat& K, const cv::Mat& d, int width, int height, double rms, const std::string& label) {
    CameraModel camera;
    camera.intrinsics = tensor64(K, {3,3});
    camera.rotation = torch::eye(3, torch::kFloat64);
    camera.translation = torch::zeros({3}, torch::kFloat64);
    camera.distortion = tensor64(d.reshape(1, 1), {static_cast<int64_t>(d.total())});
    camera.image_width=width; camera.image_height=height; camera.rms_error=rms; camera.label=label;
    return camera;
}
#endif
}  // namespace

CalibrationResult MonoCalibration::run_from_points(const torch::Tensor& object_points,
                                                     const torch::Tensor& image_points,
                                                     int width, int height,
                                                     const MonoCalibrationOptions& options) const {
    validate_points(object_points, image_points, width, height);
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw NotImplementedScientificError("Mono calibration requires NEURODIC_USE_OPENCV=ON with OpenCV development libraries");
#else
    auto objects = object_cv(object_points); auto images = image_cv(image_points);
    cv::Mat K=cv::Mat::eye(3,3,CV_64F), d=cv::Mat::zeros(8,1,CV_64F); std::vector<cv::Mat> rvecs,tvecs;
    int flags=0; if (!options.estimate_tangential_distortion) flags|=cv::CALIB_ZERO_TANGENT_DIST; if (!options.estimate_k3) flags|=cv::CALIB_FIX_K3;
    const double rms=cv::calibrateCamera(objects,images,{width,height},K,d,rvecs,tvecs,flags,
        {cv::TermCriteria::COUNT+cv::TermCriteria::EPS,options.max_iterations,options.epsilon});
    CalibrationResult result; result.type=CalibrationType::MONO; result.cameras={make_camera(K,d,width,height,rms,"mono")}; result.rms_error=rms; result.validate(); return result;
#endif
}

}  // namespace neurodic
