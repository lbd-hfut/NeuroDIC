#include "neurodic/calibration/stereo_calibration.hpp"

#include "neurodic/core/exceptions.hpp"

#ifdef NEURODIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#endif

namespace neurodic {
namespace {
void validate_stereo(const torch::Tensor& object_points, const torch::Tensor& left, const torch::Tensor& right, int width, int height) {
    if (!object_points.defined() || !left.defined() || !right.defined() || object_points.device().is_cuda() || left.device().is_cuda() || right.device().is_cuda() ||
        object_points.dim()!=3 || left.dim()!=3 || right.dim()!=3 || object_points.size(2)!=3 || left.size(2)!=2 || right.size(2)!=2 ||
        object_points.sizes().slice(0,2) != left.sizes().slice(0,2) || left.sizes()!=right.sizes() || object_points.size(0)==0 || object_points.size(1)<4 || width<=0 || height<=0)
        throw ValidationError("Stereo calibration requires matching CPU object[views,points,3], left/right[views,points,2]");
}
#ifdef NEURODIC_HAS_OPENCV
std::vector<std::vector<cv::Point3f>> objects(const torch::Tensor& t) { auto x=t.to(torch::kCPU).to(torch::kFloat32).contiguous(); auto a=x.accessor<float,3>(); std::vector<std::vector<cv::Point3f>> r(x.size(0),std::vector<cv::Point3f>(x.size(1))); for(int64_t i=0;i<x.size(0);++i)for(int64_t j=0;j<x.size(1);++j)r[i][j]={a[i][j][0],a[i][j][1],a[i][j][2]}; return r; }
std::vector<std::vector<cv::Point2f>> images(const torch::Tensor& t) { auto x=t.to(torch::kCPU).to(torch::kFloat32).contiguous(); auto a=x.accessor<float,3>(); std::vector<std::vector<cv::Point2f>> r(x.size(0),std::vector<cv::Point2f>(x.size(1))); for(int64_t i=0;i<x.size(0);++i)for(int64_t j=0;j<x.size(1);++j)r[i][j]={a[i][j][0],a[i][j][1]}; return r; }
torch::Tensor as_tensor(const cv::Mat& input, std::initializer_list<int64_t> shape) { cv::Mat x; input.convertTo(x,CV_64F); return torch::from_blob(x.ptr<double>(),shape,torch::kFloat64).clone(); }
CameraModel camera(const cv::Mat& k, const cv::Mat& d, const cv::Mat& r, const cv::Mat& t, int width, int height, double rms, const char* name) { CameraModel c; c.intrinsics=as_tensor(k,{3,3}); c.rotation=as_tensor(r,{3,3}); c.translation=as_tensor(t.reshape(1,1),{3}); c.distortion=as_tensor(d.reshape(1,1),{static_cast<int64_t>(d.total())}); c.image_width=width;c.image_height=height;c.rms_error=rms;c.label=name; return c; }
#endif
} // namespace

CalibrationResult StereoCalibration::run_from_points(const torch::Tensor& object_points, const torch::Tensor& left_points,
                                                      const torch::Tensor& right_points, int width, int height,
                                                      const StereoCalibrationOptions& options) const {
    validate_stereo(object_points,left_points,right_points,width,height);
#ifndef NEURODIC_HAS_OPENCV
    (void)options;
    throw NotImplementedScientificError("Stereo calibration requires NEURODIC_USE_OPENCV=ON with OpenCV development libraries");
#else
    auto o=objects(object_points); auto l=images(left_points); auto r=images(right_points);
    cv::Mat k1=cv::Mat::eye(3,3,CV_64F),k2=cv::Mat::eye(3,3,CV_64F),d1=cv::Mat::zeros(8,1,CV_64F),d2=cv::Mat::zeros(8,1,CV_64F),R,T,E,F;
    int flags=cv::CALIB_USE_INTRINSIC_GUESS; if(options.fix_intrinsics) flags|=cv::CALIB_FIX_INTRINSIC; if(!options.estimate_tangential_distortion)flags|=cv::CALIB_ZERO_TANGENT_DIST; if(!options.estimate_k3)flags|=cv::CALIB_FIX_K3;
    double rms=cv::stereoCalibrate(o,l,r,k1,d1,k2,d2,{width,height},R,T,E,F,flags,{cv::TermCriteria::COUNT+cv::TermCriteria::EPS,options.max_iterations,options.epsilon});
    CalibrationResult result; result.type=CalibrationType::STEREO; result.cameras={camera(k1,d1,cv::Mat::eye(3,3,CV_64F),cv::Mat::zeros(3,1,CV_64F),width,height,rms,"left"),camera(k2,d2,R,T,width,height,rms,"right")}; result.stereo_rotation=as_tensor(R,{3,3}); result.stereo_translation=as_tensor(T.reshape(1,1),{3}); result.rms_error=rms; result.validate(); return result;
#endif
}

} // namespace neurodic
