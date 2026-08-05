/** Differentiable multi-view geometry for NDeF photometric optimization. */
#pragma once

#include <vector>

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/geometry/geometry.hpp"
#include "neurodic/geometry/projection.hpp"

namespace neurodic {

class NDeFGeometry : public Geometry {
public:
    explicit NDeFGeometry(std::vector<CameraModel> cameras);
    MultiViewProjectionResult project_reference_surface(const torch::Tensor& surface) const;
    MultiViewProjectionResult project_deformed_surface(const torch::Tensor& surface,
                                                        const torch::Tensor& deformation) const;
    torch::Tensor visibility(const torch::Tensor& surface) const;

private:
    std::vector<CameraModel> cameras_;
};

}  // namespace neurodic
