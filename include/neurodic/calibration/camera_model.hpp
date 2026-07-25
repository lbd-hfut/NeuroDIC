/**
 * Camera model.
 *
 * Responsibilities: store intrinsic/extrinsic tensors for geometry modules.
 * Inputs: calibration outputs from mono/stereo/COLMAP adapters.
 * Outputs: camera parameter container.
 * Ownership: tensors use PyTorch ownership.
 * Differentiable: PARTIAL. Calibration parameters are fixed by default; geometry
 * may use them in differentiable tensor computations.
 * TODO(NeuroDIC): finalize distortion model and coordinate conventions.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct CameraModel {
    torch::Tensor intrinsics;
    torch::Tensor extrinsics;
    torch::Tensor distortion;
    void validate() const {}
};

}  // namespace neurodic
