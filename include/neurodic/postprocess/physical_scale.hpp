/**
 * Physical scaling.
 *
 * Responsibilities: convert pixel/image-space fields into physical units.
 * Inputs: result tensors and calibration scale.
 * Outputs: scaled tensors.
 * Ownership: tensor references only.
 * Differentiable: NO for exported postprocessing.
 * TODO(NeuroDIC): define scale provenance and uncertainty reporting.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic { torch::Tensor apply_physical_scale(const torch::Tensor& values, double scale); }
