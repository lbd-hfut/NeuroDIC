/**
 * Strain postprocessing.
 *
 * Responsibilities: derive strain from solved displacement fields.
 * Inputs: displacement field tensors.
 * Outputs: Green--Lagrange strain tensors.  The packed component order is
 * `[E_xx, E_yy, E_xy]` in 2D and `[E_xx, E_yy, E_zz, E_xy, E_yz, E_xz]` in 3D.
 * Ownership: tensor references only.
 * Differentiable: NO for post-solve analysis.
 * TODO(NeuroDIC): validate strain definitions before implementation.
 */
#pragma once

#include <functional>

#include <torch/torch.h>

namespace neurodic {

// `coordinate_scale` and `displacement_scale` convert the network's input and
// output coordinate systems to physical units component by component.
torch::Tensor compute_neural_strain_2d(
    const std::function<torch::Tensor(const torch::Tensor&)>& deformation,
    const torch::Tensor& coordinates, const torch::Tensor& coordinate_scale = torch::Tensor(),
    const torch::Tensor& displacement_scale = torch::Tensor());

torch::Tensor compute_neural_strain_3d(
    const std::function<torch::Tensor(const torch::Tensor&)>& deformation,
    const torch::Tensor& coordinates, const torch::Tensor& coordinate_scale = torch::Tensor(),
    const torch::Tensor& displacement_scale = torch::Tensor());

// Estimate a 3D displacement gradient at scattered reconstructed points with
// weighted local least squares, then convert it to Green--Lagrange strain.
// Invalid/undersampled rows are returned as NaN.
torch::Tensor compute_traditional_strain_3d(
    const torch::Tensor& coordinates, const torch::Tensor& displacement,
    const torch::Tensor& valid = torch::Tensor(), int64_t neighbors = 12,
    const torch::Tensor& coordinate_scale = torch::Tensor(),
    const torch::Tensor& displacement_scale = torch::Tensor());

}  // namespace neurodic
