/**
 * PIN problem.
 *
 * Responsibilities: carry the validated planar 2D PIN-DIC training inputs.
 * Inputs: prepared PIN data/calibration/initialization.
 * Outputs: problem consumed by PINSolver.
 * Ownership: value shell.
 * Differentiable: PARTIAL. Prepared observations are fixed; training path is differentiable.
 * Stereo assembly remains a later extension of this common value object.
 */
#pragma once

#include <torch/torch.h>

#include "neurodic/core/types.hpp"
#include "neurodic/data/image_precompute_context.hpp"
#include "neurodic/initialization/seed_set.hpp"
#include "neurodic/loss/photometric.hpp"
#include "neurodic/model/mlp.hpp"
#include "neurodic/problem/problem.hpp"

namespace neurodic {

class PINProblem : public DICProblem {
private:
    GeometryType geometry_type_;

public:
    explicit PINProblem(GeometryType geometry_type = GeometryType::PLANAR_2D);
    PINProblem(torch::Tensor reference_image, torch::Tensor deformed_image, torch::Tensor roi_mask,
               SeedSet seeds, PINModelOptions model_options = {}, ImagePrecomputeOptions precompute_options = {});
    [[nodiscard]] SolverType solver_type() const override { return SolverType::PIN; }
    [[nodiscard]] GeometryType geometry_type() const noexcept { return geometry_type_; }
    void validate() const override;

    torch::Tensor reference_image;
    torch::Tensor deformed_image;
    torch::Tensor roi_mask;
    ImagePrecomputeContext precompute;
    SeedSet seeds;
    PINModelOptions model_options;
    int seed_iterations{500};
    int photometric_iterations{1500};
    int photometric_sample_count{16384};
    bool photometric_sampling_enabled{true};
    int znssd_kernel_size{7};
    double seed_learning_rate{1e-3};
    double photometric_learning_rate{5e-4};
    PhotometricLossType photometric_loss{PhotometricLossType::ZNSSD};
    torch::Device device{torch::kCPU};
};

}  // namespace neurodic
