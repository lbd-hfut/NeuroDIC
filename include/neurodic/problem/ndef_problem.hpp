/**
 * NDeF problem.
 *
 * Responsibilities: carry prepared multi-view NDeF data.
 * Inputs: multi-view data, calibration, surface initialization.
 * Outputs: problem consumed by NDeFSolver.
 * Ownership: value shell.
 * Differentiable: PARTIAL. NDeF model-to-loss path is differentiable.
 * A preprocessor may attach the NDeF-DIC surface-dataset visibility and
 * reference projection tensors.  They are fixed observations, never inferred
 * from the deformed image during optimization.
 */
#pragma once

#include <vector>

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/loss/photometric.hpp"
#include "neurodic/model/ndef_internal_model.hpp"
#include "neurodic/problem/problem.hpp"

namespace neurodic {

class NDeFProblem : public DICProblem {
public:
    NDeFProblem() = default;
    NDeFProblem(torch::Tensor reference_surface, torch::Tensor reference_images,
                torch::Tensor deformed_images, torch::Tensor reference_masks,
                torch::Tensor deformed_masks, std::vector<CameraModel> cameras);
    [[nodiscard]] SolverType solver_type() const override { return SolverType::NDEF; }
    void validate() const override;

    // Fixed world-frame surface samples [N,3].  The NDeF network predicts dX at
    // these samples; it does not silently triangulate or reparameterize them.
    torch::Tensor reference_surface;
    // Fixed grayscale observations and masks, all CPU tensors [V,H,W].
    torch::Tensor reference_images;
    torch::Tensor deformed_images;
    torch::Tensor reference_masks;
    torch::Tensor deformed_masks;
    // Optional NDeF-DIC surface-dataset fields: [N,V] and [N,V,2].
    // When absent, reference visibility/UV are derived from camera projection.
    torch::Tensor reference_visibility;
    torch::Tensor reference_projected_uv;
    torch::Tensor visible_counts;
    std::vector<CameraModel> cameras;
    NDeFModelOptions model_options;
    // NDeF-DIC train_deformation.py semantics: each epoch has ceil(N / batch)
    // steps and every step samples points with replacement using randint.
    int training_epochs{100};
    int batch_size{0};  // 0 selects automatic batch sizing.
    int auto_batch_start{1024};
    int auto_batch_max{0};  // 0 means N.
    double memory_fraction{0.80};
    int max_steps_per_epoch{0};  // 0 means uncapped.
    int prediction_batch_size{262144};
    int64_t random_seed{23};
    // Legacy aliases retained for source compatibility. When explicitly set,
    // photometric_iterations overrides the epoch-derived total step count.
    int photometric_iterations{0};
    int photometric_sample_count{0};
    int bspline_degree{5};
    double photometric_learning_rate{5e-4};
    double weight_decay{0.0};
    double smoothness_weight{0.0};
    int patch_radius{2};
    double min_valid_patch_ratio{1.0};
    double invalid_patch_penalty{0.05};
    double sfm_to_world_scale{1.0};
    PhotometricLossType photometric_loss{PhotometricLossType::ZNSSD};
    bool evaluation_enabled{false};
    int evaluation_sample_count{1024};
    int64_t evaluation_seed{104729};
    torch::Device device{torch::kCPU};
};

}  // namespace neurodic
