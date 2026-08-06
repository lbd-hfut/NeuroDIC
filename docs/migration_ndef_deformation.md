# NDeF deformation-stage migration audit

This note records the Python source semantics used for the C++ migration. It is
deliberately separate from dense surface reconstruction: the deformation trainer
consumes `result/surface/deformation_surface_dataset.npz` unchanged.

## Python entry and data flow

`NDeF-DIC/run.py:339-377` runs sparse patch-DIC once per current frame, then
`run.py:380-443` assembles `DeformationTrainingConfig` and calls
`run_deformation_training`. The dataset loader at
`ndef_dic/deformation/deformation_dataset.py:41-71` reads `points`, `normals`,
`visibility_mask`, `projected_uv`, `visible_counts`, and `cam_names`; it loads
camera `K/dist/R/t` and normalized reference/current grayscale images separately.

The actual deformation training set is not split by camera, patch, shard, or
spatial region. `deformation_dataset.py:81-89` draws point indices from the whole
surface with `torch.randint(0,N,[batch])`: probability `1/N`, with replacement,
and a fresh draw every step. `normals`, `source_camera`, `projected_depth`, and
`depth_abs_error` do not enter the Python deformation loss. They remain useful
diagnostic/source fields and are preserved by the migrated exporter.

Sparse scale precalculation is a different sampler. In
`precalculation/patch_dic_precalc.py:83-176`, every source camera gets up to 300
ROI seeds. `_sample_roi_points` (`:274-333`) removes a patch-radius image margin,
uses a ceil(sqrt(count)) spatial grid, randomly tries up to 20 candidates per
cell, requires local standard deviation >= 0.02, then randomly fills missing
seeds without replacement. Camera-ring neighbours are defined at `:484-495`.
Cross-camera search centres use median projected-UV offsets from the surface
visibility fields (`:498-506`). Thus this camera-grouped seed sampling is not the
uniform deformation-network point sampling.

Patch NCC, temporal/cross thresholds, common-camera DLT, and reprojection
filtering are at `patch_dic_precalc.py:98-180` and `:336-427`. Magnitudes are
filtered by `abs(m-median) <= threshold * 1.4826 * MAD` (`:460-469`). Mean,
median, p75, p90, and max of the inliers are produced at `:472-481`; configured
training uses the mean as the network output multiplier. These values have the
same coordinate unit as the camera/surface inputs.

## Model, objective, and training semantics

Coordinate normalization is component-wise bounding-box normalization:
`center=(min+max)/2`, `scale=max((max-min)/2,1e-8)`, and
`x_norm=(x-center)/scale` (`deformation_dataset.py:67-71`,
`deformation_field.py:98-106`). The optional Fourier encoder uses bands
`pi*2**arange(F)` and emits the original xyz followed by coordinate-major
`[sin bands, cos bands]` flattened from `[N,3,F]`
(`deformation_field.py:35-46`). The configured model is five 32-wide Tanh hidden
layers and a 3-vector output multiplied by displacement scale (`:69-106`). Hidden
layers use Xavier-uniform with Tanh gain, zero biases; the last layer uses normal
std 1e-5 and zero bias (`:90-96`).

For every sampled point-camera visibility pair, the loss projects `X+u(X)`,
forms matching integer-offset patches, and samples both images bilinearly
(`deformation_loss.py:29-99`, `:141-225`). A reference pair is supervised when
its in-image patch fraction passes the configured threshold. An invalid current
centre/depth/patch receives the fixed penalty 0.05. Valid patches use pixel MSE
or ZNSSD. Pair losses are weighted by `1/visible_counts[point]`, then reduced by
weighted sum divided by weight sum. Visibility is fixed reference-surface
visibility; normals, source camera and stored projected depth do not affect it.

Smoothness (`deformation_loss.py:122-138`) is the mean squared Jacobian of output
displacement with respect to normalized coordinates. Python forces its effective
weight to zero when Fourier encoding is disabled (`train_deformation.py:87-89`),
which is the CylinderDIC configuration. There are no other active losses.

`train_deformation.py:90-149` defines `steps_per_epoch=ceil(N/batch)`, optionally
capped, then performs that many independent with-replacement batches per epoch.
Automatic CPU batch is `min(auto_batch_start,N)`; CUDA probes doubling plus
bisection against a fraction of initially free memory (`:450-524`). Optimization
is AdamW with the configured constant learning rate and weight decay; there is no
scheduler or early stopping. The minimum step loss is retained as the best state.
Both last and best checkpoints are written only after all epochs (`:151-185`).
Inference evaluates every input surface point in batches, returns `u`, and forms
`X_current=X_reference+u`; optional world scale multiplies points and displacement
only at export (`:308-346`).

## C++ correspondence and resolved differences

| Concern | Previous C++ behaviour | Migrated implementation |
|---|---|---|
| Training surface | Could use a theoretical NPY and fixed linspace subset | Existing NPZ configured at `config/ndef_multiview.yaml:14`; all point observations assembled in `python/neurodic/api/ndef_dic.py:253-303` |
| Training sampling | One fixed linspace selection | Fresh `torch::randint` per step plus per-point counts at `src/solver/ndef_solver.cpp:240-260` |
| Epoch/batch | Flat `photometric_iterations` | Python epoch, fixed/auto batch, cap, seed, and inference batch options at `include/neurodic/problem/ndef_problem.hpp:50-63` and `src/solver/ndef_solver.cpp:194-235` |
| Photometric loss | B-spline interpolation and ROI-mask intersection | Python-equivalent bilinear sampling, image-bound masks, invalid penalty, visible-count weighting, MSE/ZNSSD at `src/solver/ndef_solver.cpp:18-140` |
| Network topology | Hard-coded 32x5 | Configurable but Python-default 32x5 at `include/neurodic/model/ndef_internal_model.hpp:11-16` and `src/model/ndef_internal_model.cpp:7-50` |
| Scale seeds | Deterministic first grid pixel and scalar CPU NCC | Seeded random-in-grid sampling and batched LibTorch unfold NCC at `src/initialization/ndef_precalculation.cpp:45-143` and `:182-246` |
| Scale statistic | Robust scale helper already existed | Retained exact MAD/statistics at `src/initialization/ndef_precalculation.cpp:18-34`, now fed by the migrated sparse pipeline |
| Result | Final fields and projection diagnostics only | History, sample counts, normalization, final/best checkpoints, reference/current points, magnitude and source metadata exported at `python/neurodic/api/ndef_dic.py:72-181` |

The C++ result retains legacy `*_sfm` member names for ABI/source compatibility.
For the current CylinderDIC input, both surface and scaled cameras are already in
the calibration world frame, so YAML export scale is 1.0.
