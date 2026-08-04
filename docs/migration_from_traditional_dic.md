# Migration: Traditional-DIC C++ Modules into NeuroDIC

Status: Design draft.

This document describes how the validated C++ modules in
`/home/a306/01project/Traditional-DIC` are migrated into the NeuroDIC C++
scientific core, which modules are reused as-is, which are rewritten with
LibTorch, and in what order the migration is verified.

## 1. Goal and Constraints

NeuroDIC is a C++-first neural DIC library: a LibTorch differentiable core
with pybind11 bindings and a thin Python API. The validated traditional
algorithms (calibration, B-spline coefficients, integer search, subset ICGN,
geometry, postprocess) already exist in Traditional-DIC as C++17/Eigen code.

Migration rules:

1. **Non-differentiable path may reuse Eigen/OpenCV code as-is.** Seed search,
   calibration, B-spline coefficient precomputation, and triangulation run
   before the training loop and never touch the autograd graph.
2. **The differentiable path (model -> representation -> geometry -> B-spline
   warp -> loss) is LibTorch-only.** No NumPy/Eigen/OpenCV round-trip inside
   the training loop.
3. Namespace `dic::` becomes `neurodic::`; headers move from `include/dic/`
   to `include/neurodic/`.
4. Add characterization tests for migrated modules so numerical parity with
   Traditional-DIC is preserved. Traditional-DIC does not currently provide a
   complete standalone `tests/` tree to copy verbatim, so NeuroDIC tests are
   built from migrated APIs, existing examples/benchmarks, and shared case data.
5. Do not port Mesh-DIC (T3/Q4/Q8 global solvers) or FBPINN/MSPINN domain
   decomposition; both are out of scope for the first NeuroDIC version.

## 2. Module Inventory and Mapping

### 2.1 Pure migration (no algorithm change)

| Traditional-DIC source | NeuroDIC target | Notes |
|---|---|---|
| `core/image.hpp/.cpp` | `data/image.hpp/.cpp` | floating-point grayscale + path loading |
| `core/mask.hpp/.cpp` | `data/` or `core/` | binary mask |
| `core/roi.hpp/.cpp` | `data/roi.hpp/.cpp` | ROI container |
| `core/result.hpp` | `core/result.hpp` | result containers (merge with skeleton) |
| `core/types.hpp` | `core/types.hpp` | enum unification |
| `geometry/projection.cpp` | `geometry/projection.hpp` | pinhole projection |
| `geometry/triangulation.cpp` | `geometry/triangulation.hpp` | DLT triangulation |
| `postprocess/filtering.cpp` | `postprocess/filtering.hpp` | smoothing filters |
| `postprocess/strain_2d.cpp`, `strain_3d.cpp` | `postprocess/strain.hpp` | strain fields |
| `postprocess/coordinate_transform.cpp` | `geometry/coordinate_transform.hpp` | global/local transforms |

Verification: add matching characterization tests
(`test_projection`, `test_triangulation`, `test_roi_mask`, `test_image`) and
run them under `ctest`.

### 2.2 Calibration and self-calibration

| Traditional-DIC source | NeuroDIC target | Notes |
|---|---|---|
| `calibration/camera_model.hpp` | `calibration/camera_model.hpp` | fill existing skeleton |
| `calibration/mono_calibration.cpp` | `calibration/mono_calibration.hpp` | Zhang mono |
| `calibration/stereo_calibration.cpp` | `calibration/stereo_calibration.hpp` | stereo Zhang + outlier rejection |
| `calibration/multiview_calibration.cpp` (`calibrate_multiview_colmap_like`) | `calibration/colmap_calibration.hpp` | SIFT self-calibration, incremental reconstruction, BA |
| `calibration/multiview_calibration.cpp` (`estimate_multiview_chessboard_scale`) | `calibration/calibration_result.hpp` | sfm2world chessboard scale |

Calibration stays independent of solvers (architecture rule 10). The
`CalibrationType { NONE, MONO, STEREO, COLMAP }` enum in the skeleton already
matches this layout.

### 2.3 Seed chain: B-spline coefficients, integer search, subset ICGN

This chain produces seed displacements and the `scale_uv` normalization used
by the PIN solver. It is preprocessing, not the main solver.

| Traditional-DIC source | NeuroDIC target | Notes |
|---|---|---|
| `interpolation/bspline.cpp` (`form_coefficients`, `build_qk`, `build_local_polynomial_block`) | `interpolation/bspline_coefficients.cpp` | non-differentiable precomputation; add `to_tensor()` for the training path |
| `initialization/integer_search.cpp` | `initialization/integer_search.hpp` | integer-pixel ZNSSD search |
| `initialization/sift_initializer.cpp`, `feature_matcher.cpp` | `initialization/sift_initializer.hpp` | feature-based seed init (OpenCV) |
| `subset/seed/seed_selector.cpp` | `initialization/` (seed selector) | automatic seed selection |
| `subset/seed/reliability_propagation.cpp` | `initialization/` | reliability-guided propagation |
| `subset/solver/icgn.cpp` (first-order ZNSSD) | reusable as seed subpixel solver | IC-GN with B-spline interpolation |
| `subset/shape/first_order.hpp` | reusable as seed shape function | 6-parameter affine warp |
| `correlation/znssd.cpp` | non-differentiable reference for `loss/znssd.hpp` | keep C++ version for seeds |

#### 2.3.1 Shared image padding and precompute context

Mirror padding is a shared preprocessing concern, not a private detail of the
subset solver. The padded image must be large enough for both the integer seed
search window and the sub-pixel subset iteration window:

```
pad =
    integer_search_radius
  + max(coarse_subset_radius, fine_subset_radius, subset_radius)
  + bspline_border
```

This mirrors the validated Traditional-DIC rule
`recommended_subset_padding = search_radius + max_subset_radius +
bspline_border`, but NeuroDIC promotes it into a common precompute context:

```
ImagePrecomputeContext
  reference_padded       mirror-padded reference image
  deformed_padded        mirror-padded deformed image
  roi_padded             zero-padded ROI mask
  pad_offset             integer coordinate offset
  ref_coefficients       B-spline blocks on reference_padded
  def_coefficients       B-spline blocks on deformed_padded
```

Public APIs and result files use original image coordinates. Internal seed
search, ICGN, and differentiable interpolation use padded coordinates
`(x + pad_offset, y + pad_offset)`. This offset must be explicit in every
problem/seed object to prevent accidental mixing of original and padded spaces.

The B-spline coefficient blocks are computed once on the padded images and
shared by the classical seed path and the LibTorch PIN path:

```cpp
struct BSplineCoefficientBlock {
    int height;
    int width;
    int degree;
    int pad_offset;
    torch::Tensor coeff_cpu;   // [Hpad, Wpad, degree+1, degree+1], CPU
    torch::Tensor coeff_gpu;   // lazy optional cache: coeff_cpu.to(device)
};
```

Subset seed search consumes the CPU coefficients or a CPU-compatible view.
PIN-DIC/NDeF differentiable losses consume the same coefficient tensor moved to
the active training device. No second B-spline precomputation is allowed for
the PIN path.

#### 2.3.2 Seed sources for PIN-DIC initialization

NeuroDIC supports multiple seed sources that all produce the same `SeedSet`
contract:

```cpp
struct SeedSet {
    torch::Tensor seed_pos;   // [N, 2], original coordinates (x, y)
    torch::Tensor seed_uv;    // [N, 2], displacement (u, v), pixels
    torch::Tensor scale_uv;   // [4], mean_u, mean_v, halfrange_u, halfrange_v
};
```

1. **Traditional subset seed source**: uniform/K-means candidate placement,
   integer NCC/ZNSSD search, then first-order IC-GN sub-pixel refinement. This
   is the primary seed source for PIN-DIC pre-training.
2. **MSPINN-style SIFT grid seed source**: migrated as a C++ OpenCV
   implementation. It performs full-image SIFT, FLANN KNN matching, Lowe ratio
   filtering, per-ROI grid selection, quality ranking, and MAD displacement
   outlier rejection. It is an optional initializer for cases where feature
   matches provide a better global displacement prior.
3. **Fallback seed source**: zero/constant displacement initialization for
   debugging or low-feature images.

The output contract of the seed chain:

```
seed_pos:  (N, 2) pixel coordinates inside ROI
seed_uv:   (N, 2) integer/subpixel displacements in pixels
scale_uv:  (4,)  (mean_u, mean_v, halfrange_u, halfrange_v)
```

`scale_uv` becomes the unnormalization constants for the network output, the
same role `BufferManager.scale_uv` plays in MSPINN-DIC.

### 2.4 Differentiable core (LibTorch rewrite)

The differentiable kernels are rewritten in LibTorch using MSPINN-DIC as the
reference semantics. Details are tracked in the MSPINN-DIC migration strategy;
the module mapping is:

| NeuroDIC target | Reference (MSPINN-DIC) | Notes |
|---|---|---|
| `interpolation/torch_bspline.cpp` | `interpqbs` + `get_QK_B_QKT` | coefficient-block lookup + polynomial eval |
| `loss/photometric.cpp` | `DIC_MSE.loss_fn` | warp + MSE |
| `loss/znssd.cpp` | `DIC_ZNSSD.loss_fn` | local mean/variance normalization |
| `representation/*` | `PINN_model` + unnorm | decode physical fields from network output |
| `solver/pin_solver.cpp` | `PINNTrainer.train` | Adam (+ optional L-BFGS) loop |
| `model/*` | `DIC_networks.py` | `torch::nn::Module` implementations |

## 3. Migration Order and Verification

Each batch is independently verifiable against the Traditional-DIC baseline.

### Batch 1: core/data/geometry/postprocess

- Port `Image`, `Mask`, `ROI`, `result`, `types`.
- Port `projection`, `triangulation`, `filtering`, `strain`, coordinate transforms.
- Verify: NeuroDIC characterization tests pass; identical numerical output to
  Traditional-DIC.

### Batch 2: calibration + self-calibration

- Port mono/stereo Zhang and the COLMAP-like multiview self-calibration.
- Port the chessboard scale estimation.
- Verify: `tests/calibration/test_calibration_models.cpp` and the stereo
  outlier-rejection paths pass; run on `case/Stereo/plate_center_load`
  calibration images.

### Batch 3: seed chain

- Port shared mirror-padding and `ImagePrecomputeContext`.
- Port B-spline coefficient precomputation on padded images (degree 1/3/5,
  exact prefilter, local polynomial blocks) and expose CPU/GPU tensor views.
- Port integer search + seed selector + reliability propagation +
  first-order ZNSSD ICGN.
- Port the MSPINN-style SIFT grid seed initializer as a C++ OpenCV component.
- Verify: `tests/initialization/test_integer_search.cpp`,
  `tests/subset/test_icgn.cpp`, `tests/subset/test_seed_selector.cpp`;
  reproduce ring-case seed stats.

### Batch 4: differentiable core

- Implement `TorchBSplineInterpolator`; verify with gradcheck and by
  comparing `torch_bspline` values against the non-differentiable
  `BSplineInterpolator`.
- Implement photometric/znssd losses; compare against MSPINN-DIC loss
  numerics on the same images.
- Implement `PINSolver` 2D end-to-end; compare against MSPINN-DIC results.

## 4. Open Decisions

1. **Coefficient representation bridge**: the shared
   `BSplineCoefficientBlock` owns one CPU tensor and lazy GPU copies. The
   Eigen-based `BSplinePrecomputedImage` is allowed only as a
   CPU/non-differentiable compatibility view for seed search; it must not
   create a second authoritative coefficient representation.
2. **OpenCV dependency**: keep `NEURODIC_HAS_OPENCV` conditional compilation
   for image IO / SIFT / calibration, mirroring Traditional-DIC.
3. **Seed chain location**: sub-pixel ICGN currently lives under `subset/` in
   Traditional-DIC. In NeuroDIC it is a seed generator, so it should live
   under `initialization/` rather than being promoted to a solver.
4. **Eigen in NeuroDIC**: Eigen remains a core dependency for the
   non-differentiable path; it must not appear in headers used by the
   differentiable path.
5. **Coordinate offset discipline**: all public seed/result coordinates are in
   original image space; only `ImagePrecomputeContext` and internal samplers
   operate in padded space.
