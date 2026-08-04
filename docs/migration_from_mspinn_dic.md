# Migration: MSPINN-DIC into NeuroDIC

Status: Design draft.

This document describes how the validated PINN-DIC Python implementation in
`/home/a306/01project/MSPINN-DIC` is migrated into the NeuroDIC C++/LibTorch
differentiable core. MSPINN-DIC is written in JAX; NeuroDIC's differentiable
core is LibTorch. The migration therefore translates JAX function semantics
into LibTorch tensor operations rather than copying code.

## 1. Goal and Constraints

NeuroDIC owns a C++-first neural DIC core. The PIN-DIC 2D pipeline follows the
MSPINN-DIC design: one continuous neural field per ROI, supervised by
photometric (MSE/ZNSSD) loss with optional seed-point pre-training.

Migration rules:

1. **Only the single-network PINN path is migrated.** The FBPINN domain
   decomposition stack is out of scope (architecture rule 12).
2. **The differentiable path is LibTorch-only.** `torch::Tensor` from model
   output to loss; no NumPy/Eigen/OpenCV round-trip inside the training loop.
3. **Coordinate convention is `(x, y)` columns**: `coordinates[N, 2]` has x
   in column 0 and y in column 1. Pixel access on an image tensor remains
   `image[y, x]`.
4. **ZNSSD is implemented with MSPINN semantics first** (reflect padding,
   box-kernel convolution, ROI-count normalization) so numerics can be
   validated directly against MSPINN-DIC before any optimization.
5. **Normalization is split by solver family**: PIN-DIC uses MSPINN-style
   output normalization via `scale_uv`; NDeF-DIC uses NDeF-style input
   normalization via bounding-box center/scale.
6. Ported kernels are verified numerically against MSPINN-DIC on the same
   images before being accepted.

## 2. Migration Scope

### 2.1 Migrated (semantics translated to LibTorch)

| MSPINN-DIC source | NeuroDIC target | Role |
|---|---|---|
| `DIC_problem.py` (`DIC_MSE`, `DIC_ZNSSD`) | `loss/photometric.cpp`, `loss/znssd.cpp` | differentiable losses |
| `DIC_readImg.py` (`beta_nth`, `get_QK`, `form_bcoef`, `get_QK_B_QKT`, `image_gradient_from_bcoef`) | `interpolation/bspline_coefficients.cpp` (+ torch tensor output) | coefficient precomputation |
| `DIC_seedcalc.py` (`interpqbs`) | `interpolation/torch_bspline.cpp` | differentiable sampler |
| `DIC_seedcalc.py` (`_analyze_sift`) | `initialization/sift_grid_seed_initializer.cpp` | optional SIFT grid seed source |
| `DIC_networks.py` (FCN/SIREN/Fourier; Adaptive variants deferred) | `model/*` | neural architectures |
| `DIC_seed_trainer.py` (`PINN_seed_loss`, `train_seeds_pinn`) | `solver/pin_solver.cpp` seed pre-training phase | supervised initialization |
| `DIC_trainers.py` (`PINNTrainer` only) | `solver/pin_solver.cpp` | training loop |
| `DIC_analysis.py` (PINN branch, `scale_uv` computation) | `problem/pin_problem.cpp`, `problem_builder.hpp` | problem assembly |

### 2.2 Not migrated

| MSPINN-DIC source | Reason |
|---|---|
| `DIC_decompositions.py` (rectangular domain decomposition, multi-scale) | FBPINN only; out of scope |
| `DIC_windows.py` (partition-of-unity windows) | FBPINN only; out of scope |
| `DIC_schedulers.py` (subdomain activation schedulers) | FBPINN only; out of scope |
| `DIC_seedcalc.py` JAX NCC + IC-GN iteration body | replaced by the validated C++ seed chain from Traditional-DIC |
| `DIC_importlib.py`, `utils/*`, TensorBoard/plot helpers, illumination test scripts | Python/plumbing layer; not part of the C++ core |

## 3. Differentiable Kernel Designs

### 3.1 `interpqbs` -> `TorchBSplineInterpolator::evaluate`

Reference (JAX, `DIC_seedcalc.py`):

```python
xs_floor = jnp.floor(xs).astype(jnp.int32)        # index is non-differentiable
QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]         # (N, N, N) block lookup
x_vec = xd[:, None] ** powers[None, :]            # power vector
tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)   # polynomial evaluation
values = jnp.einsum("ni,ni->n", tmp, x_vec)
```

LibTorch contract:

- `coefficients`: `[Hpad, Wpad, N, N]` local polynomial blocks computed on the
  mirror-padded image (MSPINN `QKBQKT_def` semantics), `N = degree + 1`.
- `coordinates`: `[N, 2]`, columns `(x, y)`.
- `evaluate` returns `[N]`; `gradient` returns `[N, 2]` = `(dI/dx, dI/dy)`.
- `floor().to(torch::kLong)` indexing is inherently non-differentiable
  (equivalent to JAX `stop_gradient`); no extra detach needed.
- Boundary clamping happens on indices before lookup; an out-of-bounds mask is
  returned to callers for loss weighting (MSPINN keeps an OOB mask in the MSE
  path and relies on reflect padding in the ZNSSD path).
- `powers` is a constant tensor; `pow` must keep the gradient with respect to
  the base (`xd`, `yd`).
- Integer-pixel invariant for testing: with `xd = yd = 0`, the power vectors
  reduce to the first unit vector, so `evaluate(blocks, [x, y]) == blocks[y, x, 0, 0]`.

`gradient` uses the analytic derivative of the polynomial kernel
(`d(x^k)/dx = k * x^(k-1)`, with the 0-th power term dropped) to avoid a
second autograd pass through the sampler.

### 3.2 `get_QK_B_QKT` -> coefficient block precomputation

Reference (JAX, `DIC_readImg.py`):

```python
M = jnp.einsum("ij,hwjk,kl->hwil", QK, blocks, QKT)   # (H, W, N, N)
```

- Preprocessing only; runs once under `torch::NoGradGuard` before training.
- Coefficients are computed on the shared mirror-padded images owned by
  `ImagePrecomputeContext`; PIN-DIC coordinates are shifted internally by the
  context `pad_offset`.
- Output is the `[Hpad, Wpad, N, N]` tensor consumed by the sampler.
- Decision: the LibTorch CPU tensor is the single stored coefficient form; GPU
  training obtains a lazy `coeff_cpu.to(device)` copy. The Eigen-based
  `BSplinePrecomputedImage` (Traditional-DIC) becomes an optional CPU view for
  the seed chain only. See decision D4.

### 3.3 `DIC_MSE` / `DIC_ZNSSD` -> losses

Reference (JAX, `DIC_problem.py`).

- MSE: warp deformed coefficients at `(x+u, y+v)` via the sampler, compare
  with reference intensity.
- ZNSSD (MSPINN semantics, decision D4): reflect-pad reference and scattered
  warped image, box-kernel convolution for local sums, normalize by
  `n = max(S_roi, 1)` valid-pixel count, then
  `((ref - mu_r)/sigma_r * sigma_c - (cur - mu_c))^2`.
- The ROI-count normalization detail (`n = max(S_roi, 1)`) is critical near
  ROI boundaries and must be preserved.

### 3.4 `PINN_model` + unnorm -> representation

Reference (JAX):

```python
u_raw = network_fn(params, x_norm)
u = u_raw * sd + mu        # scale_uv unnormalization
```

LibTorch: `pin_displacement_field.cpp` implements `FieldRepresentation::decode`
with `(mean_u, mean_v, halfrange_u, halfrange_v)` stored in the problem
(decision D5, PINN branch). The constants come from the seed chain output.

### 3.5 `PINNTrainer.train` -> `PINSolver`

- `static_params` (images, coefficient blocks, mask, domain) live in
  `PINProblem`; `trainable_params` (network weights) live in a
  `torch::nn::Module`.
- Two-phase training: (1) optional supervised seed pre-training on
  `seed_pos`/`seed_uv` (MSE, with optional gradient-norm smoothness term);
  (2) photometric optimization with Adam, optional L-BFGS refinement.
- `jacfwd` chained gradients for strain become `torch::autograd::grad` with
  respect to input coordinates once per prediction; strain
  (`exx = ux`, `exy = (uy + vx)/2`, `eyy = vy`) is computed in postprocess.

### 3.6 MSPINN SIFT grid seeds -> `SiftGridSeedInitializer`

MSPINN-DIC's SIFT branch is migrated as an optional C++ seed source. It does
not replace the Traditional-DIC integer-search + IC-GN seed path; both feed
the same PIN-DIC seed pre-training interface.

Reference semantics from `DIC_seedcalc.py::_analyze_sift`:

1. Detect SIFT keypoints/descriptors on the full reference and deformed images.
2. Match descriptors with FLANN KNN (`k = 2`).
3. Keep matches passing Lowe's ratio test (`m.distance < 0.75 * n.distance`).
4. Compute displacement `uv = pt_def - pt_ref`.
5. For each ROI, keep matches whose rounded reference point lies inside the ROI
   mask.
6. Split the ROI bounding box into a grid sized from `seeds_number` and ROI
   aspect ratio.
7. Select the best match per occupied cell using
   `quality = keypoint.response / (match.distance + eps)`.
8. Reject displacement outliers with a median/MAD filter.

Suggested C++ options:

```cpp
struct SiftGridSeedOptions {
    int target_seed_count{128};
    double lowe_ratio{0.75};
    int flann_trees{5};
    int flann_checks{50};
    double mad_threshold{4.5};
    int min_seeds_per_roi{3};
};
```

## 4. Confirmed Decisions

| ID | Decision |
|---|---|
| D1 | FBPINN excluded: `DIC_decompositions.py`, `DIC_windows.py`, `DIC_schedulers.py` are not migrated; only the single-network PINN path remains. |
| D2 | Coordinate convention: `(x, y)` column order everywhere (`coordinates[N, 2]`); image access stays `image[y, x]`. |
| D3 | ZNSSD first implemented with MSPINN semantics (reflect pad + box convolution + ROI-count normalization) for numerical parity; optimization deferred. |
| D4 | Coefficients stored as one shared LibTorch CPU tensor `[Hpad, Wpad, N, N]` computed on mirror-padded images; GPU training uses lazy device copies, and Eigen form is an optional CPU view for the seed chain only. |
| D5 | Normalization split: PIN-DIC uses `scale_uv` output normalization (MSPINN); NDeF-DIC uses bounding-box center/scale input normalization (NDeF-DIC). |
| D6 | MSPINN's JAX NCC + IC-GN body is not migrated; it is replaced by the validated C++ seed chain from Traditional-DIC. MSPINN's SIFT grid selection strategy is migrated as an optional C++ seed initializer. |

## 5. Verification Strategy

| Stage | Reference | Criterion |
|---|---|---|
| `torch_bspline` vs Eigen bspline | Traditional-DIC `BSplineInterpolator` | gradcheck + random coordinate parity (1e-9) |
| `torch_bspline` vs `interpqbs` | MSPINN `DIC_seedcalc.py` | identical values on the same `QKBQKT_def` |
| ZNSSD loss | MSPINN `DIC_ZNSSD.loss_fn` | identical loss on same images + same displacement field |
| End-to-end PIN-DIC 2D | MSPINN ring case | u/v/exx/eyy field comparison (leff/RMS) |

## 6. Landing Order

```
S1  TorchBSplineInterpolator (evaluate + gradient) -> gradcheck + dual parity
S2  coefficient block precomputation (torch form) -> parity with Traditional blocks
S3  shared mirror-padding/ImagePrecomputeContext integration
S4  MSPINN-style SIFT grid seed initializer in C++/OpenCV
S5  photometric + znssd losses -> parity with MSPINN loss numerics
S6  pin_displacement_field representation (scale_uv unnormalization)
S7  model (MLP/SIREN/Fourier in LibTorch)
S8  PINProblem assembly (domain sampling + static tensors)
S9  PINSolver 2D: seed pre-training -> photometric training -> L-BFGS refinement
S10 thin Python wiring (pin_dic entry point)
```

S1-S3 establish the shared interpolation/padding foundation. S4 depends on
OpenCV but produces only seed tensors. S5-S10 depend on the shared
Traditional-DIC seed/precompute contracts (batch 3) and can then build the
PIN-DIC training path.

## 7. Relationship to the Traditional-DIC Migration

- Traditional-DIC batches 1-3 provide the seed chain: integer search, subset
  ICGN, B-spline coefficient precomputation, and the `scale_uv` output
  contract consumed by the PIN solver.
- The MSPINN-style SIFT grid initializer is a second seed source, implemented
  in C++/OpenCV, that produces the same `SeedSet` contract as the
  Traditional-DIC seed chain.
- Traditional-DIC batch 4 and MSPINN-DIC S1-S5 both touch the differentiable
  core; `torch_bspline` and the losses are implemented once (MSPINN semantics)
  and shared by both lines.
- The non-differentiable coefficient kernel is written once (decision D4) on
  mirror-padded images and feeds both the seed chain and the differentiable
  sampler.
