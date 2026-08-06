# Migration: NDeF-DIC into NeuroDIC

Status: deformation-stage core migrated; dense reference-surface preprocessing
and sparse displacement-scale preprocessing remain separate follow-up stages.

This document describes how the multi-view neural deformation DIC
implementation in `/home/a306/01project/NDeF-DIC` is migrated into the
NeuroDIC C++/LibTorch core. NDeF-DIC is already written in PyTorch, so the
differentiable kernels translate directly to LibTorch operators; the pipeline
stages that are preprocessing map to C++ modules or the thin Python layer.

## 1. Goal and Constraints

NeuroDIC's `NDeFSolver` solves multi-view DIC by learning a continuous 3D
displacement field on a reference surface, supervised by multi-camera
photometric (MSE/ZNSSD) patch losses.

Migration rules:

1. **`NDeFSolver` owns the deformation stage only.** SfM, scale recovery,
   dense surface reconstruction, and sparse patch-DIC precalculation are
   preprocessing that feed the solver.
2. **The differentiable path is LibTorch-only.** Surface points -> model ->
   displacement -> projection -> patch sampling -> loss keeps the autograd
   graph; no NumPy round-trip inside the training loop.
3. **Normalization follows the NDeF-DIC convention** (decision D5): bounding-
   box center/scale input normalization plus a displacement output scale
   derived from the precalculation statistics.
4. **The reference surface is an input contract**, not a solver responsibility.
   The surface dataset (points, normals, visibility, projected UV, visible
   counts) is produced by preprocessing and consumed by `NDeFProblem`.
5. Ported kernels are verified numerically against NDeF-DIC on the same
   cylinder case before acceptance.

## 2. Pipeline Mapping

NDeF-DIC is a config-driven 6-stage pipeline (`run.py --stages ...`). The
stage-to-module mapping:

| NDeF-DIC stage | Purpose | NeuroDIC destination |
|---|---|---|
| `sfm` (pycolmap) | self-calibrated sparse reconstruction of the reference frame | `calibration/colmap_calibration.hpp` (C++ self-calibration from Traditional-DIC) |
| `sfm2world` | chessboard-based physical scale recovery | `calibration/` chessboard scale estimation (Traditional-DIC) |
| `dense` | ROI construction, depth init + ZNSSD refinement, dense reconstruction, surface sampling | preprocessing; C++ where reusable, Python thin layer where not (see D7) |
| `precalculation` | sparse patch-DIC 3D displacement + displacement scale | `initialization/` seed chain + `geometry/triangulation.hpp`; scale stats feed `NDeFProblem` |
| `deformation` | neural deformation field training + world-scale export | `representation/ndef_*`, `problem/ndef_problem.cpp`, `solver/ndef_solver.cpp` |

## 3. Stage Designs

### 3.1 `sfm` -> self-calibration (C++)

- Migrate to `calibration/colmap_calibration.hpp` using the Traditional-DIC
  `calibrate_multiview_colmap_like` (SIFT matching, incremental
  reconstruction, bundle adjustment) instead of pycolmap.
- Output contract: cameras `(K, dist, R, t)` + sparse points + observations,
  stored for the downstream stages. This replaces the `cameras.npz` /
  `observations.npz` products NDeF-DIC currently reads.
- Decision D8: first version reuses the Traditional-DIC C++ self-calibration;
  pycolmap stays only as an optional Python-side fallback.

### 3.2 `sfm2world` -> chessboard scale

- Migrate `estimate_multiview_chessboard_scale` semantics (triangulate board
  corners in SfM units, compare edge lengths against known square size,
  robust trim) into `calibration/`.
- `NDeFProblem` stores the `sfm2world_scale`; the solver trains on SfM-scale
  geometry and exports both SfM-scale and world-scale arrays. The Python thin
  layer reads the migrated chessboard `calibration_scale.json` product.

### 3.3 `dense` -> reference surface preprocessing

Components and mapping:

| NDeF-DIC source | Nature | Migration |
|---|---|---|
| `roi_builder.py` | convex hull + Delaunay + hole/texture checks (numpy/OpenCV) | Python thin layer first; C++ only if reused elsewhere |
| `model_init.py` (SfMDepthFiLMNet) | per-camera depth network, SfM-statistics denormalization | see D7 |
| `dense_znssd.py` (DenseZNSSDLoss) | camera-conditioned depth refinement, ZNSSD loss | see D7 |
| `reconstruction_dense.py` | dense reconstruction from refined depth | Python thin layer |
| `surface_sampler.py` | visibility-aware surface sampling | Python thin layer (numpy preprocessing) |

Decision D7 (pending): the dense depth network is differentiable but it is
reference-surface preprocessing, not a DIC solver. First version keeps
`model_init` + `dense_znssd` in the Python thin layer reusing NDeF-DIC code
as-is; the C++ core consumes only the produced surface dataset. Revisit if the
depth network becomes part of the DIC problem.

Surface dataset contract consumed by `NDeFProblem`:

```
points           (M, 3)  world-scale reference surface points
normals          (M, 3)
visibility_mask  (M, C)  per-camera visibility
projected_uv     (M, C, 2) reference-frame projections
visible_counts   (M,)   number of visible cameras per point
cam_names        (C,)
```

### 3.4 `precalculation` -> sparse displacement + scale

- `patch_dic_precalc.py` runs NCC temporal matching on each source camera,
  cross-camera matching on neighbors, multi-view triangulation of reference
  and current points, MAD outlier rejection, and displacement-scale statistics.
- Reuse the Traditional-DIC seed chain (integer search + subset ICGN) for the
  2D temporal matches, and `geometry/triangulation.hpp` for the multi-view
  triangulation. Temporal patch matching should consume the same
  `ImagePrecomputeContext` contract used by PIN-DIC: mirror-padded images,
  explicit `pad_offset`, and shared B-spline coefficient blocks with CPU/GPU
  views.
- Output: sparse 3D displacement records + `displacement_scale` statistics
  (median/mean/p75/p90/max) feeding the deformation `output_scale`
  (decision D5, NDeF branch).

### 3.5 `deformation` -> `NDeFSolver`

Differentiable kernels translated from NDeF-DIC (PyTorch -> LibTorch):

| NDeF-DIC source | NeuroDIC target | Notes |
|---|---|---|
| `deformation_field.py` (`NeuralDisplacementField`, `PositionalEncoding`) | `model/ndef_internal_model.hpp` (torch `nn::Module`) | tanh MLP 3->HxL->3, optional PE, coord center/scale buffers, output_scale |
| `deformation_loss.py` (`deformation_photometric_mse`, `project_world_torch`, `distort_normalized_torch`, `bilinear_sample_single`, `znssd_patch_loss`) | `solver/ndef_solver.cpp` + `geometry/ndef_geometry.cpp` | migrated: project points, sample patches per camera, MSE/ZNSSD, `1/visible_counts` weighting, invalid-patch penalty |
| `deformation_dataset.py` (`SurfaceDeformationDataset`) | `data/multiview_dataset.hpp` + `problem/ndef_problem.cpp` | hold surface tensors + camera tensors + image stacks |
| `smoothness_loss` | `solver/ndef_solver.cpp` | migrated: Jacobian-norm penalty on normalized coords |
| `train_deformation.py` | `solver/ndef_solver.cpp` | migrated: AdamW loop and best-loss parameter restoration; persistent checkpoints/world-scale dual export remain follow-up |

Key semantics to preserve:

- Deformation is surface-based: `X_def = X_sfm + u_sfm`, where `u` is the
  network output scaled by `output_scale` and denormalized to world units.
- Patch loss: `patch_radius` (default 2 -> 5x5 patch), `min_valid_patch_ratio`,
  `invalid_patch_penalty` for out-of-bounds patches.
- Multi-camera weighting: `weights = 1 / visible_counts`.
- Only the deformation stage runs in `NDeFSolver`; everything upstream is
  preprocessing.

## 4. Decisions

### 4.1 Confirmed Decisions

| ID | Decision |
|---|---|
| D8 | SfM stage uses the Traditional-DIC C++ self-calibration (`colmap_calibration.hpp`); pycolmap is an optional Python-side fallback. |
| D9 | `NDeFSolver` owns only the deformation stage; surface dataset + scale are input contracts. |
| D10 | Done. Both SfM-scale and world-scale arrays are exported (NDeF-DIC dual-scale convention). |

### 4.2 Open Decision

| ID | Status | Decision |
|---|---|---|
| D7 | **PENDING** | `dense` depth network (model_init + dense_znssd): first-version recommendation is to keep it in the Python thin layer reusing NDeF-DIC code as-is, with the C++ core consuming only the surface dataset. Alternative: migrate the depth network into the C++/LibTorch core (as a preprocessing solver). Not yet decided; revisit when the dense stage is reached in the landing order. |

## 5. Verification Strategy

| Stage | Reference | Criterion |
|---|---|---|
| projection / distortion / patch sampling | NDeF `deformation_loss.py` | identical tensors on the CylinderDIC case |
| deformation photometric loss | NDeF `train_deformation.py` | identical loss curve start point on same data/model |
| end-to-end deformation | NDeF frame 01 result | displacement fields match within tolerance |

## 6. Landing Order

```
N1  geometry/ndef_geometry: projection + distortion + patch sampling (LibTorch)
N2  deformation losses (photometric MSE/ZNSSD patch + smoothness)
N3  model/ndef_internal_model (tanh MLP + PE + coord normalization + output_scale)
N4  data/multiview_dataset + problem/ndef_problem (surface dataset + cameras)
N5  solver/ndef_solver (AdamW loop + checkpoint + dual-scale export)
N6  precalculation (seed chain + triangulation + scale stats)
N7  sfm2world chessboard scale (calibration)
N8  Python thin layer wiring: dense preprocessing + run pipeline (ndef_dic entry)
```

N1-N5 are the solver core and depend only on the surface dataset contract;
N6-N7 feed that contract and can run in parallel; N8 assembles the pipeline.

## 7. Relationship to the Other Migrations

- Shares with PIN-DIC: `geometry/projection` + `triangulation` (Traditional-DIC
  batch 1-2), calibration + self-calibration (batch 2), and the photometric
  MSE/ZNSSD loss concepts (MSPINN S3) — the NDeF patch loss reuses the same
  ZNSSD normalization idea in a per-camera patch form.
- The seed chain (Traditional-DIC batch 3) is reused by `precalculation` for
  temporal 2D matches.
- NDeF normalization (bounding-box input + displacement output scale) is
  deliberately distinct from PIN-DIC `scale_uv` output normalization
  (decision D5).
