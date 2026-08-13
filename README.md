# NeuroDIC

NeuroDIC is a C++-first digital image correlation project with a LibTorch
differentiable core, pybind11 bindings, and a thin Python workflow layer.

```text
NeuroDIC
├── PINSolver
│   ├── PIN-DIC 2D
│   ├── PIN-DIC Stereo
│   └── PINMultiSolver (pin_multi_slover, pairwise multi-view)
└── NDeFSolver
    └── NDeF multi-view surface deformation
```

The current repository contains working PIN and NDeF numerical paths. The NDeF
CylinderDIC route includes reference-surface reconstruction, sparse patch-DIC
displacement-scale estimation, neural deformation training, full-surface
inference, checkpoints, diagnostics, and 3D visualizations.

## Development environment

The authoritative environment and troubleshooting notes are in
[`docs/development_environment.md`](docs/development_environment.md). For the
validated local setup, use:

```bash
export ND_ENV=/home/a306/miniconda3/envs/neurodic
export CXX_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-g++
export C_COMPILER=$ND_ENV/bin/x86_64-conda-linux-gnu-gcc
export CUDAHOSTCXX=$CXX_COMPILER

$ND_ENV/bin/cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH=$ND_ENV \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DPython_EXECUTABLE=$ND_ENV/bin/python \
    -DNEURODIC_ENABLE_TORCH=ON \
    -DNEURODIC_BUILD_PYTHON=ON \
    -DNEURODIC_BUILD_TESTS=ON \
    -DNEURODIC_USE_EIGEN=OFF \
    -DNEURODIC_USE_OPENCV=OFF

CUDAHOSTCXX=$CXX_COMPILER $ND_ENV/bin/cmake --build build -j
CUDAHOSTCXX=$CXX_COMPILER $ND_ENV/bin/ctest --test-dir build --output-on-failure
```

Do not use base Python or the system `g++` for CUDA/LibTorch builds. For
build-tree Python imports:

```bash
export PYTHONPATH=$PWD/python:$PWD/build/python
export MPLCONFIGDIR=/tmp/neurodic-matplotlib
$ND_ENV/bin/python -c "import neurodic; print(neurodic.native_available())"
```

## Coding-Agent / Skill Usage

The canonical, vendor-neutral coding-agent entry is
[`skills/neurodic/SKILL.md`](skills/neurodic/SKILL.md). It supervises the
versioned control-plane CLI and JSON/filesystem contracts; NeuroDIC itself does
not embed an LLM agent, MCP server, or vendor SDK.

From a source checkout, use the portable CLI prefix:

```bash
PYTHONPATH=python /home/a306/miniconda3/envs/neurodic/bin/python -m neurodic.cli --help
```

Start with `inspect`, then `evaluate` and `diagnose`; any recommendation remains
a bounded hypothesis that must be dry-run planned before an explicitly approved
guarded execution. See [`docs/agent_compatibility.md`](docs/agent_compatibility.md)
for the command/JSON boundary, current execution coverage, mutation rules, and
compatibility limitations.

Across solver families, the control plane supports inspection, evaluation,
diagnosis, bounded recommendation, trial planning, comparison, and best
management. Runtime capability is emitted in each planned action and guarded
execution fails closed when unsupported. Current real execution coverage is
partial: only PIN Multi single-pair `pair_roi` is guarded; PIN, Stereo, PIN Multi
solve/fusion, and NDeF full scientific execution are not claimed as supported.

## Running NDeF clearly

The validated example configuration is
[`config/ndef_multi.yaml`](config/ndef_multi.yaml), and its case root is
`case/Multi/CylinderDIC`.

### NDeF input contract

Before deformation training, the case needs:

- synchronized images under `images/<camera>/`;
- coherent scaled cameras in
  `result/calibration/calibration_result_scaled.json`;
- per-camera ROI masks in `result/mask/per_camera/`;
- the reference-surface dataset
  `result/surface/deformation_surface_dataset.npz`.

The surface NPZ contains:

```text
points, normals, source_camera, visibility_mask, projected_uv,
projected_depth, depth_abs_error, visible_counts, cam_names
```

`points`, `visibility_mask`, `projected_uv`, and `visible_counts` directly feed
deformation training. The remaining fields are retained in output artifacts for
diagnostics and future refinements.

Image selection currently follows the sorted files in each camera folder:

- the first image is the reference image;
- `case.frame: -1` selects the last image as the current/deformed image;
- another integer may be used to select a different synchronized current frame.

### Stage 1: reconstruct the reference surface

Run this only when the surface dataset does not already exist:

```bash
$ND_ENV/bin/python -u -c \
  "import neurodic; neurodic.pretrain_ndef_surface('config/ndef_multi.yaml')"
```

This runs sparse depth pretraining, dense ZNSSD refinement, dense inference, and
visibility/depth-consistency fusion. The deformation hand-off is written to:

```text
case/Multi/CylinderDIC/result/surface/deformation_surface_dataset.npz
```

If this NPZ already exists and is accepted, do not rerun Stage 1; start from
Stage 2. Surface fusion sampling and deformation-training point sampling are
separate algorithms.

### Stage 2: precompute the deformation scale

```bash
$ND_ENV/bin/python -u -c \
  "import neurodic; neurodic.ndef_sparse_precalculation('config/ndef_multi.yaml')"
```

This stage runs the C++/LibTorch sparse multi-view patch-DIC route:

1. seeded spatially distributed ROI points per source camera;
2. temporal and cross-camera batched NCC matching;
3. multi-view reference/current triangulation;
4. reprojection filtering and displacement-magnitude MAD filtering;
5. mean/median/p75/p90/max displacement-scale statistics.

Outputs:

```text
result/ndef/precalculation/sparse_tracks.npz
result/ndef/precalculation/sparse_scale.json
```

`precalculation.statistic` selects which robust statistic becomes the neural
field output scale. CylinderDIC currently uses `mean`.

### Stage 3: train and infer the deformation field

```bash
$ND_ENV/bin/python -u -c \
  "import neurodic; neurodic.ndef_dic('config/ndef_multi.yaml')"
```

The solver consumes every reference-surface point as one global continuous
field. Each optimizer step samples point indices uniformly with replacement.
The network predicts three world-coordinate displacement components:

```text
X_current = X_reference + [Ux, Uy, Uz]
```

The CylinderDIC YAML currently uses:

- 5 hidden Tanh layers of width 32;
- Fourier encoding disabled;
- batch size 51,172;
- 1,000 epochs and `ceil(N / batch)` steps per epoch;
- AdamW with learning rate 0.003 and zero weight decay;
- 5x5 photometric MSE patches;
- fixed reference visibility and `1 / visible_counts` camera weighting;
- smoothness weight zero, matching the original Python configuration;
- deterministic training seed 23.

Important configuration groups:

| YAML group | Purpose |
|---|---|
| `case` | case root, images, calibration, masks, current frame, surface NPZ |
| `deformation_model` | hidden width/layers, Fourier encoding, fallback output scale |
| `precalculation.sparse` | ROI seeds, NCC searches, thresholds, batch and seed |
| `precalculation` | sparse displacement file and selected scale statistic |
| `deformation_training` | device, epochs, batch, optimizer, loss, patches and seed |
| `output` | result and visualization roots |

### Run all NDeF stages

For a new case that already has calibration and ROI masks:

```bash
$ND_ENV/bin/python -u - <<'PY'
import neurodic

config = "config/ndef_multi.yaml"
neurodic.pretrain_ndef_surface(config)       # skip when the accepted surface NPZ exists
neurodic.ndef_sparse_precalculation(config)
neurodic.ndef_dic(config)
PY
```

For the current CylinderDIC case, whose surface NPZ already exists, run only:

```bash
$ND_ENV/bin/python -u - <<'PY'
import neurodic

config = "config/ndef_multi.yaml"
neurodic.ndef_sparse_precalculation(config)
neurodic.ndef_dic(config)
PY
```

### NDeF outputs

Numerical outputs are under `case/Multi/CylinderDIC/result/ndef/`:

```text
precalculation/sparse_tracks.npz
precalculation/sparse_scale.json
reconstruct/reference_surface.npz
reconstruct/current_surface.npz
deformation/reference_to_current.npz
deformation/deformation_field.pt
deformation/deformation_field_best.pt
diagnostics/projection.npz
diagnostics/training.npz
diagnostics/training_history.json
diagnostics/summary.json
```

The deformation NPZ contains reference/current points, `[Ux,Uy,Uz]`, displacement
magnitude, and compatible coordinate-scale arrays. `training.npz` contains the
eight-column training history, per-point sample counts, coordinate normalization,
batch/epoch semantics, output scale, and seed.

True 3D visualizations are written under
`case/Multi/CylinderDIC/visualization/ndef/`:

```text
reconstruct/reference_surface.png
reconstruct/current_surface.png
deformation/magnitude.png
deformation/displacement_x.png
deformation/displacement_y.png
deformation/displacement_z.png
deformation/displacement_components_3d.png
diagnostics/training_loss.png
diagnostics/valid_observations.png
```

The reconstruction figures use actual XYZ axes. The deformation figures plot
the reference surface in 3D and color it by magnitude, Ux, Uy, or Uz; component
color scales are symmetric around zero.

The detailed NDeF solver notes are in
[`docs/ndef_solver.md`](docs/ndef_solver.md).

## Running pairwise multi-view PIN-DIC (pin_multi_slover)

The independent multi-camera PIN route runs per selected camera pair: pair ROI
masks come from reference-time SIFT matches (not NDeF masks), each pair solves
three planar PIN fields in C++ (`A0→B0`, `A0→Ak`, `A0→Bk`), and every pair is
reconstructed independently into `X0`, `Xk` and `dX`.

```bash
$ND_ENV/bin/python -u -c \
  "import neurodic; neurodic.pin_multi_slover_dic('config/pin_multi.yaml')"
```

The example configuration is [`config/pin_multi.yaml`](config/pin_multi.yaml);
its CylinderDIC case root is `case/Multi/CylinderDIC`. Camera pairs are selected
with `camera_pairs.selection: auto_spatial_neighbors` (neighbor topology from
the calibration `camera_pairs.json`, closed ring with `wrap: true`). Outputs:

```text
result/pin_multi_slover/
  pair_roi/<pair_id>/       SIFT matches, left/right masks, overlay, meta
  pairs/<pair_id>/disp/     three planar PIN fields
  pairs/<pair_id>/reconstruct/{reference,current}.npz
  pairs/<pair_id>/deformation/initial_to_current.npz
  pairs/<pair_id>/quality/  per-point reason codes + quality.json
  fused/                    optional pairwise fusion (disabled by default)
  manifest.json             pair ROIs, solve statistics, fusion summary
visualization/pin_multi_slover/...
```

Per-point reason codes (`quality/reason_codes.npy`) flag invalid fields,
out-of-ROI, out-of-bounds, negative depth, and reprojection-error points with
pair-level counts. Fusion is explicit:

```yaml
fusion:
  enabled: false            # enable only after pairwise products are validated
  voxel_size: 1.0           # world units; deduplicates per voxel cell
  displacement_mad_factor: 5.0   # drop points above median + factor*MAD |dX|
  remove_rigid_body_motion: false
```

When enabled, `fused/` keeps the highest-confidence point per voxel cell with
source pair provenance (`source_pair`) and reports removed counts (reprojection
and displacement MAD).  Strain postprocessing belongs to the shared C++
`src/postprocess` route. The workflow never reads or writes
`result/mask/per_camera`, which stays owned by the NDeF ROI stage.

## Architecture rules

1. C++ is the primary scientific implementation language.
2. LibTorch owns the differentiable model-to-loss path.
3. Python performs configuration assembly, file I/O, binding calls, export, and
   visualization.
4. Calibration and reference-surface reconstruction happen before deformation
   problem construction.
5. One NDeF solve represents the selected surface as one continuous neural
   deformation field.
6. NumPy/OpenCV/Eigen round-trips are not allowed inside a differentiable path.

## Repository layout

```text
include/neurodic/       Public C++ interfaces
src/                    C++ and LibTorch implementations
bindings/python/        pybind11 bindings for neurodic._neurodic
python/neurodic/        Thin Python API, I/O, configuration, and plots
config/                 Example solver configurations
tests/cpp/              C++ numerical and invariant tests
tests/python/           Python import and binding tests
docs/                   Architecture, environment, and migration notes
case/                   Local examples and ignored generated results
```
