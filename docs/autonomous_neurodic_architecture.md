# NeuroDIC Agent-Ready Scientific Computing Architecture

Status: design baseline; Loop 4A additionally implements the fixed-evaluation
contract recorded in `docs/fixed_evaluation_contract.md`.

Loop 4B additionally records point-level stereo geometry, fixed-set NDeF
view-level observations, and preselection PIN Multi overlap evidence as
specified in `docs/multiview_observability_contract.md`.

Loop 5 adds read-only deterministic QualityReport interpretation under the
contract in `docs/diagnosis_contract.md`; it classifies failure families but
does not recommend or execute changes.

Loop 6 adds the native-free `neurodic.trial_plan/v1` dry-run contract in
`docs/trial_planning_contract.md`: immutable baseline identity, sparse typed
overrides, protected paths, explicit config ownership, canonical-DAG
invalidation, conservative artifact reuse, and would-execute action planning.
It does not create trials or execute stages.

Loop 7 adds `neurodic.execution/v1` guarded trial-workspace and atomic-publish
control primitives in `docs/trial_execution_contract.md`. Real solver adapters
remain deliberately unsupported until their output-redirection contracts are
verified; this loop does not widen solver execution capability by assumption.

Loop 8 adds `neurodic.comparison/v1` and explicit managed best references under
`docs/trial_comparison_contract.md`: compatible QualityReport comparison,
profiled directions/guardrails, eligibility, deterministic no-score selection,
and atomic explicit promotion. It does not recommend parameters or execute.

Loop 9 adds `neurodic.recommendation/v1` under
`docs/recommendation_contract.md`: a diagnosis-gated, registry-bounded sparse
override hypothesis which must survive Loop 6 planning. It neither executes a
trial nor searches parameters, and protected scientific identity is excluded.

Loop 10A adds a vendor-neutral canonical `skills/` skeleton. Skills supervise
the CLI/JSON/filesystem control plane only; they do not duplicate scientific
logic, invoke solvers, or embed an LLM agent in NeuroDIC.

Loop 10B makes the execution registry the native-free source of truth for
per-action runtime capability. Loop 10C validates the portable skill/CLI/JSON
workflow with generic shell-based tests. Loop 10D closes the documentation and
portability layer through [`docs/agent_compatibility.md`](agent_compatibility.md):
the canonical entry is `skills/neurodic/SKILL.md`, and the control plane remains
vendor-neutral. These loops do not imply complete scientific adapter coverage.

The delivered outer architecture is:

```text
Natural-language user
        |
        v
External coding agent (Supervisor)
        |
        v
Canonical NeuroDIC Skills
        |
        v
CLI + structured JSON
        |
        v
Deterministic control plane
        |
        v
Scientific core
```

Loops 1–10 are complete: 1, 2, 3, 4A, 4B, 5, 6, 7, 8, 9, and 10 are PASS.
Loop 10 comprises 10A canonical Skills, 10B compatibility hardening, 10C
workflow validation, and 10D documentation closure. Completion records the
control-plane contract, not complete guarded execution coverage for all
scientific solvers.

This document is based on the repository state at commit
`5e07c546fddc8035f79b1bda613590e9296c7bba` (2026-08-13). It distinguishes
observed behavior from proposed behavior. Paths and names under **Current** are
implemented today; paths and commands under **Target** are architectural
proposals.

## 1. Executive summary

NeuroDIC should become agent-ready by adding a deterministic control plane above
the existing Python workflows, not by putting an LLM, an agent protocol, or
workflow logic into the C++ solvers. The durable interoperability contract should
be:

```text
Markdown Skills + stable CLI + stable Python service API
                     + versioned JSON + filesystem state
                                      |
                                      v
                         existing Python workflows
                                      |
                                      v
                        pybind11 -> C++ / LibTorch
```

The first implementation priority is not automated tuning. It is a trustworthy,
read-only inventory and evaluation layer with versioned schemas. Without that
layer, a trial manager can preserve files but cannot safely decide whether a new
result is better.

Five changes to the initial concept are essential:

1. **Model workflows as solver-specific DAGs, not one universal linear
   pipeline.** PIN 2D has initialization and a single solve; Stereo and PIN Multi
   compose three planar fields and reconstruction; NDeF has ROI, surface,
   precalculation, deformation training/inference, and export. Surface and sparse
   precalculation are siblings feeding deformation, not necessarily one strict
   chain.
2. **Do not call current checkpoints resumable.** NDeF exports best and final
   model weights, but there is no load-state path, optimizer state, continuation
   cursor, or compatibility validation. They support comparison/inference only
   after a future loader is implemented.
3. **Keep immutable shared artifacts outside trials and address them by identity.**
   Calibration, images, ROI masks, and an accepted reference surface can be many
   gigabytes collectively. Trials should contain references, hashes, small
   metadata, and changed downstream products—not copied inputs.
4. **Make evaluation metric-first and threshold-profiled.** Current evidence is
   uneven: PIN Multi and NDeF have useful diagnostics; PIN 2D and Stereo lack loss
   histories and unified quality artifacts. A single quality score or automatic
   best promotion would conceal this asymmetry.
5. **Use a small hybrid Skill hierarchy.** A directory per parameter or failure
   type would duplicate causal knowledge. Use a project router, shared operating
   contracts, one workflow skill per solver family, a few NDeF stage skills, and
   one shared evaluation/diagnosis skill.

The minimum viable first phase is: versioned schemas, read-only `inspect`,
artifact identity, a normalized `quality.json` generated from evidence that
already exists, and dry-run stage planning. Trial execution, resume, promotion,
and recommendations follow only after those contracts pass golden-case tests.

## 2. Audit method and architectural constraints

The audit covered `README.md`, all current `docs/` and `config/` contracts, the
Python package and user-facing APIs, pybind11 bindings, public C++ headers,
solver/problem/result implementations, C++ and Python tests, and generated case
products under `case/`. Generated artifacts were inspected read-only.

The constraints are real and should remain:

- C++ is the scientific implementation layer; LibTorch owns model-to-loss
  differentiability.
- Python owns configuration composition, case discovery, file I/O, workflow
  assembly, export, and visualization.
- pybind11 exposes reviewed C++ contracts; it must not become another algorithm
  implementation.
- Preprocessing, initialization, calibration, reconstruction, fusion, and
  postprocessing may be non-differentiable. No NumPy/OpenCV/Eigen round trip may
  be introduced inside a differentiable path.
- The canonical environment is `/home/a306/miniconda3/envs/neurodic`; CUDA builds
  use its compiler through `CUDAHOSTCXX`. GPU visibility cannot be inferred from
  a sandboxed probe.
- Solver YAMLs are already mostly path-free. `load_case_config()` deep-merges a
  solver config with `config/case_paths.yaml`; the agent-ready layer should build
  on this separation rather than replacing it.
- Existing names such as `pin_multi_slover` are externally visible in APIs,
  manifests, tests, and old artifact directories. Correcting the spelling is a
  separate compatibility migration, not part of the control-plane MVP.

## 3. Current architecture audit

### 3.1 Boundary and call path

```text
config/*.yaml + config/case_paths.yaml + case files
                         |
                         v
python/neurodic/api/*.py and workflow helpers
  - discover ordered frames
  - load images/masks/calibration/surfaces
  - configure random state
  - create bound Problem objects
  - invoke bound Solver objects
  - save NPZ/JSON/PT/PNG
                         |
                         v
bindings/python/bind_*.cpp
                         |
                         v
include/neurodic + src
  Problem -> Solver -> Result/Diagnostics
```

The public Python functions return native result objects, not a stable run
record. File writing is controlled by `write_case_artifacts`, and most functions
write directly to their configured final directories. There is no atomic staging
directory, lifecycle status, invocation record, schema registry, or uniform
error envelope.

### 3.2 Existing reusable capabilities

| Capability | Current location | Reuse decision |
|---|---|---|
| YAML loading and recursive composition | `python/neurodic/config/__init__.py` | Reuse merge primitive; add strict override validation and canonical serialization above it. |
| Sorted planar/stereo/multiview frame discovery | `python/neurodic/case_io.py` | Reuse as the canonical inspection source; expose resolved frame identity. |
| Process-wide seed setup | `python/neurodic/runtime.py`, `include/src/neurodic/core/random.*` | Reuse; persist both runtime seed and stage-local seeds. |
| NDeF read-only readiness checks | `python/neurodic/ndef_preflight.py` | Generalize behind inspectors; preserve its non-mutating behavior. |
| NDeF isolated output roots/mapping | `python/neurodic/ndef_paths.py` | Reuse path resolution ideas; replace ad-hoc namespace use with run/trial paths over time. |
| PIN Multi pair selection and pair ROI diagnostics | `python/neurodic/pin_multi_roi.py` | Reuse outputs and skipped-pair semantics. |
| PIN Multi reason codes and quality summary | `python/neurodic/pin_multi_quality.py` | First adapter into unified quality reports. |
| PIN Multi fusion with source provenance | `python/neurodic/pin_multi_fusion.py` | Reuse scientific fusion and metrics; do not duplicate in control plane. |
| NDeF surface metadata/history | `python/neurodic/api/ndef_surface.py` | Adapt `surface_pretrain_meta.json`, `surface_dense_meta.json`, dense history, and surface dataset. |
| NDeF sparse scale/tracks | `python/neurodic/api/ndef_dic.py`, `initialization/ndef_precalculation.*` | Adapt track, NCC, reprojection, inlier, displacement, and per-camera evidence. |
| NDeF training history/checkpoints/summary | `python/neurodic/api/ndef_dic.py`, `src/solver/ndef_solver.cpp` | Adapt histories and weights; do not claim resumability. |
| Generic native diagnostics | `include/neurodic/core/result.hpp` | Reuse as raw evidence; its fields are too small to be the filesystem/CLI contract. |

### 3.3 What does not exist yet

- No installed console script or unified CLI is declared in `pyproject.toml`.
- No `Run`, `Trial`, `StageRecord`, artifact registry, best reference, or append-only
  event history exists.
- No uniform schema/version for inspection, execution, errors, evaluation, or
  comparison exists.
- No baseline/override/effective-config triplet, canonical config hash, allowed
  override registry, or reason/evidence record exists.
- No explicit dependency engine decides reuse or minimal reruns.
- No generic cache validation exists. Existing code commonly consumes a path if
  configured, but does not prove that its producer configuration and inputs
  match.
- No solver currently resumes optimization. NDeF saves weights only; optimizer
  state and training cursor are not saved or loaded. PIN saves no model state.
- No atomic output publication or overwrite guard exists. Re-running into the
  same configured root can replace files and mix generations.
- No uniform evaluator exists. PIN Multi is the strongest current quality
  implementation; NDeF has rich raw evidence; PIN 2D/Stereo mostly expose final
  loss and reconstruction validity.
- Native `SolverStatus::CONVERGED` currently means the configured loop finished,
  not that a scientific convergence rule passed. `ConvergenceMonitor` is an
  unfinished shell.
- Existing outputs do not consistently record code revision, effective config,
  input hashes, timestamps, device/runtime versions, or parent trial.

### 3.4 Documentation/config drift found during audit

- README examples and historical generated directories use `ndef` or
  `ndef_multi_slover`, while the current `case_paths.yaml` selects
  `result/ndef_multi` and `visualization/ndef_multi`.
- Tests enforce that solver configs do not embed `case` or `output` paths, so a
  bare call using only a solver YAML is not a complete case invocation. A stable
  interface must require or resolve the case-path mapping explicitly.
- README prose says PIN Multi fusion is disabled by default, but current
  `config/pin_multi.yaml` enables it; a runtime inspector must report the
  effective config, never infer behavior from prose.
- Some architecture prose describes a controlled NDeF topology while current
  Python assembly assigns `hidden_dim` and `hidden_layers`. Skills must be
  generated/reviewed against the effective public parameter registry, not copied
  from stale docs.

## 4. Solver and workflow inventory

### 4.1 PIN-DIC 2D

| Concern | Current fact |
|---|---|
| Python entry | `pin_dic()` and `run_planar_case()` in `python/neurodic/api/pin_dic.py` |
| Native entry | `PINSolver::solve(PINProblem)` in `src/solver/pin_solver.cpp` |
| Config | `config/pin_2d.yaml` plus `case_paths.yaml:pin_2d` |
| Inputs | First sorted image is reference, last sorted image is ROI, intermediate images are frames; or arrays supplied directly. |
| Preprocessing | Grayscale load, shape checks, mirror padding/B-spline precompute through `PINProblem`. |
| Initialization | `initialize_seeds`: SIFT search or integer search with optional SIFT prior, subpixel refinement, and MAD cleanup. |
| Optimization | Optional seed MSE for displacement components whose cleaned half-range exceeds the threshold, then Adam SSD/ZNSSD photometric optimization; optional sampled photometric centers. |
| Inference/postprocess | Evaluate displacement on ROI points; native 2D Green-Lagrange strain. |
| Outputs | `pin_result.npz` with coordinates, displacement, strain, iterations, final loss; displacement PNG. |
| Intermediate/reuse | Seed JSON/plots can be produced by seed utilities, but `pin_dic()` does not consume a validated persisted seed artifact by identity. An explicit `seeds` argument is reusable in-process. |
| Expensive/stochastic | Seed search and both training phases; model initialization and sampled centers depend on configured runtime state. |
| Diagnostics | Final loss, iteration count, seed count/scales, active seed components, sampling flag, ROI bounding box. No loss history or residual field is exported. |
| Tests | C++ photometric/solver invariants; Python case ordering, imports, bindings, seed/case fixtures. No unified end-to-end quality contract. |

PIN 2D cannot yet support reliable automatic learning-rate tuning because only a
final loss is persisted. Its first evaluation work should add loss history,
finite-value/coverage checks, field robust gradients, and a held-out or fixed
evaluation residual—not invent thresholds from one final training value.

### 4.2 PIN-DIC Stereo

| Concern | Current fact |
|---|---|
| Python entry | `pin_stereo_dic()` and `run_stereo_case()` in `python/neurodic/api/pin_stereo_dic.py` |
| Native entry | `PINStereoSolver::solve(PINStereoProblem)`; internally calls `PINSolver` three times. |
| Config | `config/pin_stereo.yaml` plus `case_paths.yaml:pin_stereo` |
| Inputs | Sorted left/right image series paired by index, L0 ROI, stereo camera-pair JSON, selected frame. |
| Initialization/optimization | Three independent planar problems: L0→R0, L0→Lk, L0→Rk. Each repeats PIN initialization and optimization. |
| Reconstruction | DLT/stereo geometry reconstructs reference and current points, applies depth/bounds/reprojection validity, computes `Xk-X0`. |
| Postprocess | Optional traditional weighted local least-squares 3D strain. |
| Outputs | Three planar NPZs; reference/current reconstruction NPZs; deformation/strain NPZ and summary; plots. |
| Reuse/resume | No stage-level persisted reuse contract and no resume. Reference disparity is conceptually reusable across frames but `run_stereo_case()` currently recomputes it. |
| Diagnostics | Per-field final loss/iterations, validity mask, reference/current reprojection errors, point counts. No reason codes and no loss history. |
| Tests | C++ solver/reconstruction invariants and Python case ordering; substantially less filesystem quality coverage than PIN Multi. |

The first safe optimization for repeated frames is explicit reuse of a validated
L0→R0 artifact, but only after its inputs and solver config are fingerprinted.

### 4.3 PINMultiSolver / pairwise multiview

| Concern | Current fact |
|---|---|
| Python entry | `pin_multi_slover_dic()` and `run_pin_multi_case()` in `python/neurodic/api/pin_multi_slover_dic.py` |
| Native entry | `PINMultiSolver::solve(PINMultiProblem)`; each pair is a `PINStereoProblem`. |
| Config | `config/pin_multi.yaml`, shared planar config via `pin_2d_config`, plus `case_paths.yaml:pin_multi` |
| Inputs | Multiview synchronized series, scaled calibration, selected/derived camera pairs. |
| Pair ROI | Reference-time SIFT, ratio/mutual/geometric filtering, convex hull or alpha support; failed pairs are structured `skipped`. |
| Solve/reconstruct | For each pair A/B: A0→B0, A0→Ak, A0→Bk, then independent X0/Xk/dX reconstruction. |
| Quality | Per-point prioritized reason codes: invalid field, outside ROI, bounds, negative depth, reprojection; per-pair counts, ratios, reprojection stats, final PIN losses. |
| Fusion | Optional reprojection/displacement filtering, voxel selection, surface cleanup, source-pair provenance, mesh and traditional strain. |
| Outputs | Pair ROI artifacts, pair fields/reconstructions/deformation/quality, optional fused products, and `manifest.json`. |
| Reuse/resume | Pair ROIs and pair products are materialized, but current full entry does not validate/cache-hit and skip them. No optimizer resume. |
| Expensive/stochastic | SIFT/ROI, three PIN solves per pair, all pairs/frames, and fusion over millions of points. Runtime seed is set, but artifact provenance does not fully record it. |
| Tests | Strongest workflow tests: selection, ROI isolation, output contract, reason codes, fusion, source provenance, cleanup, and synthetic displacement regression. |

PIN Multi is the best initial template for artifact manifests and evaluation, but
its route-specific manifest must be adapted rather than promoted unchanged to a
universal run schema.

### 4.4 NDeF multiview

NDeF is a DAG with distinct products:

```text
calibration + synchronized reference images
                |
                +--> ROI masks -----------------------+
                |                                     |
                +--> sparse SfM observations          |
                         |                             |
                         v                             |
                 surface pretrain/dense/fusion        |
                         | accepted surface dataset    |
                         +--------------+--------------+
                                        |
                      +-----------------+------------------+
                      |                                    |
                      v                                    v
            sparse precalculation                 deformation solve
                      | displacement scale                 ^
                      +------------------------------------+
                                                            |
                                      inference/export/evaluation
```

| Stage | Current implementation and products |
|---|---|
| Preflight/ROI | `inspect_ndef_preflight()` is read-only. `generate_ndef_roi()` creates per-camera masks and `mask_meta.json` from shared sparse observations/topology. |
| Reference surface | `pretrain_ndef_surface()` assembles `NDeFSurfaceProblem`; C++ sparse depth training and optional dense photometric refinement produce pretrain/dense NPZs and metadata. Python fuses dense charts by visibility/depth consistency into `deformation_surface_dataset.npz`. |
| Sparse precalculation | `ndef_sparse_precalculation()` calls C++ batched NCC and DLT. It saves source camera/UV, reference/current points, displacement, camera counts, reprojection, mean match score, MAD inliers, scale statistics, camera counts, and seed semantics. |
| Deformation | `ndef_dic()` assembles surface observations and scale, then `NDeFSolver` samples points with replacement, runs AdamW photometric/smoothness training, selects the minimum training-loss weights, infers the field and autograd strain, and exports results. |
| Diagnostics | Eight-column history (loss components, valid/supervised pairs, displacement RMS), per-point sample counts, projection validity, coordinate normalization, output scale, displacement distribution, native metrics, best/final weights. |
| Visualization | Reference/current XYZ surfaces, displacement components/magnitude, loss history, and valid observations. |
| Reuse/resume | Surface NPZ and sparse tracks can be manually reused through configured paths. Best/final weights are saved. There is no compatibility check, cache record, checkpoint load, optimizer state, or resume API. Surface model weights are not exported for continuation. |
| Randomness | Runtime seed, surface dense seed, surface fusion seed (default internal value), sparse random seed, and deformation seed are distinct. Surface sampling/fusion and Torch sampling are stochastic but seeded. Auto-batch may vary with available hardware memory. |
| Tests | C++ NDeF geometry/solver/precalculation invariants and Python path/ROI tests. Existing local case outputs provide rich evidence, but a compact fully automated end-to-end golden NDeF case is still needed. |

Important cost facts: dense surface prediction and fusion can materialize millions
of samples; deformation may draw tens of millions of point samples; full-field
NPZs, checkpoints, and plots can multiply storage rapidly. Trials must never copy
accepted surface/calibration/image products by default.

## 5. Automation blockers and design consequences

| Blocker | Consequence |
|---|---|
| Direct writes to final roots | Execute in a unique trial directory and publish a completed stage only after validation. Never point a trial at baseline output roots. |
| Paths are convention-based, not registered | Inspection must create an artifact inventory with producer stage, type, schema, identity, and compatibility. |
| Partial diagnostics | Report `not_available` explicitly. Do not interpret missing history as convergence. |
| No checkpoint loader | CLI must reject `--resume` for unsupported stages with a structured capability error. Reuse and resume are different operations. |
| Configuration accepts loose mappings | Define a public parameter registry with type/range/stage/causal metadata. Unknown override paths fail closed. |
| Stage functions mix solve and export | Add orchestration wrappers first; later separate preparation, native execution, and export only where needed for atomicity/testability. |
| Multiple random seeds | Resolve and persist every stage-local seed; compare trials using the same seed policy. |
| Training minimum is selected on training batches | It is not an independent validation metric. Best-trial selection needs fixed evaluation evidence. |
| Ground truth exists in synthetic/local cases | Never make ground truth a production requirement or silently use it in tuning. Mark it `evaluation_only` and record its use. |

## 6. Target agent-ready architecture

### 6.1 Layers

```text
External coding agent (replaceable, stateless)
  reads Skills; invokes CLI/Python; parses JSON
                         |
Agent-ready control plane (Python, deterministic)
  schemas | inspection | evaluation | diagnosis | recommendation
  state   | artifacts  | trials     | stage planner | execution guard
                         |
Workflow adapters (Python)
  PIN 2D | Stereo | PIN Multi | NDeF
                         |
Existing Python APIs -> pybind11 -> C++/LibTorch
```

The control plane contains no prompt execution and depends on no model vendor.
It converts domain evidence into typed findings and bounded actions. External
agents decide when to call it; NeuroDIC owns persistent computation state.

### 6.2 Repository responsibility boundary

In scope:

- local inspection, planning, execution, evaluation, diagnosis, recommendation,
  trial and best management;
- versioned local schemas and stable CLI/Python interfaces;
- scientific expert knowledge in Skills and deterministic rule tables.

Out of scope:

- messaging, web UI/server, accounts, queues, remote task distribution;
- OpenAI/Anthropic/Codex-specific SDKs or assumptions;
- an internal LLM supervisor/critic/optimizer;
- MCP in phase one. A future MCP adapter may call the same stable Python API.

## 7. Recommended repository layout

```text
NeuroDIC/
├── skills/
│   ├── neurodic/SKILL.md                 # router and global operating contract
│   ├── common/
│   │   ├── case-and-artifacts/SKILL.md
│   │   ├── trials-and-reproducibility/SKILL.md
│   │   └── evaluation-and-diagnosis/SKILL.md
│   ├── pin/SKILL.md                      # planar workflow + causal parameters
│   ├── pin-stereo/SKILL.md
│   ├── pin-multi/SKILL.md
│   └── ndef/
│       ├── SKILL.md                      # NDeF workflow router
│       ├── surface/SKILL.md
│       ├── precalculation/SKILL.md
│       └── deformation/SKILL.md
├── python/neurodic/
│   ├── agent/                            # proposed control plane; no numerical kernels
│   │   ├── schemas.py
│   │   ├── errors.py
│   │   ├── inspect.py
│   │   ├── artifacts.py
│   │   ├── state.py
│   │   ├── config.py
│   │   ├── stages.py
│   │   ├── evaluate.py
│   │   ├── diagnose.py
│   │   ├── recommend.py
│   │   ├── compare.py
│   │   ├── trials.py
│   │   └── adapters/
│   │       ├── pin.py
│   │       ├── pin_stereo.py
│   │       ├── pin_multi.py
│   │       └── ndef.py
│   └── cli.py
├── schemas/agent/v1/                     # JSON Schema documents, versioned
│   ├── envelope.schema.json
│   ├── inspection.schema.json
│   ├── run.schema.json
│   ├── stage.schema.json
│   ├── artifact.schema.json
│   ├── quality.schema.json
│   ├── diagnosis.schema.json
│   └── recommendation.schema.json
├── config/
│   ├── *.yaml                            # immutable reviewed baselines
│   ├── case_paths.yaml
│   └── quality_profiles/                 # reviewed dataset/method profiles
│       ├── default.yaml
│       └── synthetic.yaml
├── tests/
│   ├── python/agent/
│   ├── cli/
│   ├── golden/                           # compact known-good/bad metadata fixtures
│   └── existing tests unchanged
└── docs/autonomous_neurodic_architecture.md
```

Do not create a Skill per CLI command, parameter, or failure code. Those cuts
would fragment causal explanations and invite contradictions. The workflow
Skill owns solver-specific causality; the shared evaluation Skill owns report
semantics and operating safety.

## 8. Skill architecture and contract

### 8.1 Router behavior

`skills/neurodic/SKILL.md` should be short and operational:

1. Inspect before execution.
2. Identify case, solver family, frame, effective config, and available artifacts.
3. Route planar, stereo, pairwise multiview, or continuous multiview work to the
   corresponding Skill.
4. For poor results, route to evaluation/diagnosis before recommendation.
5. Never mutate baseline YAML or promote newest-as-best.
6. Use dry-run before a mutating command; require a unique trial for changed
   parameters.
7. Treat missing evidence and unsupported resume as explicit states.

It should not attempt to infer a solver solely from image count. Calibration,
requested output (2D field, pairwise 3D, or continuous surface field), and
explicit config are stronger evidence. Ambiguous cases should return candidates
and missing information rather than silently choose a costly workflow.

### 8.2 Uniform Skill contract

Every workflow/stage Skill should contain these reviewed sections:

```text
Identity                 name, contract version, compatible NeuroDIC versions
Purpose and scope        scientific goal and ownership boundary
Use / do-not-use         positive and negative routing criteria
Required inputs          type, units, coordinate frame, identity requirements
Preconditions            readiness checks and required upstream artifacts
Stage model              stage IDs, dependencies, produced artifact types
Operations               exact stable CLI/Python operations
Configuration            public paths only; defaults and allowed ranges
Causal parameter rules   evidence -> change -> expected effect -> cost/risk
Diagnostics              metrics, availability, interpretation limitations
Failure modes            codes, required evidence, competing explanations
Rerun semantics          invalidation rules and reusable artifacts
Determinism              all seeds and hardware-sensitive behavior
Cost class               relative GPU/runtime/memory/disk cost
Unsafe actions           forbidden combinations and overwrite rules
Stop/escalate rules      pass, budget, uncertainty, invalid input, human review
Examples                 machine commands plus representative JSON, kept small
Evidence sources         exact code/config/tests on which knowledge is based
```

Parameter descriptions must be causal and implementation-specific. For example:

| Parameter | Evidence-gated causal rule |
|---|---|
| `precalculation.sparse.temporal_search_radius` ↑ | Consider only when temporal matches are rejected/absent and observed displacement approaches the current radius while cross-view evidence remains healthy. It expands NCC candidates and cost and raises false-match risk; rerun sparse precalculation and deformation because output scale changes. |
| `cross_search_radius` ↑ | Consider when projected cross-camera centers are credible but cross matches fail near the search boundary. It cannot repair calibration/visibility errors and increases matching cost/ambiguity. |
| `cross_ncc_threshold` ↓ | May recover tracks when score distribution clusters just below the threshold, but increases false matches. Require reprojection and multi-camera validation; change one threshold at a time. |
| `precalculation.statistic` | Chooses output normalization from already filtered displacement magnitudes; changing it reruns deformation only. `max` is high-risk because one surviving tail value expands output scale. Prefer robust median/p75 only when scale diagnostics justify it. |
| `deformation_training.photometric_learning_rate` ↓ | Applicable to finite but oscillatory/diverging history at stable valid-pair counts. It increases time to progress; it cannot repair invalid projections, bad scale, or insufficient texture. Rerun deformation. |
| `patch_radius` ↑ | In NDeF it changes the camera-local photometric support and boundary validity. It may stabilize textured local matching but reduces valid patches near masks/images and increases memory/cost. Re-evaluate valid/supervised ratios; rerun deformation. |
| `min_valid_patch_ratio` ↓ | Admits partial patches and can increase supervision near boundaries; it also changes loss comparability and can introduce boundary bias. Use only when missing validity is demonstrably boundary-driven. |
| `smoothness_weight` ↑ | Penalizes normalized-coordinate Jacobian energy in the actual NDeF objective. It may suppress local spikes but can erase legitimate gradients; never use as the first response to calibration, scale, or matching failures. |
| `fusion.voxel_size` ↑ | Reduces PIN Multi output density/storage and merges nearby pair products; it can erase small spatial features and changes strain neighborhoods. It reruns fusion/postprocess only. |

Network width/layer count, calibration scale, coordinate conventions, topology,
and rigid-body removal are not ordinary automatic tuning knobs. They require
strong evidence or explicit human authorization because they change model
capacity or scientific interpretation.

## 9. Stable Python API and CLI

### 9.1 Python service API

Keep existing scientific functions for compatibility. Add a stable orchestration
API returning serializable records:

```python
inspect_case(request) -> InspectionReport
inspect_artifact(request) -> ArtifactReport
plan_run(request) -> RunPlan                 # read-only
evaluate_result(request) -> QualityReport    # read-only except explicit output path
diagnose_result(request) -> DiagnosisReport  # deterministic rules
recommend_changes(request) -> RecommendationReport
create_trial(request) -> TrialRecord
run_stage(request) -> StageRecord
compare_trials(request) -> ComparisonReport
promote_best(request) -> BestRecord
```

Adapters call existing `pin_dic`, `pin_stereo_dic`, `pin_multi_slover_dic`,
`pretrain_ndef_surface`, `ndef_sparse_precalculation`, and `ndef_dic`. CLI code
must contain no scientific logic.

### 9.2 CLI command surface

Recommended phase-one grammar:

```text
neurodic inspect case|config|pipeline|artifact|result ...
neurodic evaluate --run RUN --scope ...
neurodic diagnose --quality QUALITY_JSON
neurodic recommend --diagnosis DIAGNOSIS_JSON --budget BUDGET_YAML
neurodic plan --case CASE --solver SOLVER --stage STAGE [--trial TRIAL]
neurodic trial create --run RUN --override OVERRIDE_YAML ...
neurodic run --trial TRIAL --stage STAGE [--resume] [--dry-run]
neurodic compare --run RUN --left TRIAL --right TRIAL
neurodic best show|promote --run RUN ...
```

Phase one implements only inspect, evaluate, and plan/dry-run. Later commands are
reserved now so schema and state design do not collide with an ad-hoc CLI.

Every command accepts `--format json|text`; automation should use JSON. JSON goes
to stdout, diagnostics/logs to stderr. A stable envelope is mandatory:

```json
{
  "schema_version": "neurodic.agent/v1",
  "status": "ok",
  "operation": "inspect.case",
  "request_id": "...",
  "data": {},
  "warnings": [],
  "errors": []
}
```

Errors are structured with `code`, `message`, `stage`, `path`, `details`, and
`recoverable`; stack traces are opt-in debug output, never the contract.

Suggested exit codes:

| Code | Meaning |
|---:|---|
| 0 | Operation completed; quality may still contain warnings. |
| 2 | CLI usage/argument error. |
| 3 | Invalid config/schema/override. |
| 4 | Missing or incompatible input/artifact. |
| 5 | Unsafe state conflict/overwrite refused. |
| 6 | Numerical stage failed. |
| 7 | Resource/runtime failure (OOM, unavailable device). |
| 8 | Evaluation failed or evidence corrupt. |
| 9 | Unsupported capability, including unsupported resume. |

`--dry-run` must resolve effective config, inputs, dependencies, cache decisions,
output paths, invalidations, seed policy, estimated cost class, and overwrite
conflicts without creating a trial or touching artifacts.

### 9.3 Inspection / recommendation / execution separation

- **Inspection** is strictly read-only, including artifact parsing. It never
  imports a GPU solver merely to inspect a case.
- **Recommendation** reads inspection/quality/diagnosis and emits bounded proposed
  overrides plus rerun plans. It does not create a trial or execute.
- **Execution** is the only layer that creates state, runs solvers, publishes
  artifacts, or promotes best. Mutating APIs require explicit run/trial/stage.

This separation should be enforced in module dependencies: `inspect.py`,
`evaluate.py`, and `diagnose.py` must not import execution adapters.

## 10. Case, run, trial, stage, artifact, and best model

Definitions:

- **Case**: immutable scientific input namespace and case-path mapping: images,
  calibration, masks or ROI sources, frames, units, and coordinate conventions.
- **Run**: one solver family and evaluation policy applied to a defined case scope
  (for example NDeF frame 20). A run owns baseline and trials.
- **Trial**: one immutable effective configuration, parent, seed policy, and set
  of stage attempts. Baseline is a protected trial, not a mutable special folder.
- **Stage**: smallest executable/reusable DAG node with declared inputs and
  outputs. An attempt has pending/running/completed/failed/interrupted status.
- **Artifact**: immutable published output with type, schema, producer, identity,
  location, size, and compatibility metadata.
- **Best**: an atomic reference to an eligible evaluated trial plus selection
  rationale. It is never inferred from recency.

Target case-local state:

```text
<case>/result/.neurodic/
├── runs/<run_id>/
│   ├── run.json
│   ├── baseline -> trials/baseline
│   ├── best.json
│   ├── events.jsonl
│   └── trials/
│       ├── baseline/
│       │   ├── trial.json
│       │   ├── override.yaml
│       │   ├── effective.yaml
│       │   ├── effective.sha256
│       │   ├── artifacts.json
│       │   ├── quality.json
│       │   └── stages/<stage_id>/stage.json
│       └── trial_0001/...
└── locks/
```

Large numerical outputs may remain in existing solver-specific locations or a
trial artifact directory. `artifacts.json` is the authority and can point to
shared immutable products. Do not use filesystem symlinks as the only identity
or portability mechanism; JSON references are primary, symlinks optional.

State transitions are append-only events plus atomically replaced current
records. Use temporary files/directories on the same filesystem and rename only
after validation. A stale `running` stage is `interrupted`, never silently
`failed` or `completed`.

## 11. Configuration and provenance

### 11.1 Baseline + override + effective config

For every trial persist:

```text
reviewed solver baseline
+ reviewed case-path mapping
+ sparse trial override
= canonical effective configuration
```

The merge is recursive for mappings and replace-only for sequences/scalars.
Overrides use full dotted paths internally. Unknown paths, type changes, and
protected paths fail closed. Protected by default: `case.root`, raw input paths,
calibration scale/frame conventions, output roots, and solver/mode. Scope changes
such as frame selection belong to the run request, not a tuning override.

Canonical hash input includes normalized effective config, solver adapter/schema
versions, selected frame/cameras/pairs, and relevant code revision. Secrets are
not expected in current configs; if introduced later, redact values from records
and hash through a separate secure policy.

Each change records:

```json
{
  "path": "deformation_training.photometric_learning_rate",
  "old": 0.003,
  "new": 0.001,
  "reason_code": "TRAINING_OSCILLATION",
  "evidence_refs": ["quality.json#/findings/2"],
  "expected_effect": "reduce update instability",
  "risks": ["slower convergence"],
  "invalidates_from": ["ndef.deformation.train"]
}
```

### 11.2 Provenance minimum

Record run/trial IDs, parent, timestamps, argv/API request, user/agent instruction
as plain text, baseline/override/effective hashes, Git commit and dirty-tree
digest/status, package/native/Torch/CUDA versions, host/device class, every seed,
input artifact identities, stage attempts, outputs, evaluation profile, and best
decision. Do not hash multi-gigabyte inputs synchronously on every inspection:
cache a content digest keyed by canonical path, size, mtime, and digest algorithm,
and allow a fast metadata identity explicitly marked weaker than content identity.

## 12. Artifact reuse, cache, and resume

These terms must not be conflated:

- **Reuse**: consume a completed immutable upstream artifact produced by another
  trial/run after compatibility checks.
- **Cache hit**: an artifact with the exact producer signature already exists, so
  execution can be skipped.
- **Resume**: continue an interrupted stage from saved algorithm state.

An artifact signature includes artifact type/schema, producer stage/version,
relevant effective-config projection, input identities, scope, seed policy, and
code compatibility. Reuse is refused when units, coordinate frame, camera order,
frame, surface point ordering, or config projection disagree.

Current safe reuse candidates after metadata is added:

- calibration and camera topology across solver runs;
- ROI masks when their image/calibration/topology/options identities match;
- NDeF accepted reference surface across deformation trials and frames sharing
  the same reference state;
- NDeF sparse precalculation only for the same current frame and its matching
  inputs/options;
- Stereo/PIN Multi reference disparity and pair ROI across frames, once persisted
  signatures exist;
- PIN Multi pair products as inputs to fusion-only trials.

Current NDeF `.pt` files are not resume artifacts. True deformation resume needs
model state, optimizer state, completed epoch/step, RNG states, sample-count
state, best-state/loss, normalization and output scale, exact surface ordering,
and compatibility validation. Surface resume additionally needs its model and
optimizer state. Until implemented, `resume_supported=false` is returned.

Retention policy should keep baseline, current best, trial metadata/quality, and
explicitly pinned trials. Non-best heavy artifacts may be eligible for a separate
user-authorized garbage-collection command; never delete them implicitly during
promotion.

## 13. Stage dependency graphs and minimal rerun

### 13.1 Canonical stage IDs

```text
PIN 2D:
  pin.inputs -> pin.initialization -> pin.train -> pin.infer -> pin.postprocess -> pin.evaluate

Stereo:
  stereo.inputs -> stereo.initialization.{ref_disp,temporal,def_disp}
  -> stereo.train.{ref_disp,temporal,def_disp}
  -> stereo.reconstruct -> stereo.postprocess -> stereo.evaluate

PIN Multi:
  pin_multi.inputs -> pin_multi.pair_select -> pin_multi.pair_roi
  -> pin_multi.pair_solve -> pin_multi.pair_quality
  -> [pin_multi.fusion -> pin_multi.postprocess] -> pin_multi.evaluate

NDeF:
  ndef.inputs -> ndef.roi
  ndef.inputs + ndef.roi -> ndef.surface.sparse_train
  -> ndef.surface.dense_train -> ndef.surface.fuse
  ndef.inputs + ndef.roi + ndef.surface.fuse -> ndef.precalculation
  ndef.inputs + ndef.roi + ndef.surface.fuse + ndef.precalculation
  -> ndef.deformation.train -> ndef.deformation.infer
  -> ndef.postprocess -> ndef.evaluate
```

Initially, adapters may execute several internal nodes as one atomic capability
because existing functions combine them. The planner must report this honestly
(`execution_granularity`) rather than pretending fine-grained reruns exist.

### 13.2 Invalidation examples

| Change | Minimum affected stages |
|---|---|
| NDeF deformation learning rate/batch/epochs/smoothness/patch loss | deformation train onward |
| NDeF scale statistic with unchanged sparse tracks | deformation train onward |
| NDeF sparse search/NCC/MAD/seed | precalculation and deformation onward |
| NDeF surface dense/fusion parameter | surface producer and all consumers; ROI/calibration reusable if unchanged |
| NDeF ROI generation | ROI, surface, precalculation, deformation |
| Calibration/camera order/scale | all geometry-dependent stages; no surface or reconstruction reuse by default |
| PIN training parameter | affected planar train/infer/postprocess/evaluation; initialization reusable only if its config and inputs match |
| Stereo temporal-only evidence | ideally temporal field then reconstruction onward; current API may require the complete combined solve until split safely |
| PIN Multi pair ROI parameter | pair ROI and every downstream pair using it |
| PIN Multi fusion voxel/filter/strain parameter | fusion/postprocess/evaluation only |

The stage planner computes config projections per stage and compares input
signatures. It must explain every reuse and invalidation in dry-run JSON.

## 14. Unified evaluation architecture

### 14.1 Report schema

`quality.json` is a normalized view, not a replacement for raw artifacts:

```json
{
  "schema_version": "neurodic.quality/v1",
  "solver": "ndef",
  "scope": {"frame": 20, "stage": "deformation"},
  "status": "warning",
  "profile": {"id": "default", "version": 1},
  "metrics": [
    {
      "id": "training.valid_pair_ratio.final",
      "value": 0.71,
      "unit": "ratio",
      "availability": "observed",
      "source": "diagnostics/training.npz#/history"
    }
  ],
  "threshold_results": [
    {"metric_id": "training.valid_pair_ratio.final", "operator": ">=", "threshold": 0.8, "passed": false}
  ],
  "findings": [
    {
      "code": "LOW_VALID_OBSERVATION",
      "severity": "warning",
      "stage": "ndef.deformation.train",
      "evidence_refs": ["#/metrics/0"]
    }
  ],
  "failure_stage": "ndef.deformation.train",
  "eligibility": {"best_candidate": false, "reasons": ["required threshold failed"]}
}
```

Metric, threshold, finding, severity, failure stage, and recommendation remain
separate. Thresholds are versioned profiles, not hard-coded universal truths.
`unknown` is a legitimate overall status when required evidence is missing.

No primary scalar score is defined in v1. Best selection uses gates and a
lexicographic/Pareto policy. A future presentation score may be derived, never
used as the sole scientific decision.

### 14.2 NDeF evaluation from current evidence

**Reference surface (available now):** sparse observation retention and rejection
reasons, calibration reprojection mean/median/p95, sparse depth RMSE, dense loss
history, dense supported samples, sparse-anchor RMSE, chart/candidate/fused point
counts, per-point visibility counts, depth absolute error, camera/source coverage,
and surface neighbor/normal statistics derivable from NPZ. Ground-truth distance
is allowed only in tagged synthetic/benchmark profiles.

**Missing or weak:** no production truth for absolute surface accuracy; chart
coverage relative to expected physical object area is unknown; current fusion
metadata is not a complete per-point rejection trace. Therefore “surface good”
must be conditional on geometric/self-consistency gates, not absolute accuracy.

**Sparse precalculation (available now):** tracks/requested seeds per camera,
track/inlier ratios, mean match score distribution, camera count, reference and
current reprojection distributions, displacement magnitude distribution and MAD
inliers, selected scale statistic, seed, and camera balance.

**Deformation training (available now):** finite/NaN status, initial/best/last and
windowed loss trends, oscillation/plateau indicators, photometric/smoothness
components, valid-to-supervised pair ratio through time, displacement RMS through
time, point sample coverage, batch/steps/epochs, seed, best/final checkpoints, and
reference/current valid observations.

Training loss is evaluated on sampled training patches and the exported best
state is the minimum of those batches. Comparisons require identical evaluation
sampling or a future fixed holdout residual pass.

**Deformation field (available/derivable):** finite values, point/view coverage,
magnitude/component robust distributions, local kNN displacement jumps,
boundary-vs-interior statistics using visibility/masks, strain finite/outlier
statistics, current projection bounds/depth, and reference-to-current valid-view
loss. Spatial continuity is evidence; it cannot by itself distinguish genuine
local deformation from an artifact.

**New scientific metrics required before aggressive automation:** fixed evaluation
patch residuals independent of training draws, multi-view photometric residual
maps, current-state cross-view consistency, uncertainty/repeatability across
seeds where material, and domain-approved thresholds for surface/field spatial
metrics.

### 14.3 PIN evaluation

**PIN 2D current evidence:** seed counts/scales, enabled pretrain components,
final training loss, iterations, ROI geometry, finite displacement/strain, field
coverage, and derivable robust local gradients/outliers. Missing: loss history,
per-pixel residual, fixed evaluation set, seed confidence export, and reason codes.
Status should remain `unknown` for convergence quality when those are required.

**Stereo current evidence:** the three PIN evidence sets plus valid ratio,
reference/current reprojection distributions, positive-depth/bounds validity, 3D
finite field/strain, and reconstructed surface continuity. Add prioritized reason
codes by adapting PIN Multi before recommendations are enabled.

**PIN Multi current evidence:** pair ROI match funnel and mask fraction, skipped
pairs, per-field final losses, per-point reason codes, valid/reprojection ratios,
pair coverage/balance, fusion rejection counts, deduplication, source-pair
provenance, surface cleanup, strain validity, and optional benchmark truth. This
can support the first production-quality unified adapter.

Cross-pair disagreement is not fully measured after voxel selection because one
highest-confidence point is retained. A pre-selection overlap-consistency metric
should be added before using fusion agreement to tune pair solvers.

## 15. Failure taxonomy and deterministic diagnosis

Use hierarchical stable codes, with solver-specific details in evidence:

```text
INPUT.MISSING | INPUT.ORDER_MISMATCH | INPUT.SHAPE_MISMATCH
CONFIG.INVALID | CONFIG.UNKNOWN_OVERRIDE | CONFIG.INCOMPATIBLE
RUNTIME.DEVICE_UNAVAILABLE | RUNTIME.OUT_OF_MEMORY | RUNTIME.NONDETERMINISTIC
CALIBRATION.INVALID | CALIBRATION.HIGH_REPROJECTION | CALIBRATION.SCALE_AMBIGUOUS
ROI.MISSING | ROI.LOW_COVERAGE | ROI.LOW_TEXTURE | ROI.PAIR_MATCH_FAILURE
INITIALIZATION.INSUFFICIENT_SEEDS | INITIALIZATION.OUTLIERS | INITIALIZATION.SCALE_DEGENERATE
SURFACE.SPARSE_SUPPORT_LOW | SURFACE.DENSE_DIVERGENCE | SURFACE.VISIBILITY_LOW
SURFACE.DEPTH_INCONSISTENT | SURFACE.SPATIAL_OUTLIER
PRECALC.TRACK_RATIO_LOW | PRECALC.MATCH_SCORE_LOW | PRECALC.REPROJECTION_HIGH
PRECALC.CAMERA_IMBALANCE | PRECALC.SCALE_UNSTABLE
TRAINING.NONFINITE | TRAINING.OSCILLATION | TRAINING.PLATEAU
TRAINING.VALID_OBSERVATION_LOW | TRAINING.SAMPLE_COVERAGE_LOW
FIELD.NONFINITE | FIELD.LOCAL_SPIKE | FIELD.BOUNDARY_ARTIFACT
FIELD.REPROJECTION_HIGH | FIELD.OVER_SMOOTHING_SUSPECTED | FIELD.UNDERFIT_SUSPECTED
RECONSTRUCTION.NEGATIVE_DEPTH | RECONSTRUCTION.OUT_OF_BOUNDS
FUSION.PAIR_DISAGREEMENT | FUSION.SOURCE_IMBALANCE | FUSION.OUTLIER_EXCESS
ARTIFACT.MISSING | ARTIFACT.CORRUPT | ARTIFACT.INCOMPATIBLE
STATE.CONFLICT | STATE.INCOMPLETE | CAPABILITY.RESUME_UNSUPPORTED
```

“Suspected” codes must remain non-decisive until competing causes are excluded.
Do not emit fabricated confidence decimals. Prefer `support: strong|moderate|weak`
based on explicit rule prerequisites. Every diagnosis contains:

- failure stage and upstream stages checked;
- observed metrics and thresholds;
- candidate cause, supporting and contradicting evidence;
- missing evidence that prevents discrimination;
- safe next observation or minimum experiment.

Example rule: temporal search radius is a candidate cause only if temporal match
failures/scores are specifically weak, geometry and cross-camera matching pass,
and the displacement/search-boundary relation supports it. Low total tracks alone
is insufficient; ROI texture, calibration, cross search, and thresholds compete.

## 16. Parameter recommendation and bounded optimization

Recommendations are typed proposals, not free-form `set_parameter` calls. The
parameter registry stores owner stage, type/range, coupling group, automatic
eligibility, invalidations, cost, and causal rules.

Rules:

1. Evaluate, diagnose, then recommend.
2. Change the minimum number of parameters that tests one diagnosis; default
   maximum is one, hard maximum two only for declared coupled parameters.
3. Preserve baseline and previous best.
4. Use the same evaluation profile and fair seed policy. One stochastic run
   should not displace best on a marginal sampled-loss improvement.
5. Prefer observation-only actions when diagnosis support is weak.
6. Never auto-change calibration scale, frame/camera mapping, coordinate units,
   ground-truth alignment, rigid-body interpretation, network topology, or
   failure thresholds.

Budget contract:

```yaml
max_trials: 4
max_failed_trials: 2
max_runtime_seconds: 14400
max_parameter_changes_per_trial: 1
stop_after_no_improvement: 2
minimum_improvement:
  metric: evaluation.photometric_residual.median
  relative: 0.02
allowed_parameter_groups: [deformation_training]
max_disk_bytes: 20000000000
seed_policy: paired
```

Stop on quality pass, budget exhaustion, repeated no meaningful improvement,
invalid required input, resource exhaustion, conflicting metrics without a
declared preference, diagnosis uncertainty, or a human-only parameter need.

## 17. Best-result selection

Promotion is an explicit atomic operation. Eligibility gates first reject:

- failed/incomplete stages or incompatible artifacts;
- missing required metrics;
- nonfinite fields/losses;
- required validity/reprojection/coverage threshold failures;
- evaluation-profile or scope mismatch;
- unfair stochastic comparison.

Eligible candidates are compared with a declared policy, for example:

1. preserve all safety/validity gates;
2. minimize fixed evaluation photometric residual;
3. minimize reprojection/outlier metrics without degrading coverage beyond a
   tolerance;
4. prefer lower cost/complexity if scientifically equivalent.

If metrics conflict outside tolerances, return `incomparable` and retain the
previous best. `best.json` records previous/new IDs, metric deltas, gates,
policy/version, and rationale. It points to artifacts; it does not copy them.

## 18. Reproducibility and cross-agent compatibility

`runtime.random_seed` currently seeds Python, NumPy, Torch CPU/CUDA, OpenCV/core,
and deterministic Torch algorithms. Stage-local values can override it:
NDeF dense surface seed, fusion seed, sparse seed, and deformation seed. The
effective seed map must expose precedence and persist every resolved value.

Determinism is qualified, not promised absolutely: CUDA libraries, device model,
Torch version, auto-batch memory probing, and parallel numerical reductions may
affect results. Record environment and actual chosen batch. Paired comparisons
use the same hardware class, software versions, input identity, and seeds; when
differences are close to observed repeatability, do not promote.

Any external agent that can read Markdown/JSON/YAML and execute a shell can use
the architecture. Skills never rely on hidden conversation state. A replacement
agent reconstructs context from `run.json`, `trial.json`, stage records,
artifacts, quality, diagnosis, recommendation, and events.

## 19. Testing strategy

### 19.1 Unit and schema tests

- strict recursive override, protected/unknown paths, canonical serialization and
  hashes;
- schema validation and forward-compatible unknown-field handling;
- path containment and overwrite refusal;
- artifact signatures, compatibility, cache/reuse decisions;
- stage DAG cycle checks, invalidation, minimal-rerun plans;
- state transitions, atomic publication, interrupted-stage recovery;
- quality metrics on finite/NaN/empty/boundary fixtures;
- deterministic diagnostic rules with supporting/contradicting evidence;
- bounded recommendations and forbidden parameters;
- best gates, Pareto/conflict handling, and atomic reference update.

### 19.2 CLI tests

- every command's JSON stdout validates against schema and keeps logs on stderr;
- inspect is byte-for-byte non-mutating for the case tree;
- invalid config/input/artifact and unsupported resume produce documented exit
  codes and structured errors;
- dry-run creates nothing and explains reuse/invalidation/output conflicts;
- trial creation never edits baseline YAML;
- an existing non-empty destination is refused without an explicit safe policy.

### 19.3 Scientific regression tests

- retain all current C++ and Python tests for PIN, Stereo, PIN Multi, and NDeF;
- compare adapter-produced numerical artifacts with direct existing API calls on
  identical configs/seeds;
- use compact synthetic golden cases with expected ranges, not brittle exact GPU
  floats;
- create metadata-only known-bad fixtures for each diagnosis code before costly
  solver cases;
- add a small NDeF known-good CPU or bounded-iteration case and injected failures
  for missing masks, weak tracks, nonfinite history, low validity, and corrupt
  artifact;
- run optional full GPU golden cases separately, recording hardware/software and
  statistical tolerances.

Production thresholds must not be learned from the CylinderDIC ground truth and
then presented as universal. Synthetic truth validates metric implementation;
domain datasets and expert review establish quality profiles.

## 20. Phased implementation loops

Each loop is independently testable and should land without requiring later
loops.

### Loop 1 — Schema and read-only state foundation

- **Goal:** define v1 envelopes, IDs, artifact/run/stage/quality records and safe
  path/atomic-read helpers.
- **Files:** `schemas/agent/v1/*`, `python/neurodic/agent/{schemas,errors,artifacts,state}.py`.
- **Inputs:** current output contracts and generated metadata.
- **Outputs:** validated Python records and JSON Schema; no solver calls.
- **Tests:** schema round trips, malformed records, path containment, identities,
  no filesystem mutation during reads.
- **Definition of done:** representative current PIN Multi/NDeF artifacts can be
  inventoried into valid v1 records; all failures are structured.
- **Risks:** over-general schemas and expensive hashing. Keep required fields
  minimal and identity strength explicit.

### Loop 2 — Unified read-only inspection and CLI

- **Goal:** resolve effective case config, frames, solver, readiness, artifacts,
  capabilities, and current stage completion.
- **Files:** `agent/inspect.py`, adapters, `python/neurodic/cli.py`, `pyproject.toml`.
- **Inputs:** solver config, case-path key/file, case root.
- **Outputs:** inspection and pipeline JSON/text; no state writes.
- **Tests:** all four workflows, missing/misaligned inputs, stdout/stderr and exit
  codes, mutation snapshot.
- **Definition of done:** an external shell agent can determine exactly what can
  run/reuse and whether resume is supported without importing execution code.
- **Risks:** doc/config drift and ambiguous solver inference. Require explicit
  selection when evidence conflicts.

### Loop 3 — Evidence adapters and unified evaluation

- **Goal:** normalize existing metrics without changing solvers.
- **Files:** `agent/evaluate.py`, solver adapters, quality profiles.
- **Inputs:** current NPZ/JSON/manifests.
- **Outputs:** versioned `quality.json` with availability, metrics, thresholds,
  findings, and eligibility—not recommendations.
- **Tests:** PIN Multi and NDeF golden metadata, missing/corrupt artifacts,
  deterministic metric math.
- **Definition of done:** evaluation is reproducible and never labels absent PIN
  evidence as passing.
- **Risks:** arbitrary thresholds. Start conservative, profile/version them, and
  return unknown.

### Loop 4 — Missing scientific observability

- **Goal:** add only evidence needed for safe decisions: PIN histories/residual
  evaluation, Stereo reason codes, fixed NDeF evaluation residuals, and overlap
  consistency where justified.
- **Files:** affected C++ result/solver and bindings, thin exporters, tests.
- **Inputs:** existing solve results and fixed evaluation samples.
- **Outputs:** backward-compatible extra diagnostics/artifacts.
- **Tests:** numerical invariants, direct API regression, CPU synthetic cases and
  optional GPU cases.
- **Definition of done:** proposed automatic diagnoses have observable evidence;
  no inference relies only on training final loss.
- **Risks:** scientific semantics and storage. Review each metric and store maps
  only when needed.

### Loop 5 — Deterministic diagnosis

- **Goal:** map quality evidence to failure stage and evidence-backed candidate
  causes.
- **Files:** `agent/diagnose.py`, rule/profile definitions.
- **Inputs:** quality plus inspection.
- **Outputs:** diagnosis JSON; no config change or execution.
- **Tests:** known-good/bad/ambiguous fixtures, contradictory evidence, unknown
  behavior.
- **Definition of done:** every cause cites evidence and missing evidence; weak
  cases request observation/human review.
- **Risks:** false causal certainty. Avoid opaque confidence scores.

### Loop 6 — Trial/config and dry-run planner

- **Goal:** protected baseline, sparse override, effective hash, provenance, DAG
  invalidation and minimal-rerun plan.
- **Files:** `agent/{config,trials,stages}.py`, schemas, CLI additions.
- **Inputs:** inspected run, explicit override/reason/evidence.
- **Outputs:** immutable trial record and dry-run plan; still no solver execution.
- **Tests:** merge/type/range/protected fields, concurrent IDs, DAG plans, baseline
  unchanged, zero-copy references.
- **Definition of done:** a trial can be reproduced from disk and its exact
  affected stages explained.
- **Risks:** state pollution and concurrent writers. Lock narrowly and publish
  atomically.

### Loop 7 — Guarded stage execution and reuse

- **Goal:** execute existing APIs into isolated trial roots, validate outputs,
  publish stage records, and reuse compatible immutable artifacts.
- **Files:** adapters, execution module, minimal refactors of Python exporters.
- **Inputs:** approved trial and stage plan.
- **Outputs:** stage attempts, artifact records, logs, quality; no automatic best.
- **Tests:** adapter equivalence, failure/interruption, overwrite refusal, exact
  cache hits/misses, existing workflow regression.
- **Definition of done:** baseline cannot be overwritten; a failed stage cannot
  appear completed; shared surface reuse avoids copying.
- **Risks:** existing functions combine stages/direct writes. Wrap safely first,
  then split only where atomic execution requires it.

### Loop 8 — Comparison and best management

- **Goal:** gate, compare, and explicitly promote eligible trials.
- **Files:** `agent/compare.py`, best state/CLI.
- **Inputs:** same-scope quality reports and policy.
- **Outputs:** comparison and atomic `best.json` update.
- **Tests:** missing metrics, regression gates, ties/conflicts, unfair seeds,
  previous-best retention.
- **Definition of done:** newest is never implicitly best and every promotion is
  explainable/reversible by reference.
- **Risks:** false ranking from stochastic noise. Require paired evaluation and
  minimum material improvement.

### Loop 9 — Bounded recommendations

- **Goal:** emit one-diagnosis, minimal-change proposals and budget-aware stop
  decisions.
- **Files:** parameter registry, `agent/recommend.py`, workflow Skills.
- **Inputs:** diagnosis, effective config, stage DAG, budget.
- **Outputs:** recommendation JSON and proposed override; execution remains
  separate.
- **Tests:** causal rules, bounds/couplings, unsafe parameters, stop conditions,
  rerun scope.
- **Definition of done:** every proposed change is allowed, evidenced, bounded,
  and maps to a tested rerun plan.
- **Risks:** premature tuning. Enable only for failure modes with validated
  evidence and reviewed causal rules.

### Loop 10 — Skills and compatibility hardening

- **Goal:** publish reviewed project/workflow/stage Skills against stable commands
  and schemas.
- **Files:** `skills/**`, contract tests/examples.
- **Inputs:** implemented APIs, schemas, rule registry, code evidence.
- **Outputs:** cross-agent operating knowledge.
- **Tests:** referenced commands/paths exist, examples validate, routing scenarios,
  version compatibility.
- **Definition of done:** a fresh agent can inspect, explain, propose, and execute
  a bounded trial using only files and shell contracts.
- **Risks:** Skills becoming stale. CI-check references and contract versions.

True optimizer resume is a later dedicated loop after execution is stable. It
must not be smuggled into artifact reuse.

## 21. Minimum viable scope for phase one

Implement Loops 1–3 only, plus the read-only part of Loop 6 (`plan --dry-run`):

- v1 JSON schemas/envelope and structured errors;
- `inspect case/config/pipeline/artifact/result` for all four workflows;
- effective config resolution from solver YAML + named case mapping;
- artifact inventory and identity strength;
- capability flags, including `resume_supported=false` where accurate;
- unified evaluation adapters for current PIN Multi and NDeF evidence, with
  conservative PIN/Stereo availability reporting;
- dry-run dependency/reuse/invalidation plan;
- unit, CLI, golden-metadata, and existing regression tests.

Do not create mutable trials or run a solver in phase one. This scope is small
enough to validate contracts while delivering immediate value to any external
agent: it can understand a case, see what exists, detect missing dependencies,
and judge the evidence already present.

## 22. Explicit non-goals now

- No internal LLM agent, prompt workflow, vendor SDK, web service, UI, account, or
  task distribution.
- No MCP until CLI/Python/JSON contracts are stable; then MCP is a thin adapter.
- No Bayesian/evolutionary/RL search, unconstrained sweeps, or experience
  database.
- No universal scalar quality score or production thresholds inferred from one
  case.
- No automatic change to calibration, scale, units, camera/frame mapping,
  topology, rigid-body semantics, or network architecture.
- No automatic best promotion, artifact deletion, baseline overwrite, or input
  copying.
- No advertised resume until complete algorithm/RNG/optimizer state is saved and
  loaded with compatibility tests.
- No spelling/namespace migration for `pin_multi_slover` bundled with this work.
- No rewrite of numerical workflows merely to fit the CLI.

## 23. Risks and mitigations

| Risk | Mitigation |
|---|---|
| State pollution/baseline overwrite | Unique immutable trial roots, protected baseline, containment checks, atomic publish, explicit promotion. |
| Config override changes scientific meaning | Public registry, protected parameters, sparse diff, reason/evidence, fail unknown paths. |
| Stochastic false improvement | Paired seeds/hardware, fixed evaluation evidence, minimum improvement, repeatability band for close results. |
| Storage explosion | Shared references, per-stage retention, artifact sizes, disk budget, no default copies, explicit GC later. |
| Compute explosion/OOM | Dry-run cost class, budgets, stage-local reruns, actual auto-batch recording, stop on resource failure. |
| Cache poisoning | Strong producer/input/config signatures and schema validation; never trust path existence alone. |
| Misdiagnosis from missing metrics | Availability and `unknown`, competing causes, observation-first actions, reviewed thresholds. |
| Skill drift | Evidence-source links, version compatibility, CI validation of commands/config paths/schema examples. |
| Ground-truth leakage | Tag benchmark-only artifacts and forbid their use in production recommendation policies. |
| Existing combined stage functions | Report coarse execution granularity; refactor incrementally with regression equivalence. |

## 24. Open questions requiring scientific or product decisions

1. Which domain datasets and expert-reviewed tolerances define production quality
   profiles for each solver and material/scale regime?
2. What fixed evaluation sampling strategy is scientifically fair for NDeF and
   PIN, independent of stochastic training batches?
3. Should a run scope be one frame by default, with a parent series record for
   multi-frame cases, or should a run own a complete time series? The former
   simplifies independent retry/best decisions; the latter simplifies shared
   reference artifacts.
4. Which code changes are backward-compatible for the `pin_multi_slover` spelling,
   and when should a correctly spelled alias become canonical?
5. What compatibility policy permits reusing artifacts across Git revisions:
   exact commit only, adapter/scientific version, or reviewed migration?
6. Which large diagnostics should be persisted as full arrays versus streaming
   summaries plus sampled evidence?
7. How should multi-metric best selection prioritize surface fidelity,
   photometric fit, spatial regularity, and computational cost for each domain?
8. Is cross-seed repeatability required for all promoted trials or only marginal
   improvements/high-risk parameter groups?

Until these are resolved, the architecture should expose uncertainty and require
explicit policy rather than encode accidental defaults.
