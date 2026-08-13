# Trial Planning Contract

Version: `neurodic.trial_plan/v1`

Loop 6 is a native-free, read-only planning layer. `plan_trial()` merges a
validated sparse override into an in-memory effective configuration and returns
a `TrialPlan`; it never creates a trial, manifest, effective-config file,
directory, artifact, process, or solver invocation.

The baseline is immutable: its source YAML, case-path mapping, resolved
effective configuration, case root, artifacts, diagnostics, and output roots
are only read. A plan always has `dry_run: true`, `execution_performed: false`,
`baseline_writes: []`, and `would_write: []`.

Overrides contain leaf fields only. Unknown paths, empty map leaves, type
mismatches, non-finite numbers, and arbitrary deletion are rejected. Values
equal to the baseline make no config change. Canonical JSON serialization with
sorted keys supplies the effective-config SHA-256 identity, independent of YAML
formatting or map ordering.

`case`, `output`, `solver`, `mode`, and `notes` are protected for every solver.
PIN additionally protects its ROI selector; Stereo protects its ROI selector
and physical `reconstruction.world_scale`; PIN Multi additionally protects
`pin_2d_config`, `pair_roi.output`, `camera_pairs`, and physical
`reconstruction.world_scale`; NDeF protects the `scale` subtree and external
`precalculation.displacement`. Since case contains all actual
raw-image, ROI, calibration, camera/order, frame/reference, and configured
result paths, subtree policy prevents a trial from changing scientific identity
or output placement. Protected paths produce a structured blocked plan.

The ownership registry maps real algorithm paths to direct canonical stages;
the Loop 2 adapter DAG is the sole source of downstream closure. Unknown
ownership blocks rather than guessing. In particular, `evaluation.*` maps only
to the evaluation conceptual stage, while NDeF deformation training maps to
`ndef.deformation.train` and downstream stages.

Reuse is deliberately two-dimensional. `scientifically_reusable` is a DAG/config
claim and can remain true when the corresponding on-disk artifact is missing or
unproven; `adapter_can_skip` describes present executable capability. Legacy
inspection records have metadata-only identities and no producer signature,
effective-config identity, input identity, or compatibility proof, therefore
are `candidate_unverified`, never `safe_reuse`. Missing or unverified producer
outputs expand the minimum rerun closure, but do not alter scientific ownership.
`execution_actions` are only machine-readable `would_execute` descriptions;
current APIs expose coarse combined calls and no guarded stage execution.

A no-op override is always `no_effect` with an empty minimum rerun set. Missing
artifact restoration is a separate explicit dry-run intent (`restore_missing` /
`--restore-missing`), which may produce a plan with no config changes but a
nonempty rerun closure.
