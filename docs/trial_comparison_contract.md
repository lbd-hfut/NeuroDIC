# Trial Comparison and Best Contract

Version: `neurodic.comparison/v1`

Loop 8 is native-free and solver-free. It compares only existing formal
`QualityReport` records; it never reads raw arrays, invokes evaluation or
execution, changes a TrialPlan/manifest/artifact, or recommends a parameter.

The metric alignment key is `(metric_id, scope, aggregation, unit)`. Scope is
therefore camera-, pair-, frame-, and field-aware; a stereo planar field or an
NDeF camera cannot silently compare with another. Fixed-evaluation metrics
also require the same `evaluation_set_identity`. Unit conversion is forbidden.
Protected scientific identity is report provenance (`case`, calibration/camera
mapping, coordinate/scale convention under `neurodic.protected-scientific-identity/v1`);
missing or different identity makes the result scientifically incomparable.

`config/comparison_profiles/default.yaml` is the auditable, versioned source of
metric direction, role, tolerance, required evidence, and hard finding codes.
Directions are `lower_is_better`, `higher_is_better`, or `neutral`; omitted
metrics cannot decide selection. Tolerance defines only numerical equality,
not scientific significance. There is deliberately no weighted or overall score.

Missing is incomparable, never worse. Matching `not_applicable` stays not
applicable. `corrupt` is unusable and is a hard exclusion when required.
Comparison support is strong for equal sample support and moderate otherwise.

Eligibility is evaluated before preference: `eligible`, `ineligible`, or
`insufficient_evidence`. Hard quality findings, corrupt required evidence, and
partial/failed/interrupted result provenance are ineligible for a full-result
best. The Loop 7 PIN Multi pair-ROI smoke is a partial preprocessing result and
can never replace a full PIN Multi best.

Selection is lexicographic, not scalar: an eligible comparable candidate must
strictly improve at least one primary metric, must not regress a primary or
guardrail, and otherwise retains the current result. Ties and Pareto conflicts
retain current deterministically.

`compare_quality_reports()` is read-only. `update_best()` is an explicit,
separate mutation that re-verifies comparison baseline and candidate quality identities,
uses an expected-current-best CAS identity, and atomically writes only
`<managed-root>/best/current.json` plus append-only `best/history/*.json`.
Best records reference reports by identity; they never copy artifacts or modify
trials. A stale report raises `BEST.COMPARISON_STALE`; a changed pointer raises
`BEST.STATE_CHANGED`.
