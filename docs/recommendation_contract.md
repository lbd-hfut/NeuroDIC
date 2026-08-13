# Bounded Recommendation Contract

Version: `neurodic.recommendation/v1`

Loop 9 turns a formal `DiagnosisReport` into at most one bounded sparse trial
override. It separates evidence, findings, diagnosis, candidate cause,
reviewed intervention rule, and recommendation. A diagnosis is never itself a
parameter change, and the recommendation is an intervention hypothesis rather
than a causal claim or outcome guarantee.

The auditable parameter registry is `neurodic.parameter_registry/v1`. Every
entry records path, solver, Loop 6 owner, type, config-valid range,
trial-modifiable flag, auto-recommendable flag, auto-safe range, direction,
step policy, coupling, risk, protected status, and notes. Config-valid is not
auto-safe: a value must satisfy both. Protected identity (case/output/solver,
calibration, scale/world scale, camera/frame/ROI mapping, units and coordinate
convention) is never auto-recommendable. Network architecture, scientific
quality gates, evaluation protocol, smoothness, calibration, and geometry
thresholds are manual-only.

Rules are `neurodic.intervention_rules/v1`. A rule declares failure families,
candidate causes, minimum support, evidence and contradiction gates, parameter
changes, coupling, owning stage, expected mechanism, risk, stop conditions,
and evidence level. `reviewed_mechanistic` means reviewed, bounded, plausible,
implementation-compatible, and not contradicted; it is not empirical proof.

Weak and insufficient diagnoses never produce overrides. Moderate diagnoses
need an explicit low-risk opt-in; strong diagnoses still need a matching rule.
Required evidence must be present and named contradictions block the rule.
The default is exactly one change; two are permitted only by an explicit
coupled rule, and more than two are rejected.

The initial rule set deliberately contains only
`ndef.deformation.numeric_step_reduction`: strong
`TRAINING.NUMERICAL_FAILURE` with candidate cause
`OPTIMIZATION.STEP_INSTABILITY`, finite-history and valid-observation evidence,
and no listed contradiction may reduce
`deformation_training.photometric_learning_rate` by 0.5. Its hard range is
`[1e-6, 0.1]`; its narrower auto-safe range is `[0.000375, 0.003]`. It changes
only the `ndef.deformation.train` owner stage and is low risk because it
reduces optimizer step magnitude without changing protected scientific
identity. It is reviewed-mechanistic, not statistically validated.

Every candidate is passed to Loop 6 `plan_trial()`; planner policy, ownership,
and DAG invalidation remain final gates. The report embeds the dry-run
`TrialPlan` impact but never creates a workspace, writes YAML, calls
`execute_trial`, invokes a solver, changes best state, searches, or reads trial
history. If the planner blocks, recommendation status is `plan_blocked`.

`neurodic recommend --diagnosis <report> --config <yaml>` emits one strict JSON
RecommendationReport. `recommended`, `observation_only`, `no_matching_rule`,
and evidence-blocked outcomes are successful read-only operations; only input
or contract errors are nonzero.
