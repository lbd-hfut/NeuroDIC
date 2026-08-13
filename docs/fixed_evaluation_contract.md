# Fixed Evaluation Contract

Version: `neurodic.fixed_evaluation/v1`

Training observations are the losses consumed by optimizer updates. Fixed
evaluation observations are post-training, no-gradient photometric
measurements used only to describe a resulting field. They must never alter
model initialization, training draws, gradient graphs, checkpoint selection,
or optimizer state.

PIN uses a bounded stable-hash ranking of ROI coordinate indices. NDeF uses the
same policy over the supplied reference-surface ordering, and evaluates every
reference-visible camera observation for each selected point. The index order,
seed, eligible population, loss family, and patch semantics are exported in a
versioned evaluation artifact. Training hyperparameters such as learning rate,
epochs, and network topology are deliberately not part of that identity.

Evaluation is opt-in (`evaluation.enabled: true`) to preserve existing workflow
cost. `sample_count` and `seed` are solver-specific bounded settings. PIN also
accepts `patch_radius` (zero means its existing ZNSSD window radius). A fixed
residual is comparable only when case inputs, ROI/surface ordering, image pair
or camera scope, evaluation identity, and loss semantics agree.

PIN writes `diagnostics_training.npz`, `diagnostics_evaluation.npz`, and
`diagnostics_evaluation.json` beside `pin_result.npz`; stereo writes equivalent
per-field data under `disp/`; NDeF writes `diagnostics/evaluation.npz` and
`diagnostics/evaluation.json`. The JSON summary records requested, valid, and
eligible/supervised counts plus nonfinite-safe residual statistics. A missing
or nonfinite mean is evidence-unavailable/corrupt, never a zero residual.
