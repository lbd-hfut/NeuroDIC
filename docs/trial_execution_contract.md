# Guarded Trial Execution Contract

Version: `neurodic.execution/v1`

Loop 7 consumes a previously emitted `neurodic.trial_plan/v1`; execution never
accepts a config plus a new override. Before any write, it regenerates the plan
from the frozen baseline sources and sparse override, compares plan identity and
all derived fields, and rejects stale or edited plans.

`execute_trial()` creates a new isolated workspace under a caller-approved
managed root, validates its lifecycle trial ID, and refuses an existing trial
directory. All files written by the control plane are contained under that
workspace: `manifest.json`, frozen override/effective-config snapshots,
`staging/`, and attempt-versioned `artifacts/`. The baseline is never a write
root.

Each action starts an immutable stage attempt. Outputs must be nonempty regular
files contained in the attempt staging root. Only after validation are they
atomically renamed to `artifacts/<stage>/<attempt-id>/`; a completed manifest is
then atomically updated. Exceptions produce `failed`; `KeyboardInterrupt`
produces `interrupted`; neither publishes partial outputs.

New managed artifacts use content SHA-256 identities and a deterministic
producer signature: canonical stage ID, adapter-specific versioned
implementation identity, NeuroDIC `git-head-plus-dirty/v1` revision policy,
stage-owned configuration projection, adapter-declared required input
identities, scope, and output contract. Producer signatures are distinct from
artifact content hashes. Implementation identities must not depend on trial ID,
attempt ID, or filesystem location.

The CPU-only `pin_multi.separate_pair_roi_call` adapter is registered, but only
for a plan whose immutable scope contains one validated selected `pair_id`.
It calls the low-level pair-ROI function directly and never invokes the legacy
high-level wrapper or its baseline manifest path. PIN, Stereo, PIN Multi solve,
and NDeF actions remain `EXECUTION.UNSUPPORTED` before a workspace is created:
their output contracts and conceptual minimum-stage execution are not yet
verified. Test-only `TrustedAction` adapters exercise the same lifecycle.

Runtime capability is exposed on every serialized TrialPlan execution action:
`execution_supported`, `scope_requirement`, `completion_scope`, and
`capability_notes`. The native-free execution adapter registry is the sole
truth source for these fields; planning still owns conceptual stages/actions.
Currently only `pin_multi.separate_pair_roi_call` is supported and requires
`scope.pair_id`; its `requested_action_only` completion scope means a successful
attempt remains a partial trial when other planned actions exist. All PIN,
Stereo, PIN Multi solve/fusion, and NDeF conceptual actions expose false and
fail closed before workspace creation.

Legacy artifacts lack producer signatures and remain `candidate_unverified`.
A managed artifact is safe-reused only when every artifact in one prior stage
attempt has the exact expected producer signature and its current content
identity still equals the published identity. The new trial records references
to that immutable source attempt; it does not rerun the adapter or rewrite the
source evidence.
