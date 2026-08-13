# Deterministic Diagnosis Contract

`neurodic.diagnosis/v1` interprets one existing QualityReport without reading
scientific artifacts, invoking native code, or changing the filesystem.
Diagnosis rules are versioned as `neurodic-diagnosis-rules/v1`.

Support is categorical, never probabilistic. `strong` requires direct,
available stage evidence without a critical contradiction; `moderate` has a
direct pattern but alternative mechanisms or a relevant contradiction; `weak`
is indirect only; `insufficient` is not emitted as a failure family when the
required evidence cannot distinguish one.

Every diagnosis records supporting evidence, actively checked contradicting
evidence, missing evidence, bounded mechanism-level candidate causes, and a
non-operational next observation. Primary selection is deterministic and
upstream-first: workflow stage order, support rank, then code. Candidate causes
describe mechanisms, not parameter changes. The report intentionally contains
no recommendation, command, rerun, trial, or configuration-change field.
