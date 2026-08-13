# Multi-View Observability Contract

Loop 4B adds append-only evidence only. It does not change reconstruction,
fusion winner selection, NDeF optimizer updates, checkpoint selection, or the
Loop 4A fixed-evaluation identity.

Stereo writes `diagnostics/stereo_geometry.{npz,json}` with one primary reason
per point. Priority is `invalid_field`, `outside_roi`, `out_of_bounds`,
`negative_depth`, `reprojection_error`, then `valid`. Reference and current
reprojection errors remain separate arrays.

NDeF extends the existing `neurodic.fixed_evaluation/v1` artifact with one row
for every fixed-surface-index and reference-visible camera. A row records
camera, current positive-depth, in-bounds and patch-valid flags, and a residual
only when all current photometric requirements hold. Per-point cross-view
spread is `max(residual)-min(residual)` over two or more valid rows; per-camera
statistics are summary-only evidence, not causal attribution.

PIN Multi defines overlap as two or more source pairs in the same exact
post-filter, pre-selection reference-space voxel. For each overlap voxel,
disagreement is the per-candidate Euclidean distance to the componentwise
median displacement; the artifact stores median/p95 disagreement and maximum
distance from the componentwise median reference position. The grouping uses a
lexicographic voxel sort, O(N log N) time and O(N) candidate memory; it never
feeds back into source selection.
