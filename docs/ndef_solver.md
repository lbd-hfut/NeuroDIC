# NDeFSolver

`NDeFSolver` owns the NDeF multi-view DIC flow and controls its internal model
topology.

```text
NDeFProblem
   ↓
Reference Surface Representation
   ↓
NDeF Deformation Representation
   ↓
Internal Model
   ↓
NDeFGeometry
   ↓
Photometric Loss
   ↓
Optimizer
   ↓
NDeFResult
```

Public Python or YAML APIs must not expose arbitrary NDeF layer counts, widths,
skip topology, or internal branches.
