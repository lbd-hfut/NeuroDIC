# C++ Core

The C++ core exposes public interfaces under `include/neurodic` and skeleton
implementations under `src`.

`neurodic_core` is the CMake target for the scientific core. It links LibTorch
when available and is the dependency of the future pybind11 extension.

Current modules are interface-heavy and TODO-heavy by design:

```text
core, data, interpolation, initialization, calibration, problem,
representation, model, geometry, loss, optimizer, solver, postprocess
```

No module should return fabricated numerical output. Scientific methods either
throw a TODO exception or remain abstract.
