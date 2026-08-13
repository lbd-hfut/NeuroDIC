"""Run pairwise multi-camera PIN-DIC, fuse surfaces, then compute fused 3D strain."""

import neurodic


if __name__ == "__main__":
    # ``fusion.enabled`` is true in config/pin_multi.yaml: pair products are
    # fused and traditional 3D strain is evaluated only on that fused surface.
    neurodic.run_pin_multi_case(neurodic.load_case_config("config/pin_multi.yaml", "pin_multi"))
