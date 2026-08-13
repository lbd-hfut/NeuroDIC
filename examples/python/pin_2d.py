"""Run planar PIN-DIC using the ``pin_2d`` paths in config/case_paths.yaml."""

import neurodic


if __name__ == "__main__":
    config = neurodic.load_case_config("config/pin_2d.yaml", "pin_2d")
    neurodic.run_planar_case(config)
