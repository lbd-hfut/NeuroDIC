"""Run stereo PIN-DIC using the ``pin_stereo`` paths in config/case_paths.yaml."""

import neurodic



if __name__ == "__main__":
    neurodic.run_stereo_case(neurodic.load_case_config("config/pin_stereo.yaml", "pin_stereo"))
