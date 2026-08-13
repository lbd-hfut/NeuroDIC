"""Run NDeF-DIC using the ``ndef_multi`` paths in config/case_paths.yaml."""

import neurodic


if __name__ == "__main__":
    neurodic.ndef_dic(neurodic.load_case_config("config/ndef_multi.yaml", "ndef_multi"))
