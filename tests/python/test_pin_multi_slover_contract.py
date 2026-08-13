"""Regression guard: the reserved multi route must stay separate from NDeF."""

from pathlib import Path

from neurodic.config import load_case_config


ROOT = Path(__file__).resolve().parents[2]


def test_pin_multi_slover_config_owns_pairwise_sift_roi() -> None:
    config = load_case_config(ROOT / "config/pin_multi.yaml", "pin_multi", ROOT / "config/case_paths.yaml")
    assert config["solver"] == "pin_multi_slover"
    assert config["pair_roi"]["generator"] == "reference_pair_sift"
    assert "ndef" not in str(config["pair_roi"]).lower()
    assert config["pair_roi"]["output"] != "result/mask/per_camera"
