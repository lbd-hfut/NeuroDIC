"""Configuration composition keeps dataset locations out of solver YAMLs."""

from pathlib import Path

from neurodic.config import load_case_config, load_config
from neurodic.ndef_paths import ndef_run_roots


ROOT = Path(__file__).resolve().parents[2]


def test_solver_configs_do_not_embed_case_or_output_paths() -> None:
    for name in ("pin_multi.yaml", "pin_stereo.yaml", "ndef_multi.yaml"):
        values = load_config(ROOT / "config" / name)
        assert "case" not in values
        assert "output" not in values


def test_case_paths_supply_pin_multi_inputs_and_outputs() -> None:
    values = load_case_config(ROOT / "config/pin_multi.yaml", "pin_multi", ROOT / "config/case_paths.yaml")
    assert values["case"]["images"] == "images"
    assert "reference_frame" not in values["case"]
    assert values["case"]["calibration"] == "result/calibration_multiview/calibration_result_scaled.json"
    assert values["pair_roi"]["output"] == "result/pin_multi/pair_roi"
    assert values["output"]["visualization"] == "visualization/pin_multi"
    assert values["pin_2d_config"] == "config/pin_2d.yaml"
    assert values["fusion"]["enabled"] is True
    assert values["traditional_strain"]["neighbors"] == 12


def test_case_paths_supply_planar_image_order() -> None:
    values = load_case_config(ROOT / "config/pin_2d.yaml", "pin_2d", ROOT / "config/case_paths.yaml")
    assert values["case"]["images_dir"] == "."
    assert "image_sequence" not in values["case"]
    assert "roi" not in values["case"]


def test_case_paths_keep_every_generated_product_under_its_module_directory() -> None:
    paths = load_config(ROOT / "config/case_paths.yaml")
    assert paths["pin_2d"]["output"]["result"] == "result/pin_2d"
    assert paths["pin_stereo"]["output"]["result"] == "result/pin_stereo"
    assert paths["pin_multi"]["output"]["result"] == "result/pin_multi"
    assert paths["calibration_stereo"]["outputs"]["result_subdir"] == "calibration_stereo"
    assert paths["calibration_multiview"]["outputs"]["result_subdir"] == "calibration_multiview"
    ndef = load_case_config(ROOT / "config/ndef_multi.yaml", "ndef_multi", ROOT / "config/case_paths.yaml")
    result, _ = ndef_run_roots(ROOT / "case/Multi/CylinderDIC", ndef)
    assert result == (ROOT / "case/Multi/CylinderDIC/result/ndef_multi").resolve()
