from neurodic.ndef_paths import camera_name_from_label, make_ndef_run_mapping, ndef_run_roots


def test_ndef_run_roots_default_and_isolated_namespace(tmp_path):
    result, visualization = ndef_run_roots(tmp_path, {"output": {}})
    assert result == tmp_path / "result" / "ndef"
    assert visualization == tmp_path / "visualization" / "ndef"

    result, visualization = ndef_run_roots(tmp_path, {"output": {"ndef_subdir": "ndef_multi_slover"}})
    assert result == tmp_path / "result" / "ndef_multi_slover"
    assert visualization == tmp_path / "visualization" / "ndef_multi_slover"


def test_run_mapping_keeps_calibration_shared_and_ndef_products_isolated():
    source = {"case": {"root": "case/Multi/CylinderDIC"}, "precalculation": {}, "output": {}}
    mapped = make_ndef_run_mapping(source, "case/Multi/ComplexCylinderDIC")
    assert source["case"]["root"] == "case/Multi/CylinderDIC"
    assert mapped["case"] == {
        "root": "case/Multi/ComplexCylinderDIC",
        "calibration": "result/calibration/calibration_result_scaled.json",
        "masks": "result/ndef_multi_slover/roi/per_camera",
        "reference_surface": "result/ndef_multi_slover/surface/deformation_surface_dataset.npz",
    }
    assert mapped["precalculation"]["displacement"] == "result/ndef_multi_slover/precalculation/sparse_tracks.npz"


def test_camera_name_from_path_or_plain_label():
    assert camera_name_from_label("images/cam_10/001.bmp", "cam_0") == "cam_10"
    assert camera_name_from_label("cam_10", "cam_0") == "cam_10"
