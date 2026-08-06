from pathlib import Path

import neurodic.calibration as calibration


ROOT = Path(__file__).resolve().parents[2]


def test_stereo_plate_center_load_calibration():
    case_root = ROOT / "case/Stereo/plate_center_load"
    result = calibration.run_stereo_case(case_root, config=ROOT / "config/calibration.yaml")["result"]

    assert sum(item["found"] for item in result["left_detections"]) == 20
    assert sum(item["found"] for item in result["right_detections"]) == 20
    assert len(result["kept_pair_indices"]) >= 6
    assert result["rms_error"] > 0.0


def test_multiview_cylinder_calibration_and_scale_recovery():
    case_root = ROOT / "case/Multi/CylinderDIC"
    output = calibration.run_multiview_case(case_root, config=ROOT / "config/calibration.yaml")
    result = output["result"]
    scale = output["scale"]

    assert len(result["cameras"]) == 12
    assert len(result["points3d"]) > 0
    assert result["mean_reprojection_error"] >= 0.0
    assert scale["sfm_to_world_scale"] > 0.0
    assert len(scale["scaled_cameras"]) == 12
    assert len(scale["scaled_points3d"]) == len(result["points3d"])
    assert len(scale["sfm_to_world_rotation"]) == 3
    assert len(scale["sfm_to_world_translation"]) == 3
    assert scale["triangulated_corners"] == 63
    assert scale["valid_edges"] == 110
