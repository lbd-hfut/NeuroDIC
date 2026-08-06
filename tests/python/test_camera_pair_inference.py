import numpy as np

from neurodic.calibration import infer_multiview_camera_pairs


def _calibration(centers):
    cameras = [
        {"label": f"cam_{index}", "camera_center": np.asarray(center, dtype=float).tolist()}
        for index, center in enumerate(centers)
    ]
    return {"cameras": cameras, "inlier_match_counts": np.zeros((len(cameras), len(cameras))).tolist()}


def test_camera_pair_inference_recovers_closed_ring():
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    centers = np.column_stack((3.0 * np.cos(angles), 3.0 * np.sin(angles), np.zeros_like(angles)))

    result = infer_multiview_camera_pairs(_calibration(centers))

    assert result["topology"] == "closed"
    assert result["ordered_camera_names"] == [f"cam_{index}" for index in range(8)]
    assert all(len(value) == 2 for value in result["neighbors"].values())
    assert set(result["neighbors"]["cam_0"]) == {"cam_1", "cam_7"}


def test_camera_pair_inference_recovers_chain_and_single_end_neighbors():
    centers = np.asarray([[value, 0.02 * value**2, 0.0] for value in range(6)], dtype=float)

    result = infer_multiview_camera_pairs(_calibration(centers))

    assert result["topology"] == "chain"
    assert result["ordered_camera_names"] == [f"cam_{index}" for index in range(6)]
    assert result["neighbors"]["cam_0"] == ["cam_1"]
    assert result["neighbors"]["cam_5"] == ["cam_4"]
    assert all(len(result["neighbors"][f"cam_{index}"]) == 2 for index in range(1, 5))


def test_camera_pair_inference_opens_an_incomplete_arc_at_its_largest_gap():
    angles = np.deg2rad([20.0, 50.0, 80.0, 110.0, 140.0])
    centers = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))

    result = infer_multiview_camera_pairs(_calibration(centers))

    assert result["topology"] == "chain"
    assert result["ordered_camera_names"] == [f"cam_{index}" for index in range(5)]
