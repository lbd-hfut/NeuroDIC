"""Direct regression for the two PIN 2D seed initialization strategies."""

from pathlib import Path


def test_seed_case(case_root: Path, config_path: Path) -> None:
    import cv2
    import numpy as np
    from neurodic.seeds import run_seed_case

    roi = cv2.imread(str(case_root / "003.bmp"), cv2.IMREAD_GRAYSCALE) != 0
    for strategy in ("integer_search", "sift_search"):
        summary = run_seed_case(case_root, strategy, config_path)
        assert summary["seed_count"] >= 3, summary
        assert Path(summary["result_file"]).is_file(), summary
        assert Path(summary["visualization_file"]).is_file(), summary
        seeds = np.load(summary["result_file"])["seed_pos"]
        xy = np.rint(seeds).astype(int)
        assert np.all(roi[xy[:, 1], xy[:, 0]]), summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    test_seed_case(project_root / "case" / "2D" / "ring", project_root / "config" / "pin_2d.yaml")
