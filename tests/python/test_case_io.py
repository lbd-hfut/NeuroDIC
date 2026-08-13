"""Image ordering contracts for path-only case configuration."""

from pathlib import Path

from neurodic.case_io import named_multiview_image_pairs, planar_image_series, stereo_image_pairs


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_planar_series_uses_first_reference_last_roi_and_all_intermediate_frames(tmp_path: Path) -> None:
    _touch(tmp_path, "003.bmp", "001.bmp", "002.bmp", "004.bmp", "notes.txt")
    reference, deformed, roi = planar_image_series(tmp_path)
    assert reference.name == "001.bmp"
    assert [path.name for path in deformed] == ["002.bmp", "003.bmp"]
    assert roi.name == "004.bmp"


def test_stereo_and_multiview_discovery_align_frames_by_sorted_index(tmp_path: Path) -> None:
    _touch(tmp_path / "left", "00_L.bmp", "03_L.bmp", "04_L.bmp")
    _touch(tmp_path / "right", "00_R.bmp", "03_R.bmp", "04_R.bmp")
    reference, deformed = stereo_image_pairs(tmp_path / "left", tmp_path / "right")
    assert [path.name for path in reference] == ["00_L.bmp", "00_R.bmp"]
    assert [[path.name for path in pair] for pair in deformed] == [["03_L.bmp", "03_R.bmp"], ["04_L.bmp", "04_R.bmp"]]

    _touch(tmp_path / "views" / "cam_0", "001.bmp", "002.bmp", "003.bmp")
    _touch(tmp_path / "views" / "cam_1", "001.bmp", "002.bmp", "003.bmp")
    references, frames = named_multiview_image_pairs(tmp_path / "views", ["cam_0", "cam_1"])
    assert [path.name for path in references] == ["001.bmp", "001.bmp"]
    assert [[path.name for path in frame] for frame in frames] == [["002.bmp", "002.bmp"], ["003.bmp", "003.bmp"]]
