"""Filesystem-only case discovery shared by the example workflows.

Solver YAML files never name individual images.  A case directory defines its
time ordering by lexicographic image-file order: the first image is reference;
for planar cases the last image is the ROI; all intervening images are
deformed frames.  Stereo and multiview cases use index-aligned images across
their view directories.
"""

from __future__ import annotations

from pathlib import Path


_IMAGE_SUFFIXES = {".bmp", ".dib", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_files(directory: str | Path) -> list[Path]:
    """Return image files in deterministic case order, ignoring helper files."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported image files in {directory}")
    return paths


def planar_image_series(case_root: str | Path, images_dir: str | Path = ".") -> tuple[Path, list[Path], Path]:
    """Return ``(reference, deformed_frames, roi)`` for a planar case."""
    root = Path(case_root) / Path(images_dir)
    paths = image_files(root)
    if len(paths) < 3:
        raise ValueError("Planar case requires reference, at least one deformed image, and a final ROI image")
    return paths[0], paths[1:-1], paths[-1]


def stereo_image_pairs(left_dir: str | Path, right_dir: str | Path) -> tuple[tuple[Path, Path], list[tuple[Path, Path]]]:
    """Pair sorted left/right frames by index and return reference plus deformed pairs."""
    left, right = image_files(left_dir), image_files(right_dir)
    if len(left) != len(right):
        raise ValueError(f"Stereo views must have equal frame counts, got {len(left)} left and {len(right)} right")
    if len(left) < 2:
        raise ValueError("Stereo case requires one reference pair and at least one deformed pair")
    pairs = list(zip(left, right))
    return pairs[0], pairs[1:]


def multiview_image_pairs(images_dir: str | Path) -> tuple[list[str], list[Path], list[list[Path]]]:
    """Return view names, their first-frame references, and index-aligned later frames."""
    root = Path(images_dir)
    views = sorted(path for path in root.iterdir() if path.is_dir())
    if len(views) < 2:
        raise ValueError(f"Multi-view case requires at least two view directories under {root}")
    frames = [image_files(view) for view in views]
    counts = {len(items) for items in frames}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Every multi-view directory must contain the same reference-plus-deformed frame count")
    return [path.name for path in views], [items[0] for items in frames], [
        [items[index] for items in frames] for index in range(1, len(frames[0]))
    ]


def named_multiview_image_pairs(images_dir: str | Path, view_names: list[str]) -> tuple[list[Path], list[list[Path]]]:
    """Like :func:`multiview_image_pairs`, but preserves calibration view order."""
    frames = [image_files(Path(images_dir) / name) for name in view_names]
    counts = {len(items) for items in frames}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Every calibrated view must contain the same reference-plus-deformed frame count")
    return [items[0] for items in frames], [[items[index] for items in frames]
                                             for index in range(1, len(frames[0]))]
