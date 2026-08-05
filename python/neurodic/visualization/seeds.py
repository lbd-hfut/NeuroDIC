"""Sparse seed correspondence visualization."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def visualize_seed_matches(reference: np.ndarray, deformed: np.ndarray, seed_pos: np.ndarray,
                           seed_uv: np.ndarray, output_path: str | Path, strategy: str) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurodic-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    h, w = reference.shape[:2]
    canvas = np.concatenate([reference, deformed], axis=1)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    positions = np.asarray(seed_pos, dtype=float).reshape((-1, 2))
    displacement = np.asarray(seed_uv, dtype=float).reshape((-1, 2))
    if len(positions):
        deformed_points = positions + displacement + np.array([w, 0.0])
        segments = np.stack([positions, deformed_points], axis=1)
        ax.add_collection(LineCollection(segments, colors="lime", linewidths=0.45, alpha=0.72))
        ax.scatter(positions[:, 0], positions[:, 1], s=6, c="deepskyblue", linewidths=0, label="reference seed")
        ax.scatter(deformed_points[:, 0], deformed_points[:, 1], s=6, c="tomato", linewidths=0, label="deformed match")
    ax.axvline(w - 0.5, color="white", linewidth=1.0, alpha=0.8)
    ax.set(title=f"{strategy}: {len(positions)} cleaned seed correspondences")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=2, framealpha=0.85)
    fig.tight_layout(pad=0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output
