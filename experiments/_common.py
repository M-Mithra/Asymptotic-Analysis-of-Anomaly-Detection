"""Shared helpers for the experiment scripts (paths, plotting style, labels)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from universal_outlier import detectors  # noqa: E402

RESULTS = ROOT / "results"

# One fixed colour per rule so every figure in the project reads the same.
COLORS = {
    "lrt": "#222222",
    "glt": "#1f77b4",
    "kl_avg": "#d62728",
    "kl_centroid": "#ff9896",
    "mmd_avg": "#2ca02c",
    "mmd_maxmin": "#98df8a",
    "closest_pair": "#8c564b",
    "tvd_avg": "#9467bd",
    "tvd_centroid": "#c5b0d5",
    "mean_based": "#ff7f0e",
    "median_based": "#bcbd22",
}
MARKERS = dict(zip(COLORS, "Dovs^<>xP*h"))


def label(key: str) -> str:
    return detectors.get(key).label


def outdir(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False,
    })


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  saved {path.relative_to(ROOT)}")
