"""Experiment 3 - decision regions of the M = 3 rules in the (t1, t2, t3) cube.

Replaces cube_full_code*.py, code_*_cube_1.py, KL_frames/, MMD_frames/,
GLRT_Frames/, TVD_frames/, boundary_lines/boundary.py and the npz viewers.

Regions are drawn exactly with ``imshow`` on a fine slice grid. (The original
"solid shape" plots wrapped DBSCAN clusters in convex hulls, but the regions
are not convex - in a slice they form an X pattern - so hulls painted over
the neighbouring regions.)

    python experiments/decision_regions.py
    python experiments/decision_regions.py --rules glt,kl_centroid --frames
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from _common import label, outdir, plt, save, style

from universal_outlier.regions import (CLASS_COLORS, CLASS_NAMES, NO_OUTLIER, classify_cube,
                                       disagreement, slice_labels)

CMAP = ListedColormap([CLASS_COLORS[k] for k in range(5)])
LEGEND = [Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in (1, 2, 3, 4)]


def draw_slice(ax, rule, axis, value, step):
    u, v, img, names = slice_labels(rule, axis, value, step)
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmin=-0.5, vmax=4.5,
              interpolation="nearest", alpha=0.85)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(f"{label(rule)}: {axis} = {value:.2f}", fontsize=9)


def slices_grid(rules, axis, values, step, path):
    fig, axes = plt.subplots(len(rules), len(values), figsize=(2.6 * len(values), 2.6 * len(rules)),
                             squeeze=False)
    for i, r in enumerate(rules):
        for j, val in enumerate(values):
            draw_slice(axes[i, j], r, axis, val, step)
            axes[i, j].grid(False)
    fig.legend(handles=LEGEND, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    save(fig, path)


def boundary_overlay(rules, axis, value, step, path):
    """Region boundaries of several rules on one slice (replaces boundary.py)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []
    for r, c in zip(rules, colors):
        u, v, img, names = slice_labels(r, axis, value, step)
        edge = np.zeros(img.shape, bool)
        edge[:, 1:] |= img[:, 1:] != img[:, :-1]
        edge[1:, :] |= img[1:, :] != img[:-1, :]
        yy, xx = np.nonzero(edge)
        ax.scatter(u[xx], v[yy], s=1, color=c)
        handles.append(Patch(color=c, label=label(r)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(f"decision boundaries, {axis} = {value:.2f}")
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    save(fig, path)


def scatter_3d(df, rule, path, max_points=12000, seed=0):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(projection="3d")
    for k in (1, 2, 3, 4):
        s = df[df.label == k]
        if len(s) > max_points // 4:
            s = s.iloc[rng.choice(len(s), max_points // 4, replace=False)]
        ax.scatter(s.t1, s.t2, s.t3, s=2, color=CLASS_COLORS[k], alpha=0.5, label=CLASS_NAMES[k])
    ax.plot([0, 1], [0, 1], [0, 1], "k--", lw=2, label=CLASS_NAMES[NO_OUTLIER])
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    ax.set_zlabel("t3")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(label(rule))
    ax.legend(fontsize=7, loc="upper left", markerscale=4)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", default="glt,kl_centroid,kl_avg,mmd_avg,tvd_centroid")
    ap.add_argument("--step", type=float, default=0.02, help="cube grid spacing")
    ap.add_argument("--slice-step", type=float, default=0.004, help="slice image resolution")
    ap.add_argument("--frames", action="store_true", help="also write every slice as its own PNG")
    args = ap.parse_args()
    style()
    rules = args.rules.split(",")
    out = outdir("decision_regions")

    cubes = {r: classify_cube(r, args.step) for r in rules}
    for r, df in cubes.items():
        counts = df.label.map(CLASS_NAMES).value_counts(normalize=True).round(4).to_dict()
        print(f"  {label(r):<16s} {counts}")
        scatter_3d(df, r, out / f"cube3d_{r}.png")

    slices_grid(rules, "t2", [0.1, 0.3, 0.5, 0.7, 0.9], args.slice_step, out / "slices_t2.png")
    boundary_overlay(rules, "t2", 0.5, args.slice_step, out / "boundaries_t2_0.50.png")
    boundary_overlay(rules, "t1", 0.3, args.slice_step, out / "boundaries_t1_0.30.png")

    keys = list(cubes)
    mat = pd.DataFrame([[disagreement(cubes[a], cubes[b]) for b in keys] for a in keys],
                       index=keys, columns=keys)
    mat.to_csv(out / "disagreement.csv", float_format="%.4f")
    print("Fraction of the cube where two rules disagree:\n" + mat.round(4).to_string())

    if args.frames:
        for r in rules:
            for axis in ("t1", "t2", "t3"):
                fdir = outdir("decision_regions", "frames", r, axis)
                for val in np.round(np.arange(0, 1.0001, 0.1), 2):
                    fig, ax = plt.subplots(figsize=(4, 4))
                    draw_slice(ax, r, axis, val, args.slice_step)
                    ax.grid(False)
                    ax.legend(handles=LEGEND, fontsize=6, loc="upper right")
                    save(fig, fdir / f"slice_{val:.1f}.png")


if __name__ == "__main__":
    main()
