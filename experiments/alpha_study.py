"""Experiment 5 (MTP-2) - why the NP-SEQ threshold should decay as C / n.

1. Noise floor: E|M_u^2| between two streams with the same law decays as
   n^-1 (log-log slope), and the quantiles of n * M_u^2 settle to constants.
2. Stopping time vs C: E[N] follows (C / MMD^2)^(1/alpha).
3. Trade-off E[N] vs -log10 P_err for a range of alpha.

Replaces MTP2/Codes/alpha_convergence/alpha1_conv.py ... alpha_conv7.py.

    python experiments/alpha_study.py --trials 3000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import outdir, plt, save, style

from universal_outlier.alpha import (expected_stop_theory, loglog_slope, null_noise_floor,
                                     paper_delay_bound)
from universal_outlier.mmd import population_mmd2_gaussians
from universal_outlier.sequential import SeqConfig, run_np_seq

ALPHAS = {0.5: [0.3, 0.5, 0.8, 1.2, 1.6, 2.0],
          0.75: [0.6, 1.0, 1.6, 2.4, 3.5, 4.5],
          1.0: [1, 2, 3, 4, 6, 8, 11],
          1.5: [2, 6, 12, 24, 40, 60],
          2.0: [5, 15, 40, 80, 150, 250]}
COLORS = {0.5: "#4c72b0", 0.75: "#55a868", 1.0: "#c44e52", 1.5: "#dd8452", 2.0: "#8172b3"}


def noise_floor(args, rng, out):
    n = np.unique(np.geomspace(20, args.n_max, 16).astype(int))
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for paired, ls in [(False, "-"), (True, "--")]:
        df = null_noise_floor(n, args.noise_trials, args.gamma, rng, paired=paired)
        df["paired"] = paired
        rows.append(df)
        burn = df[df.n >= 100]
        slope, icpt, r2 = loglog_slope(burn.n, burn.mean_abs)
        name = "paired" if paired else "unpaired"
        print(f"  {name:8s}: E|M_u^2| ~ n^{slope:.3f}  (R^2 = {r2:.4f})")
        axes[0].loglog(df.n, df.mean_abs, "o", ls="none", label=f"{name}: slope {slope:.3f}")
        axes[0].loglog(df.n, 10**icpt * df.n**slope, color="k", ls=ls, lw=1)
        for q, c in [("nM_q50", "#4c72b0"), ("nM_q90", "#dd8452"), ("nM_q99", "#c44e52")]:
            axes[1].semilogx(df.n, df[q], color=c, ls=ls, marker="o", ms=3,
                             label=f"{q[3:]} ({name})" if q else None)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$E|M_u^2|$ under $p=q$")
    axes[0].set_title("noise floor decays as 1/n")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"quantiles of $n\,M_u^2$")
    axes[1].set_title(r"$n\,M_u^2$ has a limit law $\Rightarrow$ $\alpha=1$ is the natural scale")
    axes[1].legend(fontsize=7, ncol=2)
    save(fig, out / "noise_floor.png")
    pd.concat(rows).to_csv(out / "noise_floor.csv", index=False, float_format="%.6g")


def tradeoff(args, rng, out):
    mmd2 = population_mmd2_gaussians(0, 1, 1.2, 1, args.gamma)
    print(f"  MMD^2(N(0,1), N(1.2,1)) = {mmd2:.4f}")
    frames = []
    for stat in ["pooled", "maxmin"]:
        cfg = SeqConfig(S=5, gamma=args.gamma, statistic=stat, max_steps=args.max_steps)
        for alpha, Cs in ALPHAS.items():
            df = run_np_seq(cfg, alpha, Cs, args.trials, rng)
            df["statistic"] = stat
            frames.append(df)
            print(f"  {stat:6s} alpha={alpha:<4g} E[N]: " + " ".join(f"{v:.1f}" for v in df.E_N)
                  + "   P_err: " + " ".join(f"{v:.4f}" for v in df.P_err), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "tradeoff.csv", index=False, float_format="%.6g")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for alpha in ALPHAS:
        s = df[(df.alpha == alpha) & (df.statistic == "pooled")]
        axes[0].loglog(s.C, s.E_N, "o", color=COLORS[alpha], label=rf"$\alpha$={alpha:g}")
        C = np.geomspace(s.C.min(), s.C.max(), 50)
        axes[0].loglog(C, expected_stop_theory(C, alpha, mmd2), color=COLORS[alpha], lw=1)
    axes[0].set_xlabel("C")
    axes[0].set_ylabel(r"$E[N]$")
    axes[0].set_title(r"pooled statistic: $E[N]$ vs $(C/\mathrm{MMD}^2)^{1/\alpha}$ (lines)", fontsize=9)
    axes[0].legend(fontsize=8)
    for ax, stat in zip(axes[1:], ["maxmin", "pooled"]):
        for alpha in ALPHAS:
            s = df[(df.alpha == alpha) & (df.statistic == stat)].sort_values("C")
            hit, zero = s[s.errors > 0], s[s.errors == 0]
            ax.plot(-np.log10(hit.P_err), hit.E_N, color=COLORS[alpha], lw=1.5, marker="o", ms=4,
                    label=rf"$\alpha$={alpha:g}")
            ax.scatter(-np.log10(zero.P_hi), zero.E_N, facecolors="none", edgecolors=COLORS[alpha],
                       marker=">", s=22)
        ax.set_xlabel(r"$-\log_{10} P_{err}$")
        ax.set_ylabel(r"$E[N]$")
        ax.set_title(f"trade-off, {stat} statistic (hollow >: no errors seen, lower bound)", fontsize=9)
        ax.legend(fontsize=8)
    save(fig, out / "alpha_tradeoff.png")

    x = np.linspace(0, 3.5, 20)
    bound = pd.DataFrame({"neg_log10_p": x, "paper_bound_E_N": paper_delay_bound(x, mmd2)})
    bound.to_csv(out / "paper_delay_bound.csv", index=False, float_format="%.6g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3000, help="NP-SEQ trials per (alpha, C)")
    ap.add_argument("--noise-trials", type=int, default=1000)
    ap.add_argument("--n-max", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()
    style()
    rng = np.random.default_rng(args.seed)
    out = outdir("alpha_study")
    if not args.skip_noise:
        print("Noise floor under the null:")
        noise_floor(args, rng, out)
    print("Sequential trade-off for several alpha:")
    tradeoff(args, rng, out)


if __name__ == "__main__":
    main()
