"""Experiment 4 - sequential NP-SEQ test: delay vs error trade-off.

Reproduces the NP-SEQ paper (Fig. 1) and the internship experiments:

* ``gaussian``   p = N(0,1), q = N(1.2,1)                     (paper Fig. 1a)
* ``laplace``    p = N(0,1), q = Laplace(0, 1/sqrt 2)         (paper Fig. 1b)
* ``uniform``    p = U(0,1), q = U(0.5, 1.5)                  (internship)
* ``means``      q = N(mu,1) for mu in {0.8, 1.2, 2.0}         (internship)
* ``statistic``  paper max-min statistic vs the pooled one-vs-rest statistic
* ``multi``      NP-SEQ-UA: misidentification for A = 1, 2 and false alarm

Every curve is E[N] against -log10 P_err as the threshold constant C varies.
Runs with zero observed errors are drawn as hollow ">" markers (lower bounds).

    python experiments/np_seq.py --cases gaussian,statistic --trials 5000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import outdir, plt, save, style

from universal_outlier.backend import BACKENDS, get_backend
from universal_outlier.mmd import population_mmd2_gaussians, population_mmd2_mc
from universal_outlier.sequential import SeqConfig, run_np_seq, run_np_seq_ua, sampler

N01 = {"dist": "normal", "loc": 0.0, "scale": 1.0}
ALPHA_STYLE = {0.5: ("#4c72b0", "s"), 1.0: ("#c44e52", "o"), 2.0: ("#8172b3", "^")}

# threshold constants chosen so -log10 P_err spans roughly 0.5 .. 3.5
C_GRID = {
    "gaussian": {0.5: [0.3, 0.5, 0.8, 1.2, 1.6, 2.0], 1.0: [1, 2, 3, 4, 6, 8, 11], 2.0: [5, 15, 40, 80, 150, 250]},
    "laplace": {0.5: [0.02, 0.04, 0.06, 0.09, 0.12], 1.0: [0.3, 0.6, 1, 1.5, 2.2, 3], 2.0: [5, 20, 60, 150, 300]},
    "uniform": {0.5: [0.2, 0.4, 0.7, 1.0, 1.4], 1.0: [0.5, 1, 2, 3, 4.5, 6], 2.0: [3, 10, 30, 70, 130]},
}


def tradeoff_plot(ax, df, color, marker, lbl):
    """E[N] vs -log10 P_err. Runs with zero errors only give a lower bound on
    -log10 P_err (the Wilson upper bound on P_err); they are drawn as hollow
    '>' markers and are not joined to the curve."""
    df = df.sort_values("C")
    hit = df[df.errors > 0]
    zero = df[df.errors == 0]
    ax.plot(-np.log10(hit.P_err), hit.E_N, color=color, lw=1.5, marker=marker, ms=5, label=lbl)
    ax.scatter(-np.log10(zero.P_hi), zero.E_N, facecolors="none", edgecolors=color, marker=">", zorder=3)


def case_alpha(name, q, p, args, xp, rng, out):
    mmd2 = population_mmd2_mc(lambda r, s: sampler(p)(r, s), lambda r, s: sampler(q)(r, s),
                              args.gamma, size=6000) if q["dist"] != "normal" or p["dist"] != "normal" \
        else population_mmd2_gaussians(p["loc"], p["scale"], q["loc"], q["scale"], args.gamma)
    frames = []
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for alpha, Cs in C_GRID[name].items():
        cfg = SeqConfig(S=args.S, p=p, q=q, gamma=args.gamma, statistic=args.statistic,
                        max_steps=args.max_steps)
        df = run_np_seq(cfg, alpha, Cs, args.trials, rng, xp)
        df.insert(0, "case", name)
        frames.append(df)
        print(df[["alpha", "C", "E_N", "P_err", "timeouts"]].to_string(index=False), flush=True)
        color, marker = ALPHA_STYLE[alpha]
        tradeoff_plot(ax, df, color, marker, rf"NP-SEQ($\alpha$={alpha:g})")
    ax.set_xlabel(r"$-\log_{10} P_{err}$")
    ax.set_ylabel(r"$E[N]$")
    ax.set_title(f"{name}: MMD$^2$(p,q) = {mmd2:.4f}, S = {args.S}, {args.statistic}", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, out / f"tradeoff_{name}.png")
    return pd.concat(frames)


def case_means(args, xp, rng, out):
    frames = []
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for mu, color in zip([0.8, 1.2, 2.0], ["#55a868", "#c44e52", "#4c72b0"]):
        q = {"dist": "normal", "loc": mu, "scale": 1.0}
        Cs = np.array([1, 2, 3, 4, 6, 8, 11]) * population_mmd2_gaussians(0, 1, mu, 1, args.gamma) / 0.224
        cfg = SeqConfig(S=args.S, q=q, gamma=args.gamma, statistic=args.statistic, max_steps=args.max_steps)
        df = run_np_seq(cfg, 1.0, Cs, args.trials, rng, xp)
        df.insert(0, "case", f"mean_{mu}")
        frames.append(df)
        tradeoff_plot(ax, df, color, "o", rf"$q=N({mu},1)$")
    ax.set_xlabel(r"$-\log_{10} P_{err}$")
    ax.set_ylabel(r"$E[N]$")
    ax.set_title(r"NP-SEQ($\alpha$=1): larger shift, faster detection", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, out / "tradeoff_means.png")
    return pd.concat(frames)


def case_statistic(args, xp, rng, out):
    frames = []
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for stat, color in [("maxmin", "#c44e52"), ("pooled", "#4c72b0")]:
        cfg = SeqConfig(S=args.S, gamma=args.gamma, statistic=stat, max_steps=args.max_steps)
        Cs = C_GRID["gaussian"][1.0]
        df = run_np_seq(cfg, 1.0, Cs, args.trials, rng, xp)
        df.insert(0, "case", f"statistic_{stat}")
        frames.append(df)
        tradeoff_plot(ax, df, color, "o", {"maxmin": "max-min (paper)", "pooled": "pooled one-vs-rest"}[stat])
    ax.set_xlabel(r"$-\log_{10} P_{err}$")
    ax.set_ylabel(r"$E[N]$")
    ax.set_title(r"Gaussian, $\alpha$=1: choice of test statistic", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, out / "tradeoff_statistic.png")
    return pd.concat(frames)


def case_multi(args, xp, rng, out):
    rows = []
    cfg = SeqConfig(S=args.S, gamma=args.gamma)
    for C in [0.8, 1.0, 1.2, 1.5]:
        for A, L in [(1, 1), (2, 1), (2, 2), (1, 0), (2, 0)]:
            r = run_np_seq_ua(cfg, A, L, C, max(args.trials // 5, 400), rng, xp=xp)
            rows.append(r)
            print(f"  C={C} A={A} L={L}: P_err={r['P_err']:.4f} E[N]={r['E_N']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (A, L), s in df[df.L > 0].groupby(["A", "L"]):
        x = -np.log10(s.P_err.clip(lower=1 / s.trials))
        axes[0].plot(x, s.E_N, marker="o", label=f"A={A}, true L={L}")
    axes[0].set_xlabel(r"$-\log_{10} P(\delta \neq I)$")
    axes[0].set_ylabel(r"$E[N]$")
    axes[0].set_title("NP-SEQ-UA misidentification", fontsize=9)
    axes[0].legend(fontsize=8)
    for A, s in df[df.L == 0].groupby("A"):
        axes[1].plot(s.C**2, -np.log10(s.P_err.clip(lower=1 / s.trials)), marker="o", label=f"A={A}")
    axes[1].set_xlabel(r"$C^2$")
    axes[1].set_ylabel(r"$-\log_{10} P_0(\delta \neq \emptyset)$")
    axes[1].set_title("false alarm (no anomaly present)", fontsize=9)
    axes[1].legend(fontsize=8)
    save(fig, out / "multi_anomaly.png")
    df.insert(0, "case", "multi")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="gaussian,laplace,uniform,means,statistic,multi")
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0, help="1/(2 sigma^2); paper sigma^2 = 0.5")
    ap.add_argument("--statistic", default="maxmin", choices=["maxmin", "pooled"])
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--backend", default="cpu", choices=BACKENDS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    style()
    rng = np.random.default_rng(args.seed)  # samples are drawn on the host ...
    xp, _ = get_backend(args.backend)  # ... and the MMD engine runs on xp
    out = outdir("np_seq")
    frames = []
    for case in args.cases.split(","):
        print(f"\n== {case}", flush=True)
        if case == "gaussian":
            frames.append(case_alpha(case, {"dist": "normal", "loc": 1.2, "scale": 1.0}, N01, args, xp, rng, out))
        elif case == "laplace":
            frames.append(case_alpha(case, {"dist": "laplace", "loc": 0.0, "scale": 1 / np.sqrt(2)}, N01,
                                     args, xp, rng, out))
        elif case == "uniform":
            frames.append(case_alpha(case, {"dist": "uniform", "low": 0.5, "high": 1.5},
                                     {"dist": "uniform", "low": 0.0, "high": 1.0}, args, xp, rng, out))
        elif case == "means":
            frames.append(case_means(args, xp, rng, out))
        elif case == "statistic":
            frames.append(case_statistic(args, xp, rng, out))
        elif case == "multi":
            frames.append(case_multi(args, xp, rng, out))
        else:
            raise SystemExit(f"unknown case {case!r}")
    pd.concat(frames, ignore_index=True).to_csv(out / f"np_seq_{'_'.join(args.cases.split(','))}.csv",
                                                index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
