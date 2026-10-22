"""Experiment 1 - fixed-sample error exponents of universal outlier tests.

Bernoulli data, M sequences, one outlier. For each scenario (pi, mu) and
sequence length n, P_e(n) is estimated by Monte Carlo; the exponent c is the
slope of ln P_e(n) and is compared with the optimum 2 B(pi, mu).

Replaces the project scripts 2410.py / 2410_faster.py / gpu.py / code*.py /
MINMAX/*.py. Examples::

    python experiments/error_exponents.py --preset quick
    python experiments/error_exponents.py --preset standard --trials 1000000
    python experiments/error_exponents.py --tests glt,mmd_avg --name mmd_vs_glt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import COLORS, MARKERS, label, outdir, plt, save, style

from universal_outlier.backend import BACKENDS, get_backend
from universal_outlier.exponents import fit_exponent
from universal_outlier.fixed_sample import monte_carlo_pe
from universal_outlier.problem import Problem, bernoulli_scenarios

DEFAULT_TESTS = "lrt,glt,kl_avg,mmd_avg,closest_pair"


def run(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    xp, rng = get_backend(args.backend, args.seed)
    tests = args.tests.split(",")
    scenarios = bernoulli_scenarios(args.preset, grid=args.grid)
    curves, summary = [], []
    for name, sc in scenarios.items():
        problem = Problem.bernoulli(*sc["p"], M=args.M, gamma=args.gamma, name=name)
        print(f"\n== {name}: pi = Bern({sc['p'][0]}), mu = Bern({sc['p'][1]}), "
              f"2B = {problem.two_b:.5f}", flush=True)
        df = monte_carlo_pe(problem, tests, sc["n"], args.trials, rng, xp, verbose=True)
        df.insert(0, "scenario", name)
        curves.append(df)
        row = {"scenario": name, "p_pi": sc["p"][0], "p_mu": sc["p"][1],
               "tvd": problem.tvd, "two_b": problem.two_b}
        for key in tests:
            s = df[df.detector == key]
            fit = fit_exponent(s.n, s.pe_avg, trials=args.trials * args.M, method=args.fit)
            row[f"c_{key}"] = fit.c
            row[f"se_{key}"] = fit.se
            row[f"ratio_{key}"] = fit.c / problem.two_b
        summary.append(row)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame, tests, path):
    names = list(summary.scenario)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for key in tests:
            s = curves[(curves.scenario == name) & (curves.detector == key)]
            ax.semilogy(s.n, s.pe_avg.where(s.pe_avg > 0), marker=MARKERS[key], ms=3,
                        color=COLORS[key], label=label(key))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("n")
    axes.flat[0].set_ylabel(r"$P_e(n)$")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    save(fig, path)


def plot_efficiency(summary: pd.DataFrame, tests, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in tests:
        ax.errorbar(summary.tvd, summary[f"ratio_{key}"], yerr=summary[f"se_{key}"] / summary.two_b,
                    marker=MARKERS[key], color=COLORS[key], capsize=2, label=label(key))
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"optimum $2B(\pi,\mu)$")
    ax.set_xlabel(r"TVD$(\pi,\mu)$  (larger = easier)")
    ax.set_ylabel(r"efficiency  $\hat c\,/\,2B$")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="quick", choices=["quick", "standard"])
    ap.add_argument("--grid", default="exponent", choices=["exponent", "legacy"],
                    help="sequence lengths: n*2B in [4, 30] (exponent) or the original lengths (legacy)")
    ap.add_argument("--tests", default=DEFAULT_TESTS, help="comma-separated detector keys")
    ap.add_argument("--trials", type=int, default=200_000, help="Monte-Carlo trials per hypothesis")
    ap.add_argument("--M", type=int, default=3, help="number of sequences")
    ap.add_argument("--gamma", type=float, default=1.0, help="RBF kernel parameter")
    ap.add_argument("--fit", default="wls", choices=["wls", "half_tail"])
    ap.add_argument("--backend", default="cpu", choices=BACKENDS)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--name", default=None, help="output folder name (default: preset[_legacy])")
    args = ap.parse_args()

    style()
    default_name = args.preset + ("_legacy" if args.grid == "legacy" else "")
    out = outdir("error_exponents", args.name or default_name)
    t0 = time.perf_counter()
    curves, summary = run(args)
    curves.to_csv(out / "pe_curves.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    tests = args.tests.split(",")
    plot_curves(curves, summary, tests, out / "pe_curves.png")
    plot_efficiency(summary, tests, out / "efficiency.png")

    print("\nEfficiency c / 2B (worst case over scenarios = maximin criterion):")
    for key in tests:
        r = summary[f"ratio_{key}"]
        print(f"  {label(key):<20s} min {r.min():.3f}   median {r.median():.3f}")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
