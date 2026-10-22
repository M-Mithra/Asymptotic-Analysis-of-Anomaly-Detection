"""Experiment 2 - theoretical (Sanov) exponents, and how simulation meets them.

1. Large-deviation exponent of every rule for each Bernoulli scenario,
   overlaid on the simulated efficiencies of ``error_exponents.py``.
2. Exponent ratio E / 2B against the number of sequences M (local regime).
3. Exact P_e(n): its local slopes converge to the Sanov exponent.

    python experiments/theory_exponents.py                 # uses results/error_exponents/quick
    python experiments/theory_exponents.py --sim standard
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import COLORS, MARKERS, RESULTS, label, outdir, plt, save, style

from universal_outlier.exponents import local_slopes
from universal_outlier.fixed_sample import exact_pe_curve
from universal_outlier.problem import Problem, bernoulli_scenarios
from universal_outlier.theory import sanov_exponent

RULES = ["lrt", "glt", "kl_avg", "kl_centroid", "mmd_avg", "closest_pair", "mean_based", "median_based"]


def scenario_table(preset: str, grid: int) -> pd.DataFrame:
    rows = []
    scen = {name: sc["p"] for name, sc in bernoulli_scenarios(preset).items()}
    # asymmetric cases, where the rules separate
    scen.update({"asym_0.10_0.30": (0.10, 0.30), "asym_0.05_0.25": (0.05, 0.25),
                 "asym_0.70_0.90": (0.70, 0.90)})
    for name, (a, b) in scen.items():
        pr = Problem.bernoulli(a, b)
        row = {"scenario": name, "p_pi": a, "p_mu": b, "tvd": pr.tvd, "two_b": pr.two_b}
        for d in RULES:
            row[f"E_{d}"], _ = sanov_exponent(d, pr, grid=grid)
            row[f"ratio_{d}"] = row[f"E_{d}"] / pr.two_b
        rows.append(row)
        print(f"  {name:16s} " + " ".join(f"{d}={row[f'ratio_{d}']:.3f}" for d in RULES), flush=True)
    return pd.DataFrame(rows)


def plot_overlay(theory: pd.DataFrame, sim: pd.DataFrame | None, path):
    sym = theory[theory.scenario.str.startswith("TVD")].sort_values("tvd")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in ["glt", "kl_avg", "mmd_avg", "mean_based"]:
        ax.plot(sym.tvd, sym[f"ratio_{d}"], color=COLORS[d], lw=2, label=f"{label(d)} (theory)")
        if sim is not None and f"ratio_{d}" in sim:
            ax.errorbar(sim.tvd, sim[f"ratio_{d}"], yerr=sim[f"se_{d}"] / sim.two_b, ls="none",
                        marker=MARKERS[d], color=COLORS[d], capsize=2, label=f"{label(d)} (simulated)")
    ax.axhline(1, color="k", ls="--", lw=1, label="LRT / optimum 2B")
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.text(0.97, 1 / 3 - 0.05, "1/3", color="gray", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"TVD$(\pi,\mu)$")
    ax.set_ylabel(r"exponent / $2B(\pi,\mu)$")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, ncol=2)
    save(fig, path)


def ratio_vs_M(path_csv, path_png):
    settings = {3: (121, 41), 4: (41, 21), 5: (19, 11)}
    rows = []
    for M, (grid, rp) in settings.items():
        pr = Problem.bernoulli(0.45, 0.55, M=M)
        for d in ["glt", "kl_avg", "mmd_avg", "mean_based"]:
            e, _ = sanov_exponent(d, pr, grid=grid, refine=6, refine_points=rp)
            rows.append({"M": M, "detector": d, "ratio": e / pr.two_b})
        print(f"  M={M}: " + " ".join(f"{r['detector']}={r['ratio']:.3f}" for r in rows if r["M"] == M), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    Ms = np.arange(3, 21)
    ax.plot(Ms, (Ms - 2) / Ms, "k:", label=r"$(M-2)/M$")
    for d, s in df.groupby("detector"):
        ax.plot(s.M, s.ratio, marker=MARKERS[d], color=COLORS[d], ls="none", ms=7, label=label(d))
    ax.set_xlabel("number of sequences M")
    ax.set_ylabel(r"exponent / $2B$   (local regime)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    save(fig, path_png)


def exact_convergence(path_csv, path_png):
    pr = Problem.bernoulli(0.4, 0.6)
    rules = ["glt", "kl_avg", "mmd_avg"]
    n = np.arange(100, 1601, 100)
    df = exact_pe_curve(pr, rules, n)
    df.to_csv(path_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d in rules:
        s = df[df.detector == d]
        axes[0].semilogy(s.n, s.pe_avg, marker=MARKERS[d], color=COLORS[d], label=label(d))
        mid, slope = local_slopes(s.n, s.pe_avg)
        axes[1].plot(mid, slope / pr.two_b, marker=MARKERS[d], color=COLORS[d], label=f"{label(d)} local slope")
        e, _ = sanov_exponent(d, pr)
        axes[1].axhline(e / pr.two_b, color=COLORS[d], ls="--", lw=1)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"exact $P_e(n)$")
    axes[0].legend(fontsize=8)
    axes[0].set_title(r"$\pi$=Bern(0.4), $\mu$=Bern(0.6), M=3")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$-\Delta \ln P_e / \Delta n \;/\; 2B$")
    axes[1].set_title("local slope -> Sanov exponent (dashed)")
    axes[1].legend(fontsize=8)
    save(fig, path_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="standard", choices=["quick", "standard"])
    ap.add_argument("--sim", default="quick", help="error_exponents run to overlay")
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--skip-exact", action="store_true")
    args = ap.parse_args()
    style()
    out = outdir("theory")

    print("Sanov exponents per scenario (ratio to 2B):")
    theory = scenario_table(args.preset, args.grid)
    theory.to_csv(out / "sanov_exponents.csv", index=False, float_format="%.6g")
    sim_path = RESULTS / "error_exponents" / args.sim / "summary.csv"
    sim = pd.read_csv(sim_path) if sim_path.exists() else None
    plot_overlay(theory, sim, out / "theory_vs_simulation.png")

    print("Exponent ratio vs M:")
    ratio_vs_M(out / "ratio_vs_M.csv", out / "ratio_vs_M.png")
    if not args.skip_exact:
        print("Exact P_e convergence (takes a few minutes)...")
        exact_convergence(out / "exact_pe.csv", out / "exact_convergence.png")


if __name__ == "__main__":
    main()
