# Theory notes

## 1. Problem

M sequences of length n over a finite alphabet. One (the outlier) is i.i.d. μ,
the other M − 1 are i.i.d. π; neither is known. A test δ names the outlier.

* P_e(k) = P(δ ≠ k | outlier is k), P_e,max = max_k P_e(k)
* error exponent  E(δ) = lim −(1/n) ln P_e,max(n)
* optimum when π and μ are known (likelihood-ratio test):  **2B(π, μ)**,
  B = −ln Σ_y √(π(y) μ(y)) (Li, Nitinawarat & Veeravalli 2014, Prop. 7).

All rules in `detectors/` are permutation invariant, so P_e(k) is the same
for every k and P_e,max = P_e(0).

## 2. The rules

| key | rule (declare argmax of score) |
|---|---|
| `glt` | −Σ_{j≠i} D(P_j ‖ mean_{l≠i} P_l)  (universal GLT, Li et al. eq. 37) |
| `kl_avg` | mean_{j≠i} D(P_i ‖ P_j) |
| `kl_centroid` | D(P_i ‖ mean_{j≠i} P_j) |
| `mmd_avg` / `mmd_maxmin` | mean / min over j≠i of M_u²(i, j) (Gaussian kernel) |
| `closest_pair` | min_{j≠i} \|mean_i − mean_j\| |
| `tvd_avg` / `tvd_centroid` | TVD analogues |
| `mean_based` / `median_based` | D(P_i ‖ π̂), π̂ = mean / normalised median of all types (Geiger et al. 2026) |
| `lrt` | Σ_y P_i(y) ln(μ(y)/π(y)) — oracle, knows π and μ |

For Bernoulli data and M = 3, `mmd_avg`, `mmd_maxmin`, `closest_pair` and
`tvd_centroid` are *the same partition of type space* in the n → ∞ limit
(the MMD between Bernoulli laws is 2(1 − e^{−γ})(t_i − t_j)², and for three
points "largest average squared distance" = "left out of the closest pair").

## 3. Error exponents from the cube (Sanov's theorem)

The types (T_0, …, T_{M−1}) obey a large-deviation principle with rate
I(t) = D(t_0‖μ) + Σ_{j≥1} D(t_j‖π). A rule that decides from the types has

    E(rule) = min { I(t) : t in the closure of the rule's error region }.

For Bernoulli data the error region is exactly a coloured region of the
(t1, t2, t3) cube in `experiments/decision_regions.py`; the exponent is the
KL "distance" from the true point (μ, π, π) to the nearest wrong region.
`universal_outlier.theory.sanov_exponent` evaluates this on a refined grid.
Check: for `lrt` it returns 2B to 4 digits.

### Why every universal rule gets 1/3 of the optimum for M = 3

In the local regime (π close to μ, δ = μ(1) − π(1), V = π(1)(1 − π(1)))
D(t‖p) ≈ (t − p)² / (2V), and all symmetric rules reduce to the closest-pair
rule on the line. An error needs the three points to become equally spaced
(the outlier stops being the one left out). Minimising

    (x0 − δ)² + x1² + x2²   subject to   x2 = (x0 + x1)/2

gives x0 = δ − δ/6, x1 = −δ/6, x2 = δ/3 and the value δ²/6, so
E ≈ δ²/(12V). The optimum is 2B ≈ δ²/(4V). Ratio: **exactly 1/3**. The grid
search finds the same minimiser, t* = (μ − δ/6, π − δ/6, π + δ/3).

Numerically the local-regime ratio is (M − 2)/M for M = 3, 4, 5
(`results/theory/ratio_vs_M.csv`), approaching 1 as M grows — consistent with
Li et al., whose GLT bound tends to 2B as M → ∞. **So the efficiencies of
≈ 0.3–0.4 found in the MTP-1 simulations are the correct answer for M = 3,
not a deficiency of the code.**

Away from the local regime the rules separate. For asymmetric pairs, e.g.
(π, μ) = (Bern 0.1, Bern 0.3) vs (Bern 0.7, Bern 0.9), Max-Avg-KL ranges from
0.58 to 0.14 of 2B while the GLT stays within 0.32–0.35: the GLT is the
maximin (worst-case) choice, as Li et al. predict.

### Convergence of simulated slopes

P_e(n) ≈ K n^{−β} e^{−E n}, so the slope of ln P_e between n₁ and n₂ equals
E + O(1/n). `results/theory/exact_convergence.png` computes P_e(n) exactly
(no Monte-Carlo noise) and shows the local slopes approaching the Sanov value
from above; simulated exponents at moderate n are therefore a few percent
high, never low.

## 4. Sequential test NP-SEQ and the threshold exponent α (MTP-2)

NP-SEQ stops at the first n with Γ_n > T_n = C / n^α (Sreenivasan & Bhashyam
2021), Γ_n = max_i min_{j≠i} M_u²(i, j, n).

**Noise floor.** If streams i and j are both typical, M_u²(i, j, n) is a
*degenerate* U-statistic: E = 0 and n·M_u² → Σ_k λ_k (Z_k² − 1) in
distribution (Gretton et al. 2012, Thm 12). The noise floor therefore decays
as exactly 1/n. `results/alpha_study/noise_floor.png`: log-log slope
−0.98 (unpaired) / −1.02 (paired), and the quantiles of n·M_u² are flat in n.

**Signal.** For the anomalous stream Γ_n → MMD²(p, q) > 0, so the test stops
near N ≈ (C / MMD²)^{1/α}. For α = 1 this is Theorem 3 of the paper
(N / C → 1 / MMD²) and the simulations follow it (`alpha_tradeoff.png`, left).

**Heuristic trade-off** (scaling argument, not a proof). Write W for the
limit of n·(noise). An error needs the noise of a wrong candidate to cross
the threshold before the signal does.

* α < 1: the threshold falls slower than the noise, so errors come from the
  first few samples: −ln P_err ∝ C, but E[N] ∝ C^{1/α}, i.e. E[N] grows
  *superlinearly* in log(1/P_err). Clearly worse — visible for α = 0.5.
* α = 1: threshold and noise have the same scale; −ln P_err ∝ C and
  E[N] ≈ C / MMD² — linear, i.e. exponentially consistent.
* α > 1: the noise eventually beats the threshold, at n ~ C^{1/(α−1)}; errors
  are decided near the stopping time and −ln P_err ∝ C^{1/α}, which again
  gives E[N] linear in log(1/P_err) with the same first-order constant.

So α = 1 is the natural, scale-matched choice and α < 1 is strictly worse;
α > 1 matches α = 1 to first order. The simulations agree: α ∈ [0.75, 2]
give nearly the same curves over the range 5000 trials can resolve, while
α = 0.5 is clearly worse. The paper's own Fig. 1a shows α = 2 only slightly
worse than α = 1.

## References

1. Y. Li, S. Nitinawarat, V. V. Veeravalli, "Universal outlier hypothesis testing", IEEE Trans. IT, 2014.
2. S. C. Sreenivasan, S. Bhashyam, "Sequential nonparametric detection of anomalous data streams", IEEE SPL, 2021.
3. B. C. Geiger, T. Koch, J. Mihaljević, M. Toller, "Universal outlier hypothesis testing via mean- and median-based tests", arXiv:2601.00712, 2026.
4. A. Gretton et al., "A kernel two-sample test", JMLR, 2012.
