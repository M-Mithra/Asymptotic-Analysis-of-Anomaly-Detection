# Corrections and methodological changes relative to the original scripts

The rewrite keeps every test and experiment of the original project but fixes
the issues below. Each item says what was wrong, its effect, and the fix.

## Correctness

1. **Biased tie-breaking.** With discrete data, scores tie often (e.g. equal
   counts). `np.argmax` resolves ties to the lowest index. In scripts that
   fix the outlier at index 0 (`code1–3.py`, `MINMAX/*.py`, `MMD_KL_DB.py`)
   every tie counted as a correct decision, so P_e was *underestimated*; in
   the P_e,max scripts it inflated P_e(k) for k ≠ 0 instead.
   **Fix:** ties are broken uniformly at random, and the engine uses the exact
   conditional error of that rule (`decision.error_given_scores`).

2. **Summary-plot crash in `2410_faster.py`.** The plot looked up keys with
   `list(TEST_FUNCTIONS.keys())[list(SETTINGS['tests'].keys()).index(name)]`.
   When the two dicts differ (they did: {kl, mmd, pair, glt} vs {KL, GLT}),
   "Universal GLT" maps to `mmd` and the script raises `KeyError` after the
   whole simulation. MMD and closest-pair were also computed but never
   reported. **Fix:** a single detector registry keyed by name.

3. **Efficiency capped at 1.** `np.minimum(1.0, c / 2B)` hid estimates above
   the optimum, which signal estimation error. **Fix:** ratios are reported
   uncapped with a standard error.

4. **Convex hulls over non-convex regions.** The "solid shape" cube plots
   wrapped DBSCAN clusters in convex hulls, but the decision regions are not
   convex (X-shaped in each slice), so hulls painted over neighbouring
   regions. **Fix:** regions drawn exactly as label images. (Also: plot 4 of
   `glt_frames_code.py` drew its tick labels on the plot-3 axes.)

5. **Time-outs counted as errors in the α trade-off.** `alpha_conv7.py`
   capped each run at 300 steps and counted unfinished runs as errors, which
   bent the α = 0.5 curve; zero-error points were plotted at the 1/trials floor
   as if measured. **Fix:** a 5000-step cap (time-outs reported separately) and
   zero-error points drawn as lower bounds.

## Methodology

6. **Exponent estimation.** OLS through ln P_e over the second half of the
   non-zero points gave equal weight to points with 1–5 errors, which made
   the c / 2B curves jagged (the "dip" near TVD 0.20–0.25). **Fix:** weighted
   least squares on points with ≥ 10 errors and standard errors; plus exact
   P_e(n) and the Sanov exponents as ground truth.

7. **Sequence-length grid.** The original lengths grew like 1/TVD, but the
   exponent 2B shrinks like TVD², so the hard scenarios only reached
   n·2B ≈ 4 — far from the asymptotic regime — and their fitted exponents came
   out too high (the rise of c / 2B towards 0.4–0.5 on the hard side). The new
   default grid spaces n so that n·2B covers [4, 30] in every scenario (up to
   n ≈ 75 000, free with the counts engine). Worst-case GLT efficiency:
   0.321 (legacy grid) vs 0.330 (new grid), theory 0.327–0.333;
   compare `results/error_exponents/standard` and `standard_legacy`.

8. **P_e,max as a max of noisy estimates** is biased upwards. All rules are
   permutation invariant, so the mean over hypotheses estimates the same
   quantity with lower variance; both are reported.

9. **NP-SEQ statistic.** The internship and MTP-2 code used
   M_u²(stream i, all other streams pooled), while the paper's
   test is max_i min_{j≠i} M_u²(i, j). Both are implemented
   (`statistic="maxmin"` default, `"pooled"` for the original runs) and
   compared in `results/np_seq/tradeoff_statistic.png`.

10. **Names that did not match the computation.** In the cube, frame and
   `MINMAX/minimax.py` scripts "MMD" was rbf(t_i, t_j) on sample means, which is
   the closest-pair rule (for M = 3 it coincides with the true MMD rule, and γ
   has no effect on the decision). The "geometric centroid" rule averaged raw
   sequences position by position, so it depended on how samples were
   aligned in time, not on their distributions; it is dropped.

## Speed

| task | original | now |
|---|---|---|
| fixed-sample P_e(n), one n, 10⁴ trials | O(n²) kernel matrices per trial, multiprocessing / GPU | O(M²k) per trial from multinomial counts: ~10⁶ trials/s on a laptop CPU |
| NP-SEQ step | all MMDs recomputed, O(S² n²) | incremental block sums, O(S² n), vectorised over trials |
| full standard exponent sweep | hours (GPU) | minutes (CPU) |
