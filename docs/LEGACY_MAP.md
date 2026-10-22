# Where did my old scripts go?

Every script from the three original folders, what it did, whether it was the
final version of its line of work, and what replaces it in this repository.
The originals are kept unchanged on the branch **`archive/original-code`**
(`git checkout archive/original-code`), grouped by folder and committed in the
order they were written.

Legend: **FINAL** = last working version of that line of work ·
*superseded* = earlier iteration of a FINAL script · *dup* = byte-identical
or near-identical copy · *dropped* = intentionally not ported (reason given).

## Which file was the "finalised" one?

| Line of work | Final original script | Replaced by |
|---|---|---|
| Fixed-sample error exponents, CPU | `MTP1/nvidia/error_expo/faster/2410_faster.py` | `experiments/error_exponents.py` |
| Fixed-sample error exponents, GPU | `MTP1/gpu.py` (= `nvidia/gpu/gpu1.py` with fewer trials, MMD disabled) | `experiments/error_exponents.py --backend gpu` |
| LRT / minimax comparison | `MTP1/MINMAX/minmax2_oct.py` | `error_exponents.py --tests lrt,...` |
| Report tables and plots | `MTP1/results_for_report/create_tables.py` | `summary.csv`, `efficiency.png` of each run |
| Decision regions (cube + frames) | `KL_frames/frames_full.py`, `MMD_frames/mmd_frames.py`, `GLRT_Frames/glt_frames_code.py`, `TVD_frames/tvd_frames.py` | `experiments/decision_regions.py` |
| Boundary comparison | `MTP1/boundary_lines/boundary.py` | `decision_regions.py` (boundary overlay) |
| Sequential NP-SEQ (internship) | `Intern_Summer/temp6.py` + `TEMP/tempfalse.py`, `TEMP/MisIdentification/tempmis.py` | `experiments/np_seq.py` |
| alpha study (MTP-2) | `MTP2/.../alpha_conv6.py` (noise floor), `alpha_conv7.py` (trade-off) | `experiments/alpha_study.py` |

## MTP1 (`Desktop/MTP/MTP1`)

| File | Status | Notes / replacement |
|---|---|---|
| `nvidia/error_expo/faster/2410_faster.py` | **FINAL (CPU)** | multiprocessing, P_e,max over all hypotheses. Final summary plot crashes (key mapping bug, see review). -> `error_exponents.py` |
| `nvidia/error_expo/2410.py` | superseded | same, no timing; `error_expo_vs_bhattacharya/2410_files/2410.py` is a *dup* |
| `error_expo_vs_bhattacharya/faster_code/2410faster.py` | *dup* | older copy of 2410_faster |
| `error_expo_vs_bhattacharya/code1.py, code2.py, code_temp.py` | superseded | only tests "outlier = sequence 0" (no P_e,max) |
| `error_expo_vs_bhattacharya/code3.py`, `nvidia/error_expo/code3.py`, `code_temp.py` | superseded | 5 tests, single hypothesis; newest timestamp but older logic |
| `error_expo_vs_bhattacharya/code4.py` | fragment | the "loop over k_true" patch later merged into 2410.py |
| `gpu.py` | **FINAL (GPU)** | CuPy, KL + GLT active -> `--backend gpu` |
| `nvidia/gpu/gpu1.py` | *dup* | gpu.py with 10 000 trials and MMD enabled (produced `10000_res_mmd_glrt_kl`) |
| `MINMAX/minmax2_oct.py` | **FINAL** | KL / TVD / full-sample MMD / LRT, 5 runs -> `error_exponents.py` |
| `MINMAX/minimax.py, MINMAX2.py, minmax.py` | superseded | "MMD" there is rbf on sample means (= closest pair) |
| `MINMAX/Diff_Distributions/temp1.py, temp2.py` | superseded | binomial alphabet -> `Problem(pi, mu, alphabet=...)` handles any finite alphabet |
| `MMD_KL_DB.py` | superseded | P_e vs L for KL / "MMD", compared with B and Chernoff -> `error_exponents.py` |
| `results_for_report/create_tables.py` | **FINAL** | tables / smooth plots -> written by every experiment |
| `KL_frames/frames_full.py` | **FINAL** | KL-to-centroid cube + slices -> `decision_regions.py --rules kl_centroid --frames` |
| `KL_frames/kl_cube_grouped_frames.py` | superseded | |
| `MMD_frames/mmd_frames.py` | **FINAL** | rbf on sample means = closest-pair rule |
| `MMD_frames/mmd_frames02.py` | superseded | per-scenario boundaries |
| `GLRT_Frames/glt_frames_code.py` | **FINAL** | GLT cube (plot 4 draws its tick labels on the wrong axes) |
| `TVD_frames/tvd_frames.py` | **FINAL** | TVD-to-centroid cube |
| `boundary_lines/boundary.py` | **FINAL** | neighbour-search boundaries (O(N^2)) -> exact image edges |
| `cube_full_code.py, cube_full_code_mmd.py, code_kl_cube_1.py, code_mmd_cube_1.py` | superseded | early cube versions |
| `scatter_kl.py, scatter_kl_edgecase.py, scatter_kl_edgecase_grouped.py` | superseded | first KL cube prototypes |
| `matlab_code_cube.m` | superseded | MATLAB port of the cube |
| `pythonimageviewer.py, pythonimageviewer_textbook.py, pyth_grouped_viewer.py` | superseded | `.npz` 3-D viewers -> `cube3d_*.png` |
| `temp_sum_method.py` | *dropped* | "geometric centroid" demo, see review (not a function of the types) |
| `Metrics_comparison/*` | *dup* | copy of `Intern_Summer/Metrics_comparison` |

## MTP2 (`Desktop/MTP/MTP2/Codes/alpha_convergence`)

| File | Status | Notes / replacement |
|---|---|---|
| `alpha1_conv.py` | superseded | n^a |M_u^2| for a = 0.5, 1, 1.5, one path |
| `alpha_conv2.py` | superseded | threshold "race" plot |
| `alpha_conv3.py` | *dup* | internship `temp7(main).py` + one trajectory plot |
| `alpha_conv4.py` | superseded | min-over-others statistic, one path, log-log fit |
| `alpha_conv5.py` | superseded | averaged noise floor, 15 runs |
| `alpha_conv6.py` | **FINAL** | noise floor, burn-in 200, 100 runs, slope ~ -1 -> `alpha_study.py` part 1 |
| `alpha_conv7.py` | **FINAL** | empirical alpha + trade-off curves -> `alpha_study.py` part 2 (max_steps = 300 there turned slow runs into errors) |
| `2601.00712v1.pdf` | reference | Geiger et al., mean/median tests -> `detectors/mean_median.py` |

## Internship (`Downloads/Intern_Summer`)

| File | Status | Notes / replacement |
|---|---|---|
| `temp6.py` | **FINAL** | Gaussian / multi-anomaly / uniform / KL NP-SEQ -> `np_seq.py` |
| `temp5.py` | superseded | |
| `paper2.py` | superseded | first NP-SEQ, used the paper's max-min statistic -> `statistic="maxmin"` |
| `temp.py, temp2.py, temp3.py, temp4.py` | superseded | NP-SEQ iterations (pooled statistic) |
| `TEMP/temp7(main).py` | **FINAL (single case)** | pooled NP-SEQ, alpha in {0.5, 1, 2} -> `statistic="pooled"` |
| `TEMP/temp01.py ... temp6.py, main.py, temp8.py, ALPHA1_2/code1.py` | superseded | C-grid explorations |
| `TEMP/MulMeans/temp9.py` | superseded | -> `np_seq.py --cases means` |
| `TEMP/UNIform/tempUni.py` | superseded | -> `np_seq.py --cases uniform` |
| `TEMP/KL/tempkl.py` | *dropped* | sequential histogram-KL; the plug-in estimator's bias depends on n and bins, so a C/n^a threshold has no clean calibration |
| `TEMP/tempfalse.py` | **FINAL** | false alarm -> `np_seq.py --cases multi` (L = 0) |
| `TEMP/MisIdentification/tempmis.py, TEMP/mis.py` | **FINAL** | misidentification, A = 1, 2 -> `run_np_seq_ua` |
| `TEMP/tempfss.py` | *dropped* | fixed-sample MMD with a C/sqrt(m) threshold; the fixed-sample analysis lives in `error_exponents.py` |
| `TEMP2/temp1.py` | *dropped* | NP-SEQ on a discrete alphabet; future work |
| `sample1.py` | superseded | first GL test on a 4-letter alphabet -> `detectors/glt.py` |
| `Metrics_comparison/temp*.py` | superseded | centroid / pooled / KL type statistics on a 3-letter alphabet |
| `Metrics_comparison/scatter*.py` | superseded | first 3-D type-space scatter -> `decision_regions.py` |
| `InternReport.pdf`, `Intern_Report.zip` | report | LaTeX source of the internship report |

## Reports

| File (in `~/Downloads`) | Status |
|---|---|
| `MTP1_Report.pdf` (27 Nov 2025, 67 pp.) | **FINAL** MTP-1 thesis |
| `MTP2_report_final.pdf` / `MTP2_final_report.pdf` (19 Apr 2026, 72 pp.) | **FINAL** MTP-2 thesis; the two files have the same text, different PDF encoding |
| `MTP2_Report_draft1*.pdf`, Turnitin files | drafts / plagiarism-check receipts |
| `Intern_Summer/InternReport.pdf` | internship report (LaTeX source on `archive/original-code`) |

Chapter 5 of the MTP-2 report reuses the MTP-1 results unchanged (same tables
and maximin numbers). The figures behind both reports came from
`results_for_report/` (fixed-sample tables, produced by `gpu1.py` runs) and
`MTP2/.../alpha_conv6.py` (Fig. 6.1, noise floor).
