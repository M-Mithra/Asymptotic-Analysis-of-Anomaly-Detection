import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import math
from scipy.spatial.distance import pdist, squareform, cdist

# Suppress minor warnings from the simulation for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# ===== HELPER FUNCTIONS FOR METRICS AND ANALYSIS
# ==============================================================================

def kl_divergence(p, q):
    """
    Calculates the Kullback-Leibler (KL) Divergence D_KL(p || q).
    - Inputs: p, q (numpy arrays representing discrete probability distributions).
    - Output: A single float value for the divergence.
    """
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def total_variation_distance(p, q):
    """
    Calculates the Total Variation Distance between two discrete distributions.
    - Inputs: p, q (numpy arrays representing discrete probability distributions).
    - Output: A single float value for the distance (between 0 and 1).
    """
    return 0.5 * np.sum(np.abs(p - q))

def bhattacharyya_distance(p, q):
    """
    Calculates the Bhattacharyya distance between two discrete distributions.
    - Inputs: p, q (numpy arrays representing discrete probability distributions).
    - Output: A single float value for the distance.
    """
    # Ensure p and q are the same length for comparison by padding with zeros
    len_p, len_q = len(p), len(q)
    if len_p > len_q:
        q = np.pad(q, (0, len_p - len_q), 'constant')
    elif len_q > len_p:
        p = np.pad(p, (0, len_q - len_p), 'constant')
    
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def mmd_u_statistic_1d(X, Y, gamma):
    """
    Calculates the unbiased MMD^2 U-statistic for 1D samples.
    This is a "full sample" MMD test, more powerful than comparing means.
    - Inputs: X, Y (1D numpy arrays of samples), gamma (kernel parameter).
    - Output: A single float value for the MMD^2 distance.
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0 # MMD is not well-defined for single-point distributions

    # Reshape 1D arrays to 2D column vectors for distance calculations
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    # Calculate squared Euclidean distances for the RBF kernel
    K_XX_dists = squareform(pdist(X, 'sqeuclidean'))
    K_YY_dists = squareform(pdist(Y, 'sqeuclidean'))
    K_XY_dists = cdist(X, Y, 'sqeuclidean')

    # Apply the RBF kernel
    K_XX = np.exp(-gamma * K_XX_dists)
    K_YY = np.exp(-gamma * K_YY_dists)
    K_XY = np.exp(-gamma * K_XY_dists)

    # Calculate the U-statistic
    term1 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) # Sum of off-diagonal elements
    term2 = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    term3 = K_XY.sum() * 2 / (n * m)
    
    # MMD^2 can be slightly negative due to estimator variance, clamp at 0
    return max(0, term1 + term2 - term3)

def estimate_alpha(lengths, error_probs):
    """
    Robustly estimates the error exponent alpha from simulation results.
    - Inputs: lengths (array of L), error_probs (list of Pe for each L).
    - Output: A single float value for the estimated alpha.
    """
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    if not valid_indices: return np.inf # Perfect performance
    if len(valid_indices) < 4: return np.nan # Not enough data for a stable fit

    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2: return np.nan
    
    fit_lengths = lengths[fit_indices]
    fit_log_p = np.log(np.array(error_probs)[fit_indices])
    
    try:
        slope, _ = np.polyfit(fit_lengths, fit_log_p, 1)
        return -slope
    except np.linalg.LinAlgError:
        return np.nan

# ==============================================================================
# ===== CORE SIMULATION FUNCTION
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, gamma, lengths, trials, batch_size):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario,
    testing all four metrics.
    """
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    errors = {'kl': [], 'mmd': [], 'tvd': [], 'lrt': []}

    for L in lengths:
        err_counts = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
        print(f"    Running L={L:<3}...", end="")
        for trial in range(trials):
            if (trial + 1) % batch_size == 0:
                print(f" {int(100*(trial+1)/trials)}%", end="", flush=True)

            seq1 = np.random.binomial(1, p_mu, L)
            seq2 = np.random.binomial(1, p_pi, L)
            seq3 = np.random.binomial(1, p_pi, L)
            
            # --- KL & TVD Test (based on empirical distributions) ---
            P1 = np.bincount(seq1, minlength=2) / L
            P2 = np.bincount(seq2, minlength=2) / L
            P3 = np.bincount(seq3, minlength=2) / L
            D_kl = np.array([(kl_divergence(P1,P2)+kl_divergence(P1,P3))/2, (kl_divergence(P2,P1)+kl_divergence(P2,P3))/2, (kl_divergence(P3,P1)+kl_divergence(P3,P2))/2])
            D_tvd = np.array([(total_variation_distance(P1,P2)+total_variation_distance(P1,P3))/2, (total_variation_distance(P2,P1)+total_variation_distance(P2,P3))/2, (total_variation_distance(P3,P1)+total_variation_distance(P3,P2))/2])
            if np.argmax(D_kl) != 0: err_counts['kl'] += 1
            if np.argmax(D_tvd) != 0: err_counts['tvd'] += 1

            # --- MMD Test (New "Full Sample" Logic) ---
            D_mmd = np.array([
                (mmd_u_statistic_1d(seq1, seq2, gamma) + mmd_u_statistic_1d(seq1, seq3, gamma))/2,
                (mmd_u_statistic_1d(seq2, seq1, gamma) + mmd_u_statistic_1d(seq2, seq3, gamma))/2,
                (mmd_u_statistic_1d(seq3, seq1, gamma) + mmd_u_statistic_1d(seq3, seq2, gamma))/2
            ])
            if np.argmax(D_mmd) != 0: err_counts['mmd'] += 1
            
            # --- LRT (Optimal Test) ---
            c1,c2,c3 = np.sum(seq1), np.sum(seq2), np.sum(seq3)
            D_lrt = np.array([c1*llr_if_one+(L-c1)*llr_if_zero, c2*llr_if_one+(L-c2)*llr_if_zero, c3*llr_if_one+(L-c3)*llr_if_zero])
            if np.argmax(D_lrt) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        for key in errors: errors[key].append(err_counts[key] / trials)
        
    return {name: np.array(error_probs) for name, error_probs in errors.items()}

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- SETTINGS ---
GAMMA = 1.0 # Gamma for MMD kernel
NUM_TRIALS = 1000
BATCH_SIZE = 200
SEQUENCE_LENGTHS = np.arange(50, 251, 50) # Longer sequences for hard cases
NUM_RUNS = 5 # Number of times to run the whole experiment for statistical stability

scenarios = {
    "Easy (0.2 vs 0.8)": (0.2, 0.8),
    "Medium (0.3 vs 0.7)": (0.3, 0.7),
    "Hard (0.4 vs 0.6)": (0.4, 0.6),
    "Very Hard (0.45 vs 0.55)": (0.45, 0.55),
}

save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- EXPERIMENT LOOP (MULTIPLE RUNS) ---
all_run_summaries = []
print(f"Starting comparative experiment with {NUM_RUNS} runs. This will take a very long time...")

for run in range(NUM_RUNS):
    print(f"\n{'#'*70}\n########## STARTING RUN {run + 1} / {NUM_RUNS} ##########\n{'#'*70}")
    run_results = []
    for name, (p_pi, p_mu) in scenarios.items():
        print(f"\n--- Running Scenario: {name} ---")
        error_data = run_simulation_for_scenario(p_pi, p_mu, GAMMA, SEQUENCE_LENGTHS, NUM_TRIALS, BATCH_SIZE)
        
        pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
        b_dist = bhattacharyya_distance(pi_dist, mu_dist)
        
        run_results.append({
            "Scenario": name, "2*DB_Bound": 2*b_dist,
            "alpha_LRT": estimate_alpha(SEQUENCE_LENGTHS, error_data['lrt']),
            "alpha_KL": estimate_alpha(SEQUENCE_LENGTHS, error_data['kl']),
            "alpha_MMD": estimate_alpha(SEQUENCE_LENGTHS, error_data['mmd']),
            "alpha_TVD": estimate_alpha(SEQUENCE_LENGTHS, error_data['tvd'])
        })
    all_run_summaries.append(pd.DataFrame(run_results))

# --- FINAL ANALYSIS AND REPORTING ---
print("\n--- All Runs Finished. Aggregating final results... ---")
# Combine all run summaries into one big DataFrame
full_summary_df = pd.concat(all_run_summaries)

# Calculate efficiency for each run
for test in ['LRT', 'KL', 'MMD', 'TVD']:
    full_summary_df[f'Efficiency_{test}'] = np.minimum(1.0, full_summary_df[f'alpha_{test}'] / full_summary_df['2*DB_Bound'])

# Aggregate to find mean and standard deviation across all runs
final_agg_df = full_summary_df.groupby('Scenario').agg(['mean', 'std'])

# Calculate maximin results based on the mean efficiencies
mean_efficiencies = final_agg_df.xs('mean', axis=1, level=1)
min_efficiencies = {
    'LRT (Optimal)': mean_efficiencies['Efficiency_LRT'].min(skipna=True),
    'KL Divergence': mean_efficiencies['Efficiency_KL'].min(skipna=True),
    'MMD (Full Sample)': mean_efficiencies['Efficiency_MMD'].min(skipna=True),
    'Total Variation': mean_efficiencies['Efficiency_TVD'].min(skipna=True)
}
best_test = max(min_efficiencies, key=lambda k: min_efficiencies.get(k, 0))

# --- SAVE AND DISPLAY RESULTS ---
excel_path = os.path.join(save_folder, 'aggregated_experiment_results.xlsx')
final_agg_df.to_excel(excel_path, float_format="%.4f")

print("\n" + "="*60); print("       AGGREGATED EXPERIMENT: FINAL SUMMARY"); print("="*60)
print(f"Aggregated results (mean/std over {NUM_RUNS} runs) saved to: {os.path.basename(excel_path)}")
print("\n--- Final Mean Efficiency Table ---")
print(mean_efficiencies[[col for col in mean_efficiencies.columns if 'Efficiency' in col]].to_string(float_format="%.4f"))
print("\n--- Maximin Results (Based on Mean Efficiencies) ---")
print("Worst-Case Efficiency Ratio (min alpha / 2*DB):")
for test, value in min_efficiencies.items(): print(f"- {test:<19}: {value:.4f}")
print("-" * 60); print(f"CONCLUSION (Maximin): The most robust test is '{best_test}'"); print("="*60)

# --- CREATE AND SAVE FINAL SUMMARY PLOTS ---
# Plot 1: Efficiency vs. Difficulty Line Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
x_axis = mean_efficiencies['2*DB_Bound']
test_metrics = {'LRT (Optimal)': 'LRT', 'KL Divergence': 'KL', 'MMD (Full Sample)': 'MMD', 'Total Variation': 'TVD'}
colors = {'LRT (Optimal)': 'red', 'KL Divergence': 'blue', 'MMD (Full Sample)': 'green', 'Total Variation': 'purple'}

for long_name, short_name in test_metrics.items():
    mean_eff = mean_efficiencies[f'Efficiency_{short_name}']
    std_eff = final_agg_df[('Efficiency_' + short_name, 'std')]
    ax_summary.plot(x_axis, mean_eff, label=long_name, color=colors[long_name], marker='o', linestyle='-', markersize=8)
    ax_summary.fill_between(x_axis, mean_eff - std_eff, mean_eff + std_eff, color=colors[long_name], alpha=0.15, label=f'_nolegend_')

ax_summary.invert_xaxis()
ax_summary.axhline(y=1.0, color='black', linestyle='--', label='Theoretical Maximum')
ax_summary.set_xlabel('Problem Difficulty (2 * Bhattacharyya Distance -> smaller is harder)', fontsize=12)
ax_summary.set_ylabel('Mean Test Efficiency Ratio (alpha / 2*DB)', fontsize=12)
ax_summary.set_title(f'Test Efficiency vs. Difficulty (Mean of {NUM_RUNS} Runs)', fontsize=16)
ax_summary.legend(fontsize=11); ax_summary.set_ylim(0, 1.1); plt.tight_layout()
plot_path = os.path.join(save_folder, 'efficiency_comparison_plot.png')
plt.savefig(plot_path, dpi=300); plt.show()
print(f"\nSummary plot with error bands saved to: {plot_path}")

# Plot 2: Heatmap
heatmap_data = mean_efficiencies[[f'Efficiency_{short_name}' for short_name in test_metrics.values()]].T
heatmap_data.index = test_metrics.keys()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
plt.title('Heatmap of Mean Test Efficiencies', fontsize=16)
plt.xlabel('Scenario (Difficulty increasing to the right)')
plt.ylabel('Test Method')
plt.tight_layout()
heatmap_path = os.path.join(save_folder, 'efficiency_heatmap.png')
plt.savefig(heatmap_path, dpi=300); plt.show()
print(f"Heatmap summary saved to: {heatmap_path}")