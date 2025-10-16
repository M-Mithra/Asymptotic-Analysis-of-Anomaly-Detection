import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import math

# Suppress minor warnings from the simulation
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# ===== HELPER FUNCTIONS FOR METRICS AND ANALYSIS
# ==============================================================================

def kl_divergence(p, q):
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * (x - y)**2)

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def bhattacharyya_distance(p, q):
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def estimate_alpha(lengths, error_probs):
    """Robustly estimates the error exponent alpha from simulation results."""
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    if len(valid_indices) < 4: # Require at least 4 data points for a more stable fit
        return np.nan
    
    # Use the latter half of the data for a more stable asymptotic fit
    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2:
        return np.nan

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
    """Runs a full Monte Carlo simulation for a single (pi, mu) scenario."""
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    results_df = pd.DataFrame({'Sequence_Length': lengths})
    errors_kl, errors_mmd, errors_tvd, errors_lrt = [], [], [], []

    for L in lengths:
        err_counts = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
        print(f"    Running L={L:<3}...", end="")
        for trial in range(trials):
            if (trial + 1) % batch_size == 0:
                print(f" {int(100*(trial+1)/trials)}%", end="", flush=True)

            seq1 = np.random.binomial(1, p_mu, L)
            seq2 = np.random.binomial(1, p_pi, L)
            seq3 = np.random.binomial(1, p_pi, L)
            t1, t2, t3 = np.mean(seq1), np.mean(seq2), np.mean(seq3)
            
            P1,P2,P3 = np.array([t1,1-t1]), np.array([t2,1-t2]), np.array([t3,1-t3])
            Q1,Q2,Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
            if np.argmax([kl_divergence(P1,Q1), kl_divergence(P2,Q2), kl_divergence(P3,Q3)]) != 0: err_counts['kl'] += 1
            if np.argmax([rbf_kernel(t2,t3,gamma), rbf_kernel(t1,t3,gamma), rbf_kernel(t1,t2,gamma)]) != 0: err_counts['mmd'] += 1
            if np.argmax([total_variation_distance(P1,Q1), total_variation_distance(P2,Q2), total_variation_distance(P3,Q3)]) != 0: err_counts['tvd'] += 1
            
            c1,c2,c3 = np.sum(seq1), np.sum(seq2), np.sum(seq3)
            D_lrt = np.array([c1*llr_if_one+(L-c1)*llr_if_zero, c2*llr_if_one+(L-c2)*llr_if_zero, c3*llr_if_one+(L-c3)*llr_if_zero])
            if np.argmax(D_lrt) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        errors_lrt.append(err_counts['lrt']); errors_kl.append(err_counts['kl'])
        errors_mmd.append(err_counts['mmd']); errors_tvd.append(err_counts['tvd'])
    
    # Populate DataFrame with detailed results
    for name, err_counts in [('LRT', errors_lrt), ('KL', errors_kl), ('MMD', errors_mmd), ('TVD', errors_tvd)]:
        results_df[f'Error_Count_{name}'] = err_counts
        results_df[f'Prob_Error_{name}'] = results_df[f'Error_Count_{name}'] / trials
        results_df[f'Log_Prob_Error_{name}'] = np.log(results_df[f'Prob_Error_{name}'])
        
    return results_df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- SETTINGS ---
GAMMA = 10.0
NUM_TRIALS = 10000
BATCH_SIZE = 2000
SEQUENCE_LENGTHS = np.arange(20, 501, 20)
save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

scenarios = {
    "Very Easy (0.1 vs 0.9)": (0.1, 0.9), "Easy (0.2 vs 0.8)": (0.2, 0.8),
    "Medium (0.3 vs 0.7)": (0.3, 0.7), "Hard (0.4 vs 0.6)": (0.4, 0.6),
    "Very Hard (0.45 vs 0.55)": (0.45, 0.55), "Extremely Hard (0.48 vs 0.52)": (0.48, 0.52)
}

# --- EXPERIMENT LOOP ---
summary_results = []
excel_writer = pd.ExcelWriter(os.path.join(save_folder, 'detailed_experiment_results.xlsx'), engine='openpyxl')

print("Starting comparative experiment. This will take several minutes...")
for name, (p_pi, p_mu) in scenarios.items():
    print(f"\n{'='*60}\nRunning Scenario: {name}\n{'='*60}")
    
    # Run simulation and get the detailed results DataFrame for this scenario
    scenario_df = run_simulation_for_scenario(p_pi, p_mu, GAMMA, SEQUENCE_LENGTHS, NUM_TRIALS, BATCH_SIZE)
    scenario_df.to_excel(excel_writer, sheet_name=name.split(" ")[0], index=False)
    
    # Estimate alphas from the detailed results
    alpha_kl = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_KL'].values)
    alpha_mmd = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_MMD'].values)
    alpha_tvd = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_TVD'].values)
    alpha_lrt = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_LRT'].values)
    
    pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
    b_dist = bhattacharyya_distance(pi_dist, mu_dist)
    
    summary_results.append({
        "Scenario": name, "Pi": p_pi, "Mu": p_mu, "2*DB_Bound": 2*b_dist,
        "alpha_LRT": alpha_lrt, "alpha_KL": alpha_kl, "alpha_MMD": alpha_mmd, "alpha_TVD": alpha_tvd
    })

excel_writer.close()
print("\n--- Experiment Finished. Analyzing final results... ---")

# --- FINAL ANALYSIS AND REPORTING ---
summary_df = pd.DataFrame(summary_results)

for test in ['LRT', 'KL', 'MMD', 'TVD']:
    # Clamp efficiency at 1.0 to respect the theoretical bound
    summary_df[f'Efficiency_{test}'] = np.minimum(1.0, summary_df[f'alpha_{test}'] / summary_df['2*DB_Bound'])

min_efficiencies = {
    'LRT (Optimal)': summary_df['Efficiency_LRT'].min(skipna=True), 'KL Divergence': summary_df['Efficiency_KL'].min(skipna=True),
    'MMD': summary_df['Efficiency_MMD'].min(skipna=True), 'Total Variation': summary_df['Efficiency_TVD'].min(skipna=True)
}
best_test = max(min_efficiencies, key=min_efficiencies.get)

# --- SAVE AND DISPLAY RESULTS ---
summary_df.to_excel(os.path.join(save_folder, 'summary_experiment_results.xlsx'), index=False, float_format="%.4f")

print("\n" + "="*60); print("       COMPARATIVE EXPERIMENT: FINAL SUMMARY"); print("="*60)
print("Detailed results for each scenario saved to separate sheets in: detailed_experiment_results.xlsx")
print("\n--- Summary Table ---")
print(summary_df.to_string(float_format="%.4f"))
print("\n--- Maximin Results ---")
print("Worst-Case Efficiency Ratio (min alpha / 2*DB):")
for test, value in min_efficiencies.items(): print(f"- {test:<17}: {value:.4f}")
print("-" * 60); print(f"CONCLUSION (Maximin): The most robust test is '{best_test}'"); print("="*60)

# --- CREATE AND SAVE FINAL SUMMARY PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
x_axis = summary_df['2*DB_Bound']
test_metrics = {'LRT (Optimal)': 'LRT', 'KL Divergence': 'KL', 'MMD': 'MMD', 'Total Variation': 'TVD'}
colors = {'LRT (Optimal)': 'red', 'KL Divergence': 'blue', 'MMD': 'green', 'Total Variation': 'purple'}
markers = {'LRT (Optimal)': 'o', 'KL Divergence': 's', 'MMD': 'x', 'Total Variation': 'd'}

for long_name, short_name in test_metrics.items():
    ax_summary.plot(x_axis, summary_df[f'Efficiency_{short_name}'], label=long_name, color=colors[long_name], marker=markers[long_name], linestyle='-', markersize=8)

ax_summary.invert_xaxis()
ax_summary.axhline(y=1.0, color='black', linestyle='--', label='Theoretical Maximum Efficiency')
ax_summary.set_xlabel('Problem Difficulty (2 * Bhattacharyya Distance -> smaller is harder)', fontsize=12)
ax_summary.set_ylabel('Test Efficiency Ratio (alpha / 2*DB)', fontsize=12)
ax_summary.set_title('Test Efficiency vs. Problem Difficulty', fontsize=16)
ax_summary.legend(fontsize=11); ax_summary.set_ylim(0, 1.1); plt.tight_layout()
plot_path = os.path.join(save_folder, 'efficiency_comparison_plot.png')
plt.savefig(plot_path, dpi=300); plt.show()
print(f"Summary plot saved to: {plot_path}")