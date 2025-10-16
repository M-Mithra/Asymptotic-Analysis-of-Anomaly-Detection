import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import math

# Suppress minor warnings from the simulation for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# ===== HELPER FUNCTIONS FOR METRICS AND ANALYSIS
# ==============================================================================

def kl_divergence(p, q):
    """Calculates KL Divergence D_KL(p || q) for discrete distributions."""
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def rbf_kernel(x, y, gamma):
    """Computes the RBF kernel between two points."""
    return np.exp(-gamma * (x - y)**2)

def total_variation_distance(p, q):
    """Calculates the Total Variation Distance between two discrete distributions."""
    return 0.5 * np.sum(np.abs(p - q))

def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance for discrete distributions."""
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def estimate_alpha(lengths, error_probs):
    """
    Robustly estimates the error exponent alpha from simulation results.
    A more stable fit is achieved by using the latter half of the data points
    where the asymptotic behavior is more pronounced.
    """
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    
    # If a test makes no errors, its performance is perfect. Alpha is effectively infinite.
    if not valid_indices:
        return np.inf

    # If there are too few points with errors, a stable fit is not possible.
    if len(valid_indices) < 4:
        return np.nan # Not a Number, indicates we couldn't reliably estimate it.

    # Use the latter half of the data points for a more stable asymptotic fit
    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2:
        return np.nan # Still need at least two points for a line

    fit_lengths = lengths[fit_indices]
    fit_log_p = np.log(np.array(error_probs)[fit_indices])
    
    try:
        # Fit a line (degree 1 polynomial) to the log-probability plot
        slope, _ = np.polyfit(fit_lengths, fit_log_p, 1)
        return -slope
    except np.linalg.LinAlgError:
        # This can happen in rare cases with ill-conditioned data
        return np.nan

# ==============================================================================
# ===== CORE SIMULATION FUNCTION
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, gamma, lengths, trials, batch_size):
    """Runs a full Monte Carlo simulation for a single (pi, mu) scenario."""
    # Pre-calculate LLR values for the optimal Likelihood Ratio Test (LRT)
    # Added epsilon for stability in case p_pi or p_mu is 0 or 1
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    # This dictionary will store the detailed error counts for each length
    error_counts = {'kl': [], 'mmd': [], 'tvd': [], 'lrt': []}

    for L in lengths:
        errs = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
        print(f"    Running L={L:<3}...", end="")
        for trial in range(trials):
            # Print progress updates in batches
            if (trial + 1) % batch_size == 0:
                print(f" {int(100*(trial+1)/trials)}%", end="", flush=True)

            # Assume H1 is true: Sequence 1 is the outlier
            seq1 = np.random.binomial(1, p_mu, L)
            seq2 = np.random.binomial(1, p_pi, L)
            seq3 = np.random.binomial(1, p_pi, L)
            
            t1, t2, t3 = np.mean(seq1), np.mean(seq2), np.mean(seq3)
            
            # --- Run all 4 tests ---
            P1,P2,P3 = np.array([t1,1-t1]), np.array([t2,1-t2]), np.array([t3,1-t3])
            Q1,Q2,Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
            if np.argmax([kl_divergence(P1,Q1), kl_divergence(P2,Q2), kl_divergence(P3,Q3)]) != 0: errs['kl'] += 1
            if np.argmax([rbf_kernel(t2,t3,gamma), rbf_kernel(t1,t3,gamma), rbf_kernel(t1,t2,gamma)]) != 0: errs['mmd'] += 1
            if np.argmax([total_variation_distance(P1,Q1), total_variation_distance(P2,Q2), total_variation_distance(P3,Q3)]) != 0: errs['tvd'] += 1
            
            c1,c2,c3 = np.sum(seq1), np.sum(seq2), np.sum(seq3)
            D_lrt = np.array([c1*llr_if_one+(L-c1)*llr_if_zero, c2*llr_if_one+(L-c2)*llr_if_zero, c3*llr_if_one+(L-c3)*llr_if_zero])
            if np.argmax(D_lrt) != 0: errs['lrt'] += 1
        
        print(" Done.")
        for key in error_counts: error_counts[key].append(errs[key])
    
    # Create a detailed DataFrame for this experiment
    results_df = pd.DataFrame({'Sequence_Length': lengths})
    for name, counts in [('LRT', error_counts['lrt']), ('KL', error_counts['kl']), ('MMD', error_counts['mmd']), ('TVD', error_counts['tvd'])]:
        results_df[f'Error_Count_{name}'] = counts
        results_df[f'Prob_Error_{name}'] = results_df[f'Error_Count_{name}'] / trials
        # Use a small floor for log to avoid -inf, which messes up plots
        results_df[f'Log_Prob_Error_{name}'] = np.log(results_df[f'Prob_Error_{name}'] + 1e-9)
        
    return results_df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- SETTINGS ---
GAMMA = 10.0
NUM_TRIALS = 5000      # A good balance of accuracy and speed
BATCH_SIZE = 1000      # How often to print progress updates
SEQUENCE_LENGTHS = np.arange(20, 401, 20) # A good range of lengths

# Define more scenarios to test, from easy to very hard
scenarios = {
    "Very Easy (0.1 vs 0.9)": (0.1, 0.9),
    "Easy (0.2 vs 0.8)": (0.2, 0.8),
    "Medium (0.3 vs 0.7)": (0.3, 0.7),
    "Hard (0.4 vs 0.6)": (0.4, 0.6),
    "Very Hard (0.45 vs 0.55)": (0.45, 0.55),
    "Extremely Hard (0.48 vs 0.52)": (0.48, 0.52)
}

save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- EXPERIMENT LOOP ---
summary_results = []
excel_writer = pd.ExcelWriter(os.path.join(save_folder, 'detailed_experiment_results.xlsx'), engine='openpyxl')

print("Starting comparative experiment. This will take several minutes...")
for name, (p_pi, p_mu) in scenarios.items():
    print(f"\n{'='*60}\nRunning Scenario: {name}\n{'='*60}")
    
    # Run simulation and get the detailed results DataFrame for this scenario
    scenario_df = run_simulation_for_scenario(p_pi, p_mu, GAMMA, SEQUENCE_LENGTHS, NUM_TRIALS, BATCH_SIZE)
    # Save the detailed trial-by-trial results to its own sheet
    scenario_df.to_excel(excel_writer, sheet_name=name.split(" ")[0], index=False)
    
    # Estimate alphas from the detailed results
    alpha_kl = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_KL'].values)
    alpha_mmd = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_MMD'].values)
    alpha_tvd = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_TVD'].values)
    alpha_lrt = estimate_alpha(SEQUENCE_LENGTHS, scenario_df['Prob_Error_LRT'].values)
    
    pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
    b_dist = bhattacharyya_distance(pi_dist, mu_dist)
    
    # Print intermediate summary for this scenario
    print("\n  --- Results for this Scenario ---")
    print(f"  Theoretical 2*DB Bound: {2*b_dist:.4f}")
    print(f"  Est. Alpha - LRT (Optimal): {alpha_lrt:.4f}")
    print(f"  Est. Alpha - KL:            {alpha_kl:.4f}")
    print(f"  Est. Alpha - MMD:           {alpha_mmd:.4f}")
    print(f"  Est. Alpha - TVD:           {alpha_tvd:.4f}")
    
    # Store results for final summary table
    summary_results.append({
        "Scenario": name, "Pi": p_pi, "Mu": p_mu, "2*DB_Bound": 2*b_dist,
        "alpha_LRT": alpha_lrt, "alpha_KL": alpha_kl, "alpha_MMD": alpha_mmd, "alpha_TVD": alpha_tvd
    })
    
    # --- Generate Log-Error Decay Plot for this scenario ---
    fig_decay, ax_decay = plt.subplots(figsize=(10, 7))
    ax_decay.plot(SEQUENCE_LENGTHS, scenario_df['Log_Prob_Error_LRT'], 'o-', color='red', label=f'LRT (alpha~{alpha_lrt:.3f})')
    ax_decay.plot(SEQUENCE_LENGTHS, scenario_df['Log_Prob_Error_KL'], 's-', color='blue', label=f'KL (alpha~{alpha_kl:.3f})')
    ax_decay.plot(SEQUENCE_LENGTHS, scenario_df['Log_Prob_Error_MMD'], 'x-', color='green', label=f'MMD (alpha~{alpha_mmd:.3f})')
    ax_decay.plot(SEQUENCE_LENGTHS, scenario_df['Log_Prob_Error_TVD'], 'd-', color='purple', label=f'TVD (alpha~{alpha_tvd:.3f})')
    
    ax_decay.set_title(f'Log-Error Decay for Scenario: {name}', fontsize=16)
    ax_decay.set_xlabel('Sequence Length (L)', fontsize=12)
    ax_decay.set_ylabel('Log of Error Probability (ln Pe)', fontsize=12)
    ax_decay.grid(True, which='both', linestyle='--')
    ax_decay.legend()
    decay_plot_path = os.path.join(save_folder, f'decay_plot_{name.split(" ")[0]}.png')
    fig_decay.savefig(decay_plot_path, dpi=150)
    plt.close(fig_decay)

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

# Write summary to its own sheet in the Excel file
summary_df.to_excel(excel_writer, sheet_name='Final_Summary', index=False)
excel_writer.close()

print("\n--- Experiment Finished. Analyzing final results... ---")
print("\n" + "="*60); print("       COMPARATIVE EXPERIMENT: FINAL SUMMARY"); print("="*60)
print("Detailed trial-by-trial results saved to separate sheets in:\ndetailed_experiment_results.xlsx")
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
print(f"\nLog-error decay plots and summary plot saved to: {save_folder}")