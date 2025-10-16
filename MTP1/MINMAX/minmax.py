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
    if len(valid_indices) < 4: return np.nan, (np.nan, np.nan)
    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2: return np.nan, (np.nan, np.nan)
    fit_lengths = lengths[fit_indices]
    fit_log_p = np.log(np.array(error_probs)[fit_indices])
    try:
        slope, intercept = np.polyfit(fit_lengths, fit_log_p, 1)
        return -slope, (slope, intercept)
    except np.linalg.LinAlgError:
        return np.nan, (np.nan, np.nan)

# ==============================================================================
# ===== CORE SIMULATION FUNCTION
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, gamma, lengths, trials, batch_size):
    """Runs a full Monte Carlo simulation for a single (pi, mu) scenario."""
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    detailed_errors = {'kl': [], 'mmd': [], 'tvd': [], 'lrt': []}

    for L in lengths:
        err_counts = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
        print(f"    Running L={L:<3}...", end="")
        for trial in range(trials):
            if (trial + 1) % batch_size == 0:
                print(f" {int(100*(trial+1)/trials)}%", end="", flush=True)

            seq1 = np.random.binomial(1, p_mu, L); seq2 = np.random.binomial(1, p_pi, L); seq3 = np.random.binomial(1, p_pi, L)
            t1, t2, t3 = np.mean(seq1), np.mean(seq2), np.mean(seq3)
            
            # --- KL & TVD Test (New Pairwise Logic) ---
            P1,P2,P3 = np.array([t1,1-t1]), np.array([t2,1-t2]), np.array([t3,1-t3])
            # Score for a sequence is its avg distance to the others
            D_kl = np.array([(kl_divergence(P1,P2)+kl_divergence(P1,P3))/2, (kl_divergence(P2,P1)+kl_divergence(P2,P3))/2, (kl_divergence(P3,P1)+kl_divergence(P3,P2))/2])
            D_tvd = np.array([(total_variation_distance(P1,P2)+total_variation_distance(P1,P3))/2, (total_variation_distance(P2,P1)+total_variation_distance(P2,P3))/2, (total_variation_distance(P3,P1)+total_variation_distance(P3,P2))/2])
            if np.argmax(D_kl) != 0: err_counts['kl'] += 1
            if np.argmax(D_tvd) != 0: err_counts['tvd'] += 1

            # --- MMD Test (Original Logic) ---
            k23=rbf_kernel(t2,t3,gamma); k13=rbf_kernel(t1,t3,gamma); k12=rbf_kernel(t1,t2,gamma)
            if np.argmax([k23, k13, k12]) != 0: err_counts['mmd'] += 1
            
            # --- LRT (Optimal Test) ---
            c1,c2,c3 = np.sum(seq1), np.sum(seq2), np.sum(seq3)
            D_lrt = np.array([c1*llr_if_one+(L-c1)*llr_if_zero, c2*llr_if_one+(L-c2)*llr_if_zero, c3*llr_if_one+(L-c3)*llr_if_zero])
            if np.argmax(D_lrt) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        for key in detailed_errors: detailed_errors[key].append(err_counts[key] / trials)
        
    results_df = pd.DataFrame({'Sequence_Length': lengths})
    for name, counts in [('LRT', detailed_errors['lrt']), ('KL', detailed_errors['kl']), ('MMD', detailed_errors['mmd']), ('TVD', detailed_errors['tvd'])]:
        error_probs = np.array(counts)
        results_df[f'Prob_Error_{name}'] = error_probs
        results_df[f'Log_Prob_Error_{name}'] = np.log(error_probs + 1e-9)
        
    return results_df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- SETTINGS ---
GAMMA = 10.0
NUM_TRIALS = 5000
BATCH_SIZE = 1000
SEQUENCE_LENGTHS = np.arange(20, 401, 20)
save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

scenarios = {
    "Very Easy (0.1 vs 0.9)": (0.1, 0.9), "Easy (0.2 vs 0.8)": (0.2, 0.8),
    "Medium (0.3 vs 0.7)": (0.3, 0.7), "Hard (0.4 vs 0.6)": (0.4, 0.6),
    "Very Hard (0.45 vs 0.55)": (0.45, 0.55),
}

# --- EXPERIMENT LOOP ---
summary_results = []
excel_writer = pd.ExcelWriter(os.path.join(save_folder, 'detailed_experiment_results.xlsx'), engine='openpyxl')

print("Starting comparative experiment. This will take several minutes...")
for name, (p_pi, p_mu) in scenarios.items():
    print(f"\n{'='*60}\nRunning Scenario: {name}\n{'='*60}")
    
    scenario_df = run_simulation_for_scenario(p_pi, p_mu, GAMMA, SEQUENCE_LENGTHS, NUM_TRIALS, BATCH_SIZE)
    scenario_df.to_excel(excel_writer, sheet_name=name.split(" ")[0], index=False)
    
    # --- Analyze and Plot Results for this Scenario ---
    pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
    b_dist = bhattacharyya_distance(pi_dist, mu_dist)
    
    alphas, fits = {}, {}
    for test in ['LRT', 'KL', 'MMD', 'TVD']:
        alphas[test], fits[test] = estimate_alpha(SEQUENCE_LENGTHS, scenario_df[f'Prob_Error_{test}'].values)

    summary_results.append({
        "Scenario": name, "Pi": p_pi, "Mu": p_mu, "2*DB_Bound": 2*b_dist,
        "alpha_LRT": alphas['LRT'], "alpha_KL": alphas['KL'], "alpha_MMD": alphas['MMD'], "alpha_TVD": alphas['TVD']
    })
    
    # --- Generate Plot A: Exponential Decay (Pe vs L) ---
    fig_decay, ax_decay = plt.subplots(figsize=(10, 7))
    for test, color in [('LRT', 'red'), ('KL', 'blue'), ('MMD', 'green'), ('TVD', 'purple')]:
        ax_decay.plot(SEQUENCE_LENGTHS, scenario_df[f'Prob_Error_{test}'], 'o', color=color, label=f'{test} (Data)')
        if not np.isnan(alphas[test]):
            y_fit = np.exp(-alphas[test] * SEQUENCE_LENGTHS) * (scenario_df[f'Prob_Error_{test}'][0] / np.exp(-alphas[test]*SEQUENCE_LENGTHS[0]))
            ax_decay.plot(SEQUENCE_LENGTHS, y_fit, '--', color=color, label=f'{test} (Fit, alpha~{alphas[test]:.3f})')
    ax_decay.set_title(f'Error Probability vs. Length ({name})', fontsize=16)
    ax_decay.set_xlabel('Sequence Length (L)'); ax_decay.set_ylabel('Probability of Error (Pe)')
    ax_decay.set_yscale('log'); ax_decay.grid(True, which='both', linestyle='--'); ax_decay.legend()
    fig_decay.savefig(os.path.join(save_folder, f'decay_Pe_{name.split(" ")[0]}.png'), dpi=150); plt.close(fig_decay)

    # --- Generate Plot B: Linear Fit (ln(Pe) vs L) ---
    fig_log, ax_log = plt.subplots(figsize=(10, 7))
    for test, color in [('LRT', 'red'), ('KL', 'blue'), ('MMD', 'green'), ('TVD', 'purple')]:
        ax_log.plot(SEQUENCE_LENGTHS, scenario_df[f'Log_Prob_Error_{test}'], 'o', color=color, label=f'{test} (Data)')
        if not np.isnan(fits[test][0]):
            slope, intercept = fits[test]
            ax_log.plot(SEQUENCE_LENGTHS, slope * SEQUENCE_LENGTHS + intercept, '--', color=color, label=f'{test} (Fit, alpha={-slope:.3f})')
    ax_log.set_title(f'Log-Error vs. Length for Alpha Estimation ({name})', fontsize=16)
    ax_log.set_xlabel('Sequence Length (L)'); ax_log.set_ylabel('Log of Error Probability (ln Pe)')
    ax_log.grid(True, which='both', linestyle='--'); ax_log.legend()
    fig_log.savefig(os.path.join(save_folder, f'decay_LogPe_{name.split(" ")[0]}.png'), dpi=150); plt.close(fig_log)

# --- FINAL ANALYSIS AND REPORTING ---
summary_df = pd.DataFrame(summary_results)
for test in ['LRT', 'KL', 'MMD', 'TVD']:
    summary_df[f'Efficiency_{test}'] = np.minimum(1.0, summary_df[f'alpha_{test}'] / summary_df['2*DB_Bound'])

min_efficiencies = {
    'LRT (Optimal)': summary_df['Efficiency_LRT'].min(skipna=True), 'KL Divergence': summary_df['Efficiency_KL'].min(skipna=True),
    'MMD': summary_df['Efficiency_MMD'].min(skipna=True), 'Total Variation': summary_df['Efficiency_TVD'].min(skipna=True)
}
best_test = max(min_efficiencies, key=lambda k: min_efficiencies.get(k, 0))

summary_df.to_excel(excel_writer, sheet_name='Final_Summary', index=False)
excel_writer.close()

print("\n--- Experiment Finished. Analyzing final results... ---")
print("\n" + "="*60); print("       COMPARATIVE EXPERIMENT: FINAL SUMMARY"); print("="*60)
print(f"Detailed results saved to separate sheets in: {os.path.basename(excel_writer.path)}")
print("\n--- Summary Table ---"); print(summary_df.to_string(float_format="%.4f"))
print("\n--- Maximin Results ---"); print("Worst-Case Efficiency Ratio (min alpha / 2*DB):")
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
print(f"\nAll detailed decay plots and the summary plot have been saved to: {save_folder}")