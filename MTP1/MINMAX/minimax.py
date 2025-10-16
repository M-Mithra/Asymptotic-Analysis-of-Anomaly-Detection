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
    
    # If a test makes no errors, its performance is perfect. Alpha is effectively infinite.
    if not valid_indices:
        return np.inf

    # If there's only one point with errors, we can't fit a line.
    if len(valid_indices) < 2:
        return np.nan # Not a Number, indicates we couldn't estimate it.

    fit_lengths = lengths[valid_indices]
    fit_log_p = np.log(np.array(error_probs)[valid_indices])
    
    # Fit a line (degree 1 polynomial) to the log-probability plot
    slope, _ = np.polyfit(fit_lengths, fit_log_p, 1)
    
    return -slope

# ==============================================================================
# ===== CORE SIMULATION FUNCTION
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, gamma, lengths, trials, batch_size):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario.
    """
    # Pre-calculate LLR values for the optimal Likelihood Ratio Test (LRT)
    # Added epsilon for stability in case p_pi or p_mu is 0 or 1
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    # This dictionary will store the detailed results for plotting
    detailed_errors = {'kl': [], 'mmd': [], 'tvd': [], 'lrt': []}

    for L in lengths:
        err_counts = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
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
            if np.argmax([kl_divergence(P1,Q1), kl_divergence(P2,Q2), kl_divergence(P3,Q3)]) != 0: err_counts['kl'] += 1
            if np.argmax([rbf_kernel(t2,t3,gamma), rbf_kernel(t1,t3,gamma), rbf_kernel(t1,t2,gamma)]) != 0: err_counts['mmd'] += 1
            if np.argmax([total_variation_distance(P1,Q1), total_variation_distance(P2,Q2), total_variation_distance(P3,Q3)]) != 0: err_counts['tvd'] += 1
            
            c1,c2,c3 = np.sum(seq1), np.sum(seq2), np.sum(seq3)
            Z1 = c1*llr_if_one + (L-c1)*llr_if_zero
            Z2 = c2*llr_if_one + (L-c2)*llr_if_zero
            Z3 = c3*llr_if_one + (L-c3)*llr_if_zero
            if np.argmax([Z1, Z2, Z3]) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        for key in detailed_errors: detailed_errors[key].append(err_counts[key] / trials)
        
    return detailed_errors

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- SETTINGS ---
GAMMA = 10.0
NUM_TRIALS = 10000     # Increased for higher accuracy
BATCH_SIZE = 2000      # How often to print progress updates
SEQUENCE_LENGTHS = np.arange(20, 401, 20) # Wider range of lengths

# Define more scenarios to test, from easy to very hard
scenarios = {
    "Extremely Easy (0.05 vs 0.95)": (0.05, 0.95),
    "Very Easy (0.1 vs 0.9)": (0.1, 0.9),
    "Easy (0.2 vs 0.8)": (0.2, 0.8),
    "Medium (0.3 vs 0.7)": (0.3, 0.7),
    "Hard (0.4 vs 0.6)": (0.4, 0.6),
    "Very Hard (0.45 vs 0.55)": (0.45, 0.55)
}

save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- EXPERIMENT LOOP ---
results_list = []
print("Starting comparative experiment. This will take several minutes...")
for name, (p_pi, p_mu) in scenarios.items():
    print(f"\n{'='*60}\nRunning Scenario: {name}\n{'='*60}")
    
    # Run simulation to get error probabilities
    error_data = run_simulation_for_scenario(p_pi, p_mu, GAMMA, SEQUENCE_LENGTHS, NUM_TRIALS, BATCH_SIZE)
    
    # Estimate alpha for each test from the error data
    alpha_kl = estimate_alpha(SEQUENCE_LENGTHS, error_data['kl'])
    alpha_mmd = estimate_alpha(SEQUENCE_LENGTHS, error_data['mmd'])
    alpha_tvd = estimate_alpha(SEQUENCE_LENGTHS, error_data['tvd'])
    alpha_lrt = estimate_alpha(SEQUENCE_LENGTHS, error_data['lrt'])
    
    # Calculate theoretical bound
    pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
    b_dist = bhattacharyya_distance(pi_dist, mu_dist)
    
    # Print intermediate results
    print("\n  --- Results for this Scenario ---")
    print(f"  Theoretical 2*DB Bound: {2*b_dist:.4f}")
    print(f"  Est. Alpha - LRT (Optimal): {alpha_lrt:.4f}")
    print(f"  Est. Alpha - KL:            {alpha_kl:.4f}")
    print(f"  Est. Alpha - MMD:           {alpha_mmd:.4f}")
    print(f"  Est. Alpha - TVD:           {alpha_tvd:.4f}")
    
    # Store results for final analysis
    results_list.append({
        "Scenario": name, "Pi": p_pi, "Mu": p_mu, "2*DB_Bound": 2*b_dist,
        "alpha_LRT": alpha_lrt, "alpha_KL": alpha_kl, "alpha_MMD": alpha_mmd, "alpha_TVD": alpha_tvd
    })
    
    # --- Generate Log-Error Decay Plot for this scenario ---
    fig_decay, ax_decay = plt.subplots(figsize=(10, 7))
    ax_decay.plot(SEQUENCE_LENGTHS, np.log(error_data['lrt']), 'o-', color='red', label='LRT (Optimal)')
    ax_decay.plot(SEQUENCE_LENGTHS, np.log(error_data['kl']), 's-', color='blue', label='KL Divergence')
    ax_decay.plot(SEQUENCE_LENGTHS, np.log(error_data['mmd']), 'x-', color='green', label='MMD')
    ax_decay.plot(SEQUENCE_LENGTHS, np.log(error_data['tvd']), 'd-', color='purple', label='Total Variation')
    
    ax_decay.set_title(f'Log-Error Decay for Scenario: {name}', fontsize=16)
    ax_decay.set_xlabel('Sequence Length (L)', fontsize=12)
    ax_decay.set_ylabel('Log of Error Probability (ln Pe)', fontsize=12)
    ax_decay.grid(True, which='both', linestyle='--')
    ax_decay.legend()
    decay_plot_path = os.path.join(save_folder, f'decay_plot_{name.split(" ")[0]}.png')
    fig_decay.savefig(decay_plot_path, dpi=150)
    plt.close(fig_decay)

# --- FINAL ANALYSIS AND REPORTING ---
print("\n--- Experiment Finished. Analyzing final results... ---")
df = pd.DataFrame(results_list)

for test in ['LRT', 'KL', 'MMD', 'TVD']:
    df[f'Efficiency_{test}'] = df[f'alpha_{test}'] / df['2*DB_Bound']

min_efficiencies = {
    'LRT (Optimal)': df['Efficiency_LRT'].min(skipna=True), 'KL Divergence': df['Efficiency_KL'].min(skipna=True),
    'MMD': df['Efficiency_MMD'].min(skipna=True), 'Total Variation': df['Efficiency_TVD'].min(skipna=True)
}
best_test = max(min_efficiencies, key=min_efficiencies.get)

excel_path = os.path.join(save_folder, 'comparative_experiment_results.xlsx')
df.to_excel(excel_path, index=False, float_format="%.4f")

print("\n" + "="*60); print("       COMPARATIVE EXPERIMENT: FINAL SUMMARY"); print("="*60)
print("Detailed results for each scenario saved to:", excel_path)
print("\nWorst-Case Efficiency Ratio (min alpha / 2*DB):")
for test, value in min_efficiencies.items(): print(f"- {test:<17}: {value:.4f}")
print("-" * 60); print(f"CONCLUSION (Maximin): The most robust test is '{best_test}'"); print("="*60)

# --- CREATE AND SAVE FINAL SUMMARY PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
x_axis = df['2*DB_Bound']
test_metrics = {'LRT (Optimal)': 'LRT', 'KL Divergence': 'KL', 'MMD': 'MMD', 'Total Variation': 'TVD'}
colors = {'LRT (Optimal)': 'red', 'KL Divergence': 'blue', 'MMD': 'green', 'Total Variation': 'purple'}
markers = {'LRT (Optimal)': 'o', 'KL Divergence': 's', 'MMD': 'x', 'Total Variation': 'd'}

for long_name, short_name in test_metrics.items():
    ax_summary.plot(x_axis, df[f'Efficiency_{short_name}'], label=long_name, color=colors[long_name], marker=markers[long_name], linestyle='-', markersize=8)

ax_summary.invert_xaxis()
ax_summary.axhline(y=1.0, color='black', linestyle='--', label='Theoretical Maximum Efficiency')
ax_summary.set_xlabel('Problem Difficulty (2 * Bhattacharyya Distance -> smaller is harder)', fontsize=12)
ax_summary.set_ylabel('Test Efficiency Ratio (alpha / 2*DB)', fontsize=12)
ax_summary.set_title('Test Efficiency vs. Problem Difficulty', fontsize=16)
ax_summary.legend(fontsize=11); ax_summary.set_ylim(0, 1.1); plt.tight_layout()
plot_path = os.path.join(save_folder, 'efficiency_comparison_plot.png')
plt.savefig(plot_path, dpi=300); plt.show()

print(f"\nLog-error decay plots and summary plot saved to: {save_folder}")