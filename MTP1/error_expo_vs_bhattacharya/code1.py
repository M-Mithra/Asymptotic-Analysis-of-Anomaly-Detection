import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import math
import dataframe_image as dfi
from scipy.spatial.distance import pdist, squareform, cdist

# Suppress minor warnings from the simulation for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# ===== HELPER FUNCTIONS: METRICS AND ANALYSIS
# ==============================================================================

def kl_divergence(p, q):
    """Calculates KL Divergence D_KL(p || q)."""
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance."""
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def mmd_u_statistic_1d(X, Y, gamma):
    """Calculates the unbiased MMD^2 U-statistic for 1D samples."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2: return 0.0
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    # Efficiently calculate kernel matrices
    K_XX = np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))
    K_YY = np.exp(-gamma * squareform(pdist(Y, 'sqeuclidean')))
    K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    # Calculate the U-statistic
    term1 = (K_XX.sum() - n) / (n * (n - 1)) if n > 1 else 0
    term2 = (K_YY.sum() - m) / (m * (m - 1)) if m > 1 else 0
    term3 = K_XY.sum() * 2 / (n * m) if n > 0 and m > 0 else 0
    return max(0, term1 + term2 - term3)

def estimate_c(lengths, error_probs):
    """Robustly estimates the error exponent 'c' and fit parameters."""
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    # If a test makes no errors, its performance is perfect.
    if not valid_indices: return np.inf, (np.nan, np.nan)
    # If there are too few points with errors, a stable fit is not possible.
    if len(valid_indices) < 3: return np.nan, (np.nan, np.nan)
    # Use latter half of data for a more stable asymptotic fit
    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2: return np.nan, (np.nan, np.nan)
    
    fit_lengths = lengths[fit_indices]
    fit_log_p = np.log(np.array(error_probs)[fit_indices])
    try:
        slope, intercept = np.polyfit(fit_lengths, fit_log_p, 1)
        # Return the exponent 'c' (which is -slope) and the fit parameters
        return -slope, (slope, intercept)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, (np.nan, np.nan)

# ==============================================================================
# ===== CORE SIMULATION AND DECISION LOGIC
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, settings):
    """Runs a full Monte Carlo simulation for a single (pi, mu) scenario."""
    gamma, lengths, trials, n_seqs = settings['gamma'], settings['lengths'], settings['trials'], settings['num_sequences']
    
    epsilon = 1e-10
    llr_if_one = np.log((p_mu + epsilon) / (p_pi + epsilon))
    llr_if_zero = np.log((1 - p_mu + epsilon) / (1 - p_pi + epsilon))
    
    errors = {'kl': [], 'mmd': [], 'geo': [], 'lrt': []}

    for n in lengths:
        err_counts = {key: 0 for key in errors}
        print(f"    Running n={n:<4}...", end="", flush=True)
        for _ in range(trials):
            seqs = [np.random.binomial(1, p_mu if i==0 else p_pi, n) for i in range(n_seqs)]
            
            # --- KL Test ---
            emp_dists = [np.bincount(s, minlength=2)/n if n > 0 else np.array([0.5,0.5]) for s in seqs]
            kl_scores = [sum(kl_divergence(emp_dists[i],emp_dists[j]) for j in range(n_seqs) if i!=j) for i in range(n_seqs)]
            if np.argmax(kl_scores) != 0: err_counts['kl'] += 1

            # --- MMD Test ---
            mmd_scores = [sum(mmd_u_statistic_1d(seqs[i], seqs[j], gamma) for j in range(n_seqs) if i!=j) for i in range(n_seqs)]
            if np.argmax(mmd_scores) != 0: err_counts['mmd'] += 1
            
            # --- Geometric Centroid Test ---
            dispersions = []
            for i in range(n_seqs):
                temp_seqs = [s for idx, s in enumerate(seqs) if idx != i]
                center = np.mean(temp_seqs, axis=0)
                dispersion = sum(np.linalg.norm(s - center)**2 for s in temp_seqs)
                dispersions.append(dispersion)
            if np.argmin(dispersions) != 0: err_counts['geo'] += 1
            
            # --- LRT ---
            lrt_scores = [np.sum(s)*llr_if_one + (n-np.sum(s))*llr_if_zero for s in seqs]
            if np.argmax(lrt_scores) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        for key in errors: errors[key].append(err_counts[key] / trials)
        
    df = pd.DataFrame({'Sequence_Length': lengths})
    for name, key in [('LRT', 'lrt'), ('KL', 'kl'), ('MMD', 'mmd'), ('Geometric', 'geo')]:
        df[f'Prob_Error_{name}'] = errors[key]
    return df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- 1. SETTINGS ---
SETTINGS = {
    'gamma': 1.0, 'trials': 3000, 'num_sequences': 3,
    'tests': {
        'LRT (Optimal)': {'key': 'LRT', 'color': 'red'}, 'KL Divergence': {'key': 'KL', 'color': 'blue'},
        'MMD (Full Sample)': {'key': 'MMD', 'color': 'green'}, 'Geometric Centroid': {'key': 'Geometric', 'color': 'purple'}
    }
}
SCENARIOS = {
    "Easy": {'params': (0.2, 0.8), 'lengths': np.arange(10, 101, 10)},
    "Medium": {'params': (0.3, 0.7), 'lengths': np.arange(20, 201, 20)},
    "Hard": {'params': (0.4, 0.6), 'lengths': np.arange(50, 501, 50)},
    "VeryHard": {'params': (0.45, 0.55), 'lengths': np.arange(100, 1001, 100)}
}
save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- 2. CREATE PER-METRIC FOLDER STRUCTURE ---
for test_name in SETTINGS['tests']:
    os.makedirs(os.path.join(save_folder, test_name.replace(' ', '_')), exist_ok=True)

# --- 3. EXPERIMENT LOOP ---
summary_results = []
# Define the Excel file path ONCE, before the loop
excel_path = os.path.join(save_folder, 'all_results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    print("Starting comparative experiment. This will take several minutes...")
    for name, details in SCENARIOS.items():
        p_pi, p_mu = details['params']; current_lengths = details['lengths']
        print(f"\n{'='*60}\nRunning Scenario: {name} ({p_pi} vs {p_mu})\n{'='*60}")
        
        current_settings = SETTINGS.copy(); current_settings['lengths'] = current_lengths
        scenario_df = run_simulation_for_scenario(p_pi, p_mu, current_settings)
        scenario_df.to_excel(writer, sheet_name=name, index=False, float_format="%.5f")
        
        pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
        b_dist = bhattacharyya_distance(pi_dist, mu_dist)
        
        scenario_summary = {"Scenario": name, "pi": p_pi, "mu": p_mu, "2*DB_Bound": 2*b_dist}
        
        # --- Generate and Save Per-Metric, Per-Scenario Plots ---
        for test_name, test_info in SETTINGS['tests'].items():
            key = test_info['key']
            error_probs = scenario_df[f'Prob_Error_{key}'].values
            c, fit_params = estimate_c(current_lengths, error_probs)
            scenario_summary[f'alpha_{key}'] = c
            
            metric_folder = os.path.join(save_folder, test_name.replace(' ', '_'))
            
            # --- PLOT A: THE EXPONENTIAL DECAY CURVE (Linear Scale) ---
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            ax1.plot(current_lengths, error_probs, 'o', color=test_info['color'], label='Simulation Data (Pe)')
            if not np.isnan(c):
                y_fit = np.exp(fit_params[1]) * np.exp(-c * current_lengths)
                ax1.plot(current_lengths, y_fit, '--', color='black', label=f'Fitted Curve (c={c:.4f})')
            ax1.set_title(f'Exponential Decay for {test_name}\nScenario: {name}')
            ax1.set_xlabel('Sequence Length (n)'); ax1.set_ylabel('Probability of Error (Pe)')
            ax1.grid(True, which='both'); ax1.legend(); ax1.set_ylim(bottom=0)
            plt.savefig(os.path.join(metric_folder, f'{name}_DecayCurve.png'), dpi=150); plt.close(fig1)

            # --- PLOT B: THE LINEAR FIT PLOT (Log Scale) ---
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            log_error_probs = np.log(error_probs + 1e-9)
            ax2.plot(current_lengths, log_error_probs, 'o', color=test_info['color'], label='Log of Simulation Data (ln Pe)')
            if not np.isnan(c):
                slope, intercept = fit_params
                ax2.plot(current_lengths, slope * current_lengths + intercept, '--', color='black', label=f'Linear Fit (slope={slope:.4f})')
            ax2.set_title(f'Linear Fit for {test_name}\nScenario: {name}')
            ax2.set_xlabel('Sequence Length (n)'); ax2.set_ylabel('Log of Error Probability (ln Pe)')
            ax2.grid(True, which='both'); ax2.legend()
            plt.savefig(os.path.join(metric_folder, f'{name}_LinearFit.png'), dpi=150); plt.close(fig2)
            
        summary_results.append(scenario_summary)

    # --- 4. FINAL ANALYSIS AND REPORTING ---
    summary_df = pd.DataFrame(summary_results)
    for test_info in SETTINGS['tests'].values():
        key = test_info['key']
        summary_df[f'Efficiency_{key}'] = np.minimum(1.0, summary_df[f'alpha_{key}'] / summary_df['2*DB_Bound'])
    
    summary_df.to_excel(writer, sheet_name='Final_Summary', index=False, float_format="%.4f")

print("\n--- Experiment Finished. Finalizing results... ---")
print(f"\nDetailed results and summary saved to: {os.path.basename(excel_path)}")
dfi.export(summary_df.style.format(precision=4), os.path.join(save_folder, 'summary_table.png'))
print(f"Summary table image saved to: {os.path.join(save_folder, 'summary_table.png')}")

min_efficiencies = {name: summary_df[f'Efficiency_{info["key"]}'].min(skipna=True) for name, info in SETTINGS['tests'].items()}
best_test = max(min_efficiencies, key=lambda k: min_efficiencies.get(k, 0))

print("\n--- Maximin Results ---"); print("Worst-Case Efficiency Ratio (min alpha / 2*DB):")
for test, value in min_efficiencies.items(): print(f"- {test:<19}: {value:.4f}")
print("-" * 60); print(f"CONCLUSION (Maximin): The most robust test is '{best_test}'"); print("="*60)

# --- 5. CREATE AND SAVE FINAL SUMMARY PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
x_axis = summary_df['2*DB_Bound']
for long_name, info in SETTINGS['tests'].items():
    ax_summary.plot(x_axis, summary_df[f'Efficiency_{info["key"]}'], label=long_name, color=info['color'], marker='o', linestyle='-')

ax_summary.invert_xaxis(); ax_summary.axhline(y=1.0, color='k', linestyle='--', label='Theoretical Max')
ax_summary.set_xlabel('Problem Difficulty (2*DB -> smaller is harder)'); ax_summary.set_ylabel('Test Efficiency Ratio (alpha / 2*DB)')
ax_summary.set_title('Test Efficiency vs. Problem Difficulty'); ax_summary.legend(); ax_summary.set_ylim(0, 1.1)
plt.tight_layout(); plt.savefig(os.path.join(save_folder, 'final_efficiency_comparison.png'), dpi=300); plt.show()
print(f"\nProcess complete. All plots and data files have been saved to: {save_folder}")