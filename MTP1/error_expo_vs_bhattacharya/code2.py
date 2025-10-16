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

# ==============================================================================
# ===== HELPER FUNCTIONS: METRICS AND ANALYSIS
# ==============================================================================

def kl_divergence(p, q):
    """Calculates KL Divergence D_KL(p || q) for discrete probability distributions."""
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance for discrete probability distributions."""
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def mmd_u_statistic_1d(X, Y, gamma):
    """Calculates the unbiased MMD^2 U-statistic for 1D sample arrays."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2: return 0.0
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    K_XX = np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))
    K_YY = np.exp(-gamma * squareform(pdist(Y, 'sqeuclidean')))
    K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    term1 = (K_XX.sum() - n) / (n * (n - 1)) if n > 1 else 0
    term2 = (K_YY.sum() - m) / (m * (m - 1)) if m > 1 else 0
    term3 = K_XY.sum() * 2 / (n * m) if n > 0 and m > 0 else 0
    return max(0, term1 + term2 - term3)

def estimate_c(lengths, error_probs):
    """
    Robustly estimates the error exponent 'c' (alpha) and fit parameters from simulation results.
    - INPUTS:
        - lengths: Array of sequence lengths (n) tested.
        - error_probs: Array of corresponding probability of error (Pe) values.
    - OUTPUTS:
        - c: The estimated error exponent.
        - fit_params: The (slope, intercept) of the linear fit on the log-plot.
    """
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    if not valid_indices: return np.inf, (np.nan, np.nan)
    if len(valid_indices) < 3: return np.nan, (np.nan, np.nan)
    fit_indices = valid_indices[len(valid_indices)//2:]
    if len(fit_indices) < 2: return np.nan, (np.nan, np.nan)
    
    fit_lengths = lengths[fit_indices]
    fit_log_p = np.log(np.array(error_probs)[fit_indices])
    try:
        slope, intercept = np.polyfit(fit_lengths, fit_log_p, 1)
        return -slope, (slope, intercept)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, (np.nan, np.nan)

# ==============================================================================
# ===== FOUR INDEPENDENT DECISION FUNCTIONS
# ==============================================================================

def decision_max_avg_kl(seqs, **kwargs):
    """
    Decision Rule: Max Average KL Divergence.
    1. Models each sequence as an empirical probability distribution (histogram).
    2. For each sequence, calculates its average KL divergence to all other sequences.
    3. The outlier is the sequence with the MAXIMUM average divergence.
    """
    n_seqs = len(seqs)
    num_outcomes = kwargs.get('num_outcomes', 2)
    emp_dists = [np.bincount(s, minlength=num_outcomes)/len(s) if len(s) > 0 else np.zeros(num_outcomes) for s in seqs]
    
    avg_distances = np.zeros(n_seqs)
    for i in range(n_seqs):
        # Sum the distance from sequence 'i' to all other sequences 'j'
        total_dist = sum(kl_divergence(emp_dists[i], emp_dists[j]) for j in range(n_seqs) if i != j)
        avg_distances[i] = total_dist / (n_seqs - 1)
        
    return np.argmax(avg_distances)

def decision_max_avg_mmd(seqs, **kwargs):
    """
    Decision Rule: Max Average MMD.
    1. Considers each sequence as a set of raw samples.
    2. For each sequence, calculates its average MMD (full sample) to all other sequences.
    3. The outlier is the sequence with the MAXIMUM average MMD.
    """
    n_seqs = len(seqs)
    gamma = kwargs.get('gamma', 1.0)
    
    avg_distances = np.zeros(n_seqs)
    for i in range(n_seqs):
        total_dist = sum(mmd_u_statistic_1d(seqs[i], seqs[j], gamma) for j in range(n_seqs) if i != j)
        avg_distances[i] = total_dist / (n_seqs - 1)
        
    return np.argmax(avg_distances)

def decision_geometric_centroid(seqs, **kwargs):
    """
    Decision Rule: Geometric Centroid (your method).
    1. For each sequence, temporarily removes it from the set.
    2. Calculates the geometric center (mean) of the remaining sequences.
    3. Measures the total squared distance (dispersion) of the group around that center.
    4. The outlier is the sequence whose removal resulted in the MINIMUM dispersion.
    """
    n_seqs = len(seqs)
    dispersions = []
    for i in range(n_seqs):
        temp_seqs = [s for idx, s in enumerate(seqs) if idx != i]
        center = np.mean(temp_seqs, axis=0)
        dispersion = sum(np.linalg.norm(s - center)**2 for s in temp_seqs)
        dispersions.append(dispersion)
        
    return np.argmin(dispersions)

def decision_closest_pair(seqs, **kwargs):
    """
    Decision Rule: Closest Pair (your "Distance Test").
    1. Summarizes each sequence by its mean value.
    2. Calculates the Euclidean distance between each pair of means (d1=dist(t2,t3), etc.).
    3. The outlier is the sequence LEFT OUT of the pair with the MINIMUM distance.
    """
    n_seqs = len(seqs)
    means = [np.mean(s) for s in seqs]
    
    min_dist = np.inf
    outlier_idx = -1
    
    # Iterate through all unique pairs
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            dist = np.abs(means[i] - means[j])
            if dist < min_dist:
                min_dist = dist
                # The outlier is the index not in the current pair {i, j}
                all_indices = set(range(n_seqs))
                outlier_idx = (all_indices - {i, j}).pop()
                
    return outlier_idx

# ==============================================================================
# ===== CORE SIMULATION FUNCTION
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, settings, tests_to_run):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario,
    executing a provided dictionary of test functions.
    """
    lengths, trials, n_seqs = settings['lengths'], settings['trials'], settings['num_sequences']
    
    errors = {key: [] for key in tests_to_run.keys()}

    for n in lengths:
        err_counts = {key: 0 for key in errors}
        print(f"    Running n={n:<4}...", end="", flush=True)
        for _ in range(trials):
            # The "ground truth" is that the first sequence (index 0) is the outlier
            seqs = [np.random.binomial(1, p_mu if i==0 else p_pi, n) for i in range(n_seqs)]
            
            # Run each test and check for errors
            for test_key, test_info in tests_to_run.items():
                decision = test_info['func'](seqs, **test_info['kwargs'])
                if decision != 0: # An error occurs if the chosen index is not 0
                    err_counts[test_key] += 1
        
        print(" Done.")
        # For each test, append the calculated Probability of Error for this length 'n'
        for key in errors:
            errors[key].append(err_counts[key] / trials)
        
    # Create the final results dataframe for this scenario
    df = pd.DataFrame({'Sequence_Length': lengths})
    for test_key, test_name in zip(tests_to_run.keys(), settings['tests'].keys()):
        df[f'Prob_Error_{test_name}'] = errors[test_key]
        df[f'Log_Prob_Error_{test_name}'] = np.log(np.array(errors[test_key]) + 1e-9)
    return df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- 1. SETTINGS ---
SETTINGS = {
    'gamma': 1.0, 'trials': 3000, 'num_sequences': 3,
    'tests': {
        'Max Avg KL': {'color': 'blue', 'marker': 's'},
        'Max Avg MMD': {'color': 'green', 'marker': 'x'},
        'Geometric Centroid': {'color': 'purple', 'marker': '^'},
        'Closest Pair': {'color': 'brown', 'marker': 'd'}
    }
}
SCENARIOS = {
    "Easy": {'params': (0.2, 0.8), 'lengths': np.arange(10, 101, 10)},
    "Medium": {'params': (0.3, 0.7), 'lengths': np.arange(20, 201, 20)},
    "Hard": {'params': (0.4, 0.6), 'lengths': np.arange(50, 501, 50)},
    "VeryHard": {'params': (0.45, 0.55), 'lengths': np.arange(200, 2001, 200)}
}

save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)

# Define the test functions to be run
TEST_FUNCTIONS = {
    'kl': {'func': decision_max_avg_kl, 'kwargs': {'num_outcomes': 2}},
    'mmd': {'func': decision_max_avg_mmd, 'kwargs': {'gamma': SETTINGS['gamma']}},
    'geo': {'func': decision_geometric_centroid, 'kwargs': {}},
    'pair': {'func': decision_closest_pair, 'kwargs': {}}
}

# --- 2. CREATE PER-METRIC FOLDER STRUCTURE ---
for test_name in SETTINGS['tests']:
    os.makedirs(os.path.join(save_folder, test_name.replace(' ', '_')), exist_ok=True)

# --- 3. EXPERIMENT LOOP ---
summary_results = []
excel_path = os.path.join(save_folder, 'all_results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    print("Starting comparative experiment. This will take a long time...")
    for name, details in SCENARIOS.items():
        p_pi, p_mu = details['params']; current_lengths = details['lengths']
        print(f"\n{'='*60}\nRunning Scenario: {name} ({p_pi} vs {p_mu})\n{'='*60}")
        
        current_settings = SETTINGS.copy(); current_settings['lengths'] = current_lengths
        scenario_df = run_simulation_for_scenario(p_pi, p_mu, current_settings, TEST_FUNCTIONS)
        scenario_df.to_excel(writer, sheet_name=name, index=False, float_format="%.5f")
        
        pi_dist = np.array([p_pi, 1-p_pi]); mu_dist = np.array([p_mu, 1-p_mu])
        b_dist = bhattacharyya_distance(pi_dist, mu_dist)
        
        scenario_summary = {"Scenario": name, "pi": p_pi, "mu": p_mu, "2*DB_Bound": 2*b_dist}
        
        # --- Generate and Save Per-Metric, Per-Scenario Plots ---
        for test_name, test_info in SETTINGS['tests'].items():
            key_name = list(TEST_FUNCTIONS.keys())[list(SETTINGS['tests'].keys()).index(test_name)]
            error_probs = scenario_df[f'Prob_Error_{test_name}'].values
            c, fit_params = estimate_c(current_lengths, error_probs)
            scenario_summary[f'c_{key_name.upper()}'] = c
            
            # PLOT A: Exponential Decay Curve (Linear Scale)
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            ax1.plot(current_lengths, error_probs, 'o', color=test_info['color'], label='Simulation Data (Pe)')
            if not np.isnan(c):
                y_fit = np.exp(fit_params[1]) * np.exp(-c * current_lengths)
                ax1.plot(current_lengths, y_fit, '--', color='black', label=f'Fitted Curve (c={c:.4f})')
            ax1.set_title(f'Exponential Decay for {test_name}\nScenario: {name}')
            ax1.set_xlabel('Sequence Length (n)'); ax1.set_ylabel('Probability of Error (Pe)')
            ax1.grid(True, which='both'); ax1.legend(); ax1.set_ylim(bottom=0)
            metric_folder = os.path.join(save_folder, test_name.replace(' ', '_'))
            plt.savefig(os.path.join(metric_folder, f'{name}_DecayCurve.png'), dpi=150); plt.close(fig1)
            
        summary_results.append(scenario_summary)

    # --- 4. FINAL ANALYSIS AND REPORTING ---
    summary_df = pd.DataFrame(summary_results)
    for key in TEST_FUNCTIONS.keys():
        summary_df[f'Ratio_{key.upper()}'] = np.minimum(1.0, summary_df[f'c_{key.upper()}'] / summary_df['2*DB_Bound'])
    
    summary_df.to_excel(writer, sheet_name='Final_Summary', index=False, float_format="%.4f")

print("\n--- Experiment Finished. Finalizing results... ---")
print(f"\nDetailed results and summary saved to: {os.path.basename(excel_path)}")
dfi.export(summary_df.style.format(precision=4), os.path.join(save_folder, 'summary_table.png'))
print(f"Summary table image saved to: {os.path.join(save_folder, 'summary_table.png')}")

min_efficiencies = {name: summary_df[f'Ratio_{key.upper()}'].min(skipna=True) for name, key in zip(SETTINGS['tests'].keys(), TEST_FUNCTIONS.keys())}
best_test = max(min_efficiencies, key=lambda k: min_efficiencies.get(k, 0))

print("\n--- Maximin Results ---"); print("Worst-Case Efficiency Ratio (min c / 2*DB):")
for test, value in min_efficiencies.items(): print(f"- {test:<19}: {value:.4f}")
print("-" * 60); print(f"CONCLUSION (Maximin): The most robust test is '{best_test}'"); print("="*60)

# --- 5. CREATE AND SAVE FINAL SUMMARY PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
x_axis = summary_df['2*DB_Bound']
for long_name, info in SETTINGS['tests'].items():
    key = list(TEST_FUNCTIONS.keys())[list(SETTINGS['tests'].keys()).index(long_name)]
    ax_summary.plot(x_axis, summary_df[f'Ratio_{key.upper()}'], label=long_name, color=info['color'], marker=info['marker'])
ax_summary.invert_xaxis(); ax_summary.axhline(y=1.0, color='k', linestyle='--', label='Theoretical Max')
ax_summary.set_xlabel('Problem Difficulty (2*DB -> smaller is harder)'); ax_summary.set_ylabel('Test Efficiency Ratio (c / 2*DB)')
ax_summary.set_title('Test Efficiency vs. Problem Difficulty'); ax_summary.legend(); ax_summary.set_ylim(0, 1.1)
plt.tight_layout(); plt.savefig(os.path.join(save_folder, 'final_efficiency_comparison.png'), dpi=300); plt.show()
print(f"\nProcess complete. All plots and data files have been saved to: {save_folder}")