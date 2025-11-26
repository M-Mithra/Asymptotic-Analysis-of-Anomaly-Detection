import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import math
import dataframe_image as dfi
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.interpolate import make_interp_spline
import multiprocessing
from functools import partial
import time

# Suppress minor warnings from the simulation for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
# ===== WORKER FUNCTION FOR PARALLEL PROCESSING (Corrected)
# ==============================================================================

def run_one_trial(trial_index, k_true, n, p_pi, p_mu, test_functions, settings):
    """
    Runs a single trial for a specific ground truth hypothesis k_true and length n.
    The 'trial_index' argument is present to absorb the value passed by pool.map,
    but it is not used in the calculation.
    """
    n_seqs = settings['num_sequences']
    
    # Generate data according to the current true hypothesis k_true
    # NOTE: This creates a list of arrays, which is what the non-Numba functions expect
    seqs = [np.random.binomial(1, p_mu if i==k_true else p_pi, n) for i in range(n_seqs)]
    
    trial_errors = {}
    trial_times = {} # New dictionary to store execution times

    # Run each test and check for errors against the current ground truth
    for test_key, test_info in test_functions.items():
        
        # --- ADDED TIMING ---
        start_time_test = time.time()
        decision = test_info['func'](seqs, **test_info['kwargs'])
        end_time_test = time.time()
        # --- END ADDED TIMING ---
        
        # An error occurs if the chosen index is not k_true
        trial_errors[test_key] = 1 if decision != k_true else 0
        
        trial_times[test_key] = end_time_test - start_time_test
        
    # --- MODIFIED RETURN ---
    # This ensures the function ALWAYS returns a 2-tuple
    return trial_errors, trial_times
    # --- END MODIFIED RETURN ---

# ==============================================================================
# ===== FIVE INDEPENDENT DECISION FUNCTIONS
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
        total_dist = sum(kl_divergence(emp_dists[i], emp_dists[j]) for j in range(n_seqs) if i != j)
        avg_distances[i] = total_dist / (n_seqs - 1) if n_seqs > 1 else 0
        
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
        avg_distances[i] = total_dist / (n_seqs - 1) if n_seqs > 1 else 0
        
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
        if not temp_seqs: continue
        center = np.mean(temp_seqs, axis=0)
        dispersion = sum(np.linalg.norm(s - center)**2 for s in temp_seqs)
        dispersions.append(dispersion)
        
    return np.argmin(dispersions) if dispersions else 0

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
                outlier_idx = (set(range(n_seqs)) - {i, j}).pop()
                
    return outlier_idx if outlier_idx != -1 else 0


def decision_universal_glt(seqs, **kwargs):
    """
    Decision Rule: Universal GLT from the paper (Li, Nitinawarat, Veeravalli).
    This is equivalent to finding the most probabilistically homogenous leave-one-out group.
    1. For each sequence, temporarily removes it.
    2. Estimates a "typical" distribution by averaging the remaining sequences.
    3. Measures the total KL divergence of the group from that average.
    4. The outlier is the sequence whose removal resulted in the MINIMUM total KL divergence.
    """
    n_seqs = len(seqs)
    num_outcomes = kwargs.get('num_outcomes', 2)
    # Convert all raw sequences to empirical probability distributions (histograms)
    emp_dists = [np.bincount(s, minlength=num_outcomes)/len(s) if len(s) > 0 else np.zeros(num_outcomes) for s in seqs]
    
    dispersions = []
    for i in range(n_seqs):
        # Create the leave-one-out group of distributions
        temp_dists = [d for idx, d in enumerate(emp_dists) if idx != i]
        if not temp_dists: continue
        
        # Estimate the typical distribution for this group by averaging
        pi_hat_i = np.mean(temp_dists, axis=0)
        
        # Calculate the dispersion score for this group
        # This is the sum of KL divergences from each member to the group's own average
        dispersion = sum(kl_divergence(d, pi_hat_i) for d in temp_dists)
        dispersions.append(dispersion)
        
    return np.argmin(dispersions) if dispersions else 0

# ==============================================================================
# ===== CORE SIMULATION FUNCTION (Parallelized for Pe_max with Progress Tracker)
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, settings, tests_to_run):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario,
    using multiprocessing to parallelize the trials. Calculates Pe_max,
    stores the individual P(e|k) values, AND records average computation time.
    """
    lengths, trials, n_seqs = settings['lengths'], settings['trials'], settings['num_sequences']
    num_workers = multiprocessing.cpu_count()
    print(f"    Using {num_workers} CPU cores for parallel simulation.")

    results_list_for_df = []
    total_lengths_to_run = len(lengths)

    for i, n in enumerate(lengths):
        # --- TIMER IS STARTED HERE ---
        start_time_n = time.time() 
        
        hypothesis_error_probs = {key: [] for key in tests_to_run.keys()}
        hypothesis_avg_times = {key: [] for key in tests_to_run.keys()} # To store avg time for k=0, 1, 2

        print(f"    Running n={n:<4} ({i + 1}/{total_lengths_to_run}) (Testing H1, H2, H3)...", end="", flush=True)

        for k_true in range(n_seqs):
            worker_func = partial(run_one_trial, k_true=k_true, n=n, p_pi=p_pi, p_mu=p_mu, 
                                  test_functions=tests_to_run, settings=settings)
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                # results_list is now a list of tuples: [(errors_dict, times_dict), ...]
                results_list = pool.map(worker_func, range(trials))

            total_errors_for_k = {key: 0 for key in tests_to_run.keys()}
            total_times_for_k = {key: 0.0 for key in tests_to_run.keys()} # Note: 0.0 for float
            
            # --- THIS IS THE CORRECT UNPACKING LOOP ---
            # It expects a 2-tuple, which run_one_trial now provides
            for trial_errors, trial_times in results_list:
                for test_key in tests_to_run.keys():
                    total_errors_for_k[test_key] += trial_errors[test_key]
                    total_times_for_k[test_key] += trial_times[test_key]
            # --- END OF LOOP ---

            for key in tests_to_run.keys():
                pe_k = total_errors_for_k[key] / trials
                hypothesis_error_probs[key].append(pe_k)
                
                avg_time_k = total_times_for_k[key] / trials # Avg time for this k_true
                hypothesis_avg_times[key].append(avg_time_k)

        # Build a dictionary for the current 'n' row
        row_data = {'n': n}
        for key in tests_to_run.keys():
            pe_k_list = hypothesis_error_probs[key] # This is [Pe(k=0), Pe(k=1), Pe(k=2)]
            pe_max = max(pe_k_list)
            
            # Avg time for this 'n' (by averaging the times from k=0, 1, 2)
            avg_time_n = np.mean(hypothesis_avg_times[key]) 
            row_data[f'Avg_Time_s_{key}'] = avg_time_n
            
            # Add individual P(e|k) values
            for k_idx in range(n_seqs):
                row_data[f'Pe_k{k_idx}_{key}'] = pe_k_list[k_idx]
            
            # Add Pe_max and Log_Pe_max
            row_data[f'Pe_max_{key}'] = pe_max
            row_data[f'Log_Pe_max_{key}'] = np.log(pe_max + 1e-9)
            
        results_list_for_df.append(row_data)

        # --- TIMER IS USED AND PRINTED HERE ---
        elapsed_n = time.time() - start_time_n
        el_minutes = int(elapsed_n // 60)
        el_seconds = int(elapsed_n % 60)
        print(f" Done. (Elapsed: {el_minutes}m {el_seconds}s)")
        # --- END OF ELAPSED TIME SECTION ---

    df = pd.DataFrame(results_list_for_df)
    return df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    # --- 1. SETTINGS ---
    SETTINGS = {
        'gamma': 1.0, 
        'trials': 5000, 
        'num_sequences': 3,
        'tests': {
            'Max Avg KL': {'key': 'kl', 'color': 'blue', 'marker': 'o'},
            #'Max Avg MMD': {'key': 'mmd', 'color': 'green', 'marker': 's'},
            #'Geometric Centroid': {'key': 'geo', 'color': 'purple', 'marker': '^'},
            #'Closest Pair': {'key': 'pair', 'color': 'brown', 'marker': 'D'},
            'Universal GLT': {'key': 'glt', 'color': 'cyan', 'marker': 'v'}
        }

    }

    # --- SCENARIOS ---
    # Using custom-tuned lengths for each scenario is the best methodology.
    # - Easy cases use shorter lengths because the error rate drops to zero very quickly.
    # - Hard cases use longer lengths because the error rate drops very slowly.
    # This ensures we always capture a clean decay curve to accurately estimate 'c'.
    # --- SCENARIOS ---
    # Using custom-tuned lengths for each scenario is the best methodology.
    # This expanded list provides more data points for a smoother final plot.
    # === NEW EXPANDED SCENARIOS ===
    # 12 scenarios with descriptive names (Easy/Medium/Hard) and TVD values.
    SCENARIOS = {
        "Easy_TVD_0.50":    {'params': (0.25, 0.75), 'lengths': np.arange(20, 151, 15)},
        "Easy_TVD_0.40":    {'params': (0.3, 0.7), 'lengths': np.arange(20, 201, 20)},
        "Easy_TVD_0.35":    {'params': (0.325, 0.675), 'lengths': np.arange(30, 301, 30)},
        "Medium_TVD_0.30":  {'params': (0.35, 0.65), 'lengths': np.arange(40, 401, 40)},
        "Medium_TVD_0.25":  {'params': (0.375, 0.625), 'lengths': np.arange(45, 451, 45)},
        "Medium_TVD_0.20":  {'params': (0.4, 0.6), 'lengths': np.arange(50, 501, 50)},
        "Hard_TVD_0.15":    {'params': (0.425, 0.575), 'lengths': np.arange(100, 1001, 100)},
        "Hard_TVD_0.10":    {'params': (0.45, 0.55), 'lengths': np.arange(200, 2001, 200)},
        "VHard_TVD_0.08":   {'params': (0.46, 0.54), 'lengths': np.arange(300, 3001, 300)},
        "VHard_TVD_0.06":   {'params': (0.47, 0.53), 'lengths': np.arange(400, 4001, 400)},
        "XHard_TVD_0.04":   {'params': (0.48, 0.52), 'lengths': np.arange(500, 5001, 500)},
        "XHard_TVD_0.02":   {'params': (0.49, 0.51), 'lengths': np.arange(1000, 10001, 1000)},
    }

    # Define the test functions to be run
    TEST_FUNCTIONS = {
        'kl': {'func': decision_max_avg_kl, 'kwargs': {'num_outcomes': 2}},
        'mmd': {'func': decision_max_avg_mmd, 'kwargs': {'gamma': SETTINGS['gamma']}},
        #'geo': {'func': decision_geometric_centroid, 'kwargs': {}},
        'pair': {'func': decision_closest_pair, 'kwargs': {}},
        'glt': {'func': decision_universal_glt, 'kwargs': {'num_outcomes': 2}} # ADD THIS LINE
    }

    save_folder = input("Enter folder to save experiment results: ").strip()
    os.makedirs(save_folder, exist_ok=True)

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
            
            # Calculate Total Variation Distance for this scenario
            tvd_dist = 0.5 * np.sum(np.abs(pi_dist - mu_dist)) # Direct calculation for Bernoulli
            
            scenario_summary = {"Scenario": name, "pi": p_pi, "mu": p_mu, "2*DB_Bound": 2*b_dist}
            scenario_summary['TVD_Dist'] = tvd_dist
            
            
            # --- Generate and Save Per-Metric, Per-Scenario Plots ---
            for test_name, test_info in SETTINGS['tests'].items():
                
                # --- MODIFIED LINE: Get the short key (e.g., 'kl') ---
                key_name = test_info['key'] 
                
                # --- MODIFIED LINE: Use the new column name ---
                error_probs = scenario_df[f'Pe_max_{key_name}'].values
                
                c, fit_params = estimate_c(current_lengths, error_probs) # Assumes you are still using your original estimate_c
                scenario_summary[f'c_{key_name.upper()}'] = c
                
                metric_folder = os.path.join(save_folder, test_name.replace(' ', '_'))
                
                # --- PLOT A: THE EXPONENTIAL DECAY CURVE (Linear Scale) ---
                fig1, ax1 = plt.subplots(figsize=(10, 7))
                ax1.plot(current_lengths, error_probs, 'o', color=test_info['color'], label='Simulation Data (Pe)')
                if not np.isnan(c):
                    # This logic assumes estimate_c returns c, (slope, intercept)
                    y_fit = np.exp(fit_params[1]) * np.exp(-c * current_lengths) 
                    ax1.plot(current_lengths, y_fit, '--', color='black', label=f'Fitted Curve (c={c:.4f})')
                ax1.set_title(f'Exponential Decay for {test_name}\nScenario: {name}')
                ax1.set_xlabel('Sequence Length (n)')
                ax1.set_ylabel('Probability of Error (Pe)')
                
                ax1.set_yscale('linear')
                ax1.set_ylim(bottom=0)
                ax1.grid(True, which='both')
                ax1.legend()
                
                plt.savefig(os.path.join(metric_folder, f'{name}_DecayCurve.png'), dpi=150)
                plt.close(fig1)

                # --- PLOT B: THE LINEAR FIT PLOT (Log Scale) ---
                fig2, ax2 = plt.subplots(figsize=(10, 7))
                
                # --- MODIFIED LINE: Use the new column name ---
                log_error_probs = scenario_df[f'Log_Pe_max_{key_name}'].values
                
                ax2.plot(current_lengths, log_error_probs, 'o', color=test_info['color'], label='Log of Simulation Data (ln Pe)')
                if not np.isnan(c):
                    slope, intercept = fit_params
                    ax2.plot(current_lengths, slope * current_lengths + intercept, '--', color='black', label=f'Linear Fit (slope={slope:.4f})')
                ax2.set_title(f'Linear Fit for {test_name} (to find c)\nScenario: {name}')
                ax2.set_xlabel('Sequence Length (n)')
                ax2.set_ylabel('Log of Error Probability (ln Pe)')
                ax2.grid(True, which='both')
                ax2.legend()
                
                plt.savefig(os.path.join(metric_folder, f'{name}_LinearFit.png'), dpi=150)
                plt.close(fig2)
                
                # --- PLOT C: EXPONENT CONVERGENCE PLOT (-ln(Pe)/n vs n with Curve) ---
                fig3, ax3 = plt.subplots(figsize=(10, 7))
                
                # --- MODIFIED LINE: Use the new log_error_probs variable ---
                exponent_estimate_vs_n = -log_error_probs / (current_lengths + 1e-9)

                # Filter out infinite or NaN values that might occur if Pe is 0 or very small
                valid_exponent_indices = np.isfinite(exponent_estimate_vs_n) & (error_probs > 0)
                
                x_valid = current_lengths[valid_exponent_indices]
                y_valid = exponent_estimate_vs_n[valid_exponent_indices]
                
                # Plot the raw estimate points
                ax3.plot(x_valid, y_valid, 'o', color=test_info['color'], label='Estimate (-ln(Pe)/n)')

                # --- Add smooth curve connecting the points ---
                if len(x_valid) > 3: # Need enough points for a smooth curve
                    # Sort data for spline
                    sort_idx = np.argsort(x_valid)
                    x_sorted = x_valid[sort_idx]
                    y_sorted = y_valid[sort_idx]
                    
                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
                    try:
                        spline = make_interp_spline(x_sorted, y_sorted)
                        y_smooth = spline(x_smooth)
                        y_smooth = np.maximum(y_smooth, 0) 
                        ax3.plot(x_smooth, y_smooth, linestyle='-', color=test_info['color'], alpha=0.7, label='Trend')
                    except ValueError as e:
                        print(f"    - Could not create spline for convergence plot ({test_name}, {name}): {e}")
                        ax3.plot(x_sorted, y_sorted, linestyle='--', color=test_info['color'], alpha=0.6) # Fallback line
                elif len(x_valid) > 1: # Fallback line for 2-3 points
                    sort_idx = np.argsort(x_valid)
                    ax3.plot(x_valid[sort_idx], y_valid[sort_idx], linestyle='--', color=test_info['color'], alpha=0.6)

                if not np.isnan(c) and c != np.inf:
                    ax3.axhline(y=c, color='black', linestyle='--', label=f'Final Estimated c ({c:.4f})')

                ax3.set_title(f'Exponent Estimate Convergence for {test_name}\nScenario: {name}')
                ax3.set_xlabel('Sequence Length (n)')
                ax3.set_ylabel('Exponent Estimate (-ln(Pe)/n)')
                ax3.grid(True, which='both')
                ax3.legend()
                ax3.set_ylim(bottom=0) 
                
                plt.savefig(os.path.join(metric_folder, f'{name}_ExponentConvergence.png'), dpi=150)
                plt.close(fig3)
                
            summary_results.append(scenario_summary)

        # --- 4. FINAL ANALYSIS AND REPORTING ---
        summary_df = pd.DataFrame(summary_results)
        
        # --- MODIFIED LOOP: Use the correct keys ---
        for test_name, test_info in SETTINGS['tests'].items():
            key = test_info['key']
            summary_df[f'Ratio_{key.upper()}'] = np.minimum(1.0, summary_df[f'c_{key.upper()}'] / summary_df['2*DB_Bound'])
        
        summary_df.to_excel(writer, sheet_name='Final_Summary', index=False, float_format="%.4f")

    print("\n--- Experiment Finished. Finalizing results... ---")
    print(f"\nDetailed results and summary saved to: {os.path.basename(excel_path)}")
    dfi.export(summary_df.style.format(precision=4), os.path.join(save_folder, 'summary_table.png'))
    print(f"Summary table image saved to: {os.path.join(save_folder, 'summary_table.png')}")

    # --- MODIFIED BLOCK: Use correct keys ---
    min_efficiencies = {}
    for test_name, test_info in SETTINGS['tests'].items():
        key = test_info['key']
        min_efficiencies[test_name] = summary_df[f'Ratio_{key.upper()}'].min(skipna=True)
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
        ax_summary.plot(x_axis, summary_df[f'Ratio_{key.upper()}'], label=long_name, color=info['color'], marker=info['marker'], linestyle='-')
    ax_summary.invert_xaxis(); ax_summary.axhline(y=1.0, color='k', linestyle='--', label='Theoretical Max')
    ax_summary.set_xlabel('Problem Difficulty (2*DB -> smaller is harder)'); ax_summary.set_ylabel('Test Efficiency Ratio (c / 2*DB)')
    ax_summary.set_title('Test Efficiency vs. Problem Difficulty'); ax_summary.legend(); ax_summary.set_ylim(0, 1.1)
    plt.tight_layout(); plt.savefig(os.path.join(save_folder, 'final_efficiency_comparison.png'), dpi=300); plt.show()
    print(f"\nProcess complete. All plots and data files have been saved to: {save_folder}")

    # ==============================================================================
    # ===== 6. CREATE AND SAVE ERROR EXPONENT vs TVD PLOTS
    # ==============================================================================

    print("\n--- Generating Error Exponent vs. TVD plots ---")
    tvd_plot_folder = os.path.join(save_folder, 'Exponent_vs_TVD_Plots')
    os.makedirs(tvd_plot_folder, exist_ok=True)

    # Use TVD as the x-axis (Problem Difficulty)
    x_axis_tvd = summary_df['TVD_Dist'].values

    # --- Loop to create a SEPARATE plot for each test ---
    for long_name, info in SETTINGS['tests'].items():
        key = info['key']
        color = info['color']
        
        # Get the y-axis data (Error Exponent 'c') for the current test
        y_axis_c = summary_df[f'c_{key.upper()}'].values
        
        # Create a new figure for this specific test
        fig_tvd, ax_tvd = plt.subplots(figsize=(10, 7))
        
        # Filter out NaN values which cause spline fitting to fail
        valid_indices = ~np.isnan(y_axis_c)
        x_valid = x_axis_tvd[valid_indices]
        y_valid = y_axis_c[valid_indices]

        # Sort data by x-value for spline
        sorted_indices = np.argsort(x_valid)
        x_sorted = x_valid[sorted_indices]
        y_sorted = y_valid[sorted_indices]

        # Plot original data points
        ax_tvd.plot(x_axis_tvd, y_axis_c, 'o', color=color, markersize=8, label=f'{long_name} (Data Points)')

        # Create and plot the smooth curve if enough points exist
        if len(x_sorted) > 3:
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            try:
                spline = make_interp_spline(x_sorted, y_sorted)
                y_smooth = spline(x_smooth)
                ax_tvd.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f'{long_name} Trend')
            except ValueError as e:
                print(f"  - Could not create spline for {long_name}: {e}")
                # Fallback to plotting just the line connecting points
                ax_tvd.plot(x_sorted, y_sorted, color=color, linestyle='--', label=f'{long_name} Trend (Linear)')

        # --- Finalize the plot ---
        # No axis inversion needed here, higher TVD means easier problem
        ax_tvd.set_xlabel('Problem Difficulty (Total Variation Distance -> larger is easier)', fontsize=12)
        ax_tvd.set_ylabel("Estimated Error Exponent ('c')", fontsize=12)
        ax_tvd.set_title(f"Error Exponent vs. TVD for {long_name}", fontsize=16)
        ax_tvd.legend(fontsize=11)
        ax_tvd.grid(True, which='both', linestyle='--')
        ax_tvd.set_ylim(bottom=0) # Exponent should be non-negative
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(tvd_plot_folder, f'Exponent_vs_TVD_{key}.png')
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"Saved Exponent vs TVD plot for {long_name} to: {plot_path}")

    # --- Create and Save the COMBINED Exponent vs TVD plot ---
    print("\n--- Generating combined Exponent vs. TVD plot ---")
    fig_tvd_comb, ax_tvd_comb = plt.subplots(figsize=(12, 8))

    for long_name, info in SETTINGS['tests'].items():
        key = info['key']
        color = info['color']
        marker = info['marker']
        
        y_axis_c = summary_df[f'c_{key.upper()}'].values
        
        valid_indices = ~np.isnan(y_axis_c)
        x_valid = x_axis_tvd[valid_indices]
        y_valid = y_axis_c[valid_indices]
        sorted_indices = np.argsort(x_valid)
        x_sorted = x_valid[sorted_indices]
        y_sorted = y_valid[sorted_indices]
        
        # Plot original data points
        ax_tvd_comb.plot(x_axis_tvd, y_axis_c, marker=marker, color=color, linestyle='None', markersize=8, label=f'{long_name}')
        
        # Create and plot smooth curve
        if len(x_sorted) > 3:
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            try:
                spline = make_interp_spline(x_sorted, y_sorted)
                y_smooth = spline(x_smooth)
                ax_tvd_comb.plot(x_smooth, y_smooth, color=color, linewidth=2)
            except ValueError:
                ax_tvd_comb.plot(x_sorted, y_sorted, color=color, linestyle='--') # Fallback line

    # Finalize the combined plot
    ax_tvd_comb.set_xlabel('Problem Difficulty (Total Variation Distance -> larger is easier)', fontsize=12)
    ax_tvd_comb.set_ylabel("Estimated Error Exponent ('c')", fontsize=12)
    ax_tvd_comb.set_title('Combined: Error Exponent vs. TVD for All Tests', fontsize=16)
    ax_tvd_comb.legend(fontsize=11)
    ax_tvd_comb.grid(True, which='both', linestyle='--')
    ax_tvd_comb.set_ylim(bottom=0)
    plt.tight_layout()

    plot_path_combined = os.path.join(tvd_plot_folder, 'Exponent_vs_TVD_ALL_CURVES.png')
    plt.savefig(plot_path_combined, dpi=300)
    plt.show()
    print(f"Saved combined Exponent vs TVD plot to: {plot_path_combined}")