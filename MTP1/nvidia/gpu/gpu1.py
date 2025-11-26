import numpy as np
import cupy as cp  # <-- ADDED: This is the GPU-accelerated version of NumPy
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import math
import dataframe_image as dfi
import cupyx.scipy.spatial.distance # <-- ADD THIS IMPORT

# --- MODIFIED IMPORTS ---
# We no longer need the CPU-based distance functions or multiprocessing
# from scipy.spatial.distance import pdist, squareform, cdist # <-- REMOVED
from scipy.interpolate import make_interp_spline # <-- KEPT (for plotting)
from scipy.optimize import curve_fit # <-- ADDED (for final CPU-based curve fitting)
# import multiprocessing # <-- REMOVED
# from functools import partial # <-- REMOVED
import time

# Suppress minor warnings from the simulation for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# ===== HELPER FUNCTIONS: METRICS AND ANALYSIS
# ==============================================================================

@cp.fuse()  # <-- ADDED: CuPy decorator to fuse element-wise ops into one kernel
def kl_divergence_gpu(p, q):
    """Calculates KL Divergence D_KL(p || q) for discrete probability distributions."""
    epsilon = 1e-10; p_safe = p + epsilon; q_safe = q + epsilon
    # --- MODIFIED: Use CuPy 'cp' instead of NumPy 'np' ---
    return cp.sum(p_safe * cp.log(p_safe / q_safe))

def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance for discrete probability distributions."""
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

# (Keep @cp.fuse() removed)
def mmd_u_statistic_1d_gpu(X, Y, gamma):
    """
    Calculates the unbiased MMD^2 U-statistic for 1D sample arrays *on the GPU*.
    Assumes X and Y are 1D CuPy arrays.
    """
    n = X.shape[0]
    m = Y.shape[0]

    if n < 2 or m < 2: return 0.0

    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)

    # --- CORRECTED CALLS using cupyx.scipy.spatial.distance ---
    K_XX = cp.exp(-gamma * cupyx.scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))
    K_YY = cp.exp(-gamma * cupyx.scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
    K_XY = cp.exp(-gamma * cupyx.scipy.spatial.distance.cdist(X, Y, 'sqeuclidean'))
    # --- END CORRECTION ---

    term1 = (K_XX.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    term2 = (K_YY.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
    term3 = K_XY.sum() * 2 / (n * m) if n > 0 and m > 0 else 0.0

    return cp.maximum(0.0, term1 + term2 - term3)

#mmd rff
def mmd_rff_1d_gpu(X, Y, gamma, D):
    """
    Calculates the MMD^2 using Random Fourier Features (RFF) on the GPU.
    This is an O(n*D) approximation of the O(n^2) MMD.
    
    - X, Y: 1D CuPy arrays (the sequences)
    - gamma: The RBF kernel width
    - D: The number of random features (e.g., 200)
    """
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0 or D == 0:
        return 0.0

    # 1. Generate Random Features (omega and b)
    # The random frequencies (omega) are sampled from a Gaussian
    # with std. dev = sqrt(2 * gamma)
    omega_std = cp.sqrt(2.0 * gamma)
    omega = cp.random.normal(loc=0.0, scale=omega_std, size=(D,), dtype=cp.float64)
    # The random phases (b) are sampled from Uniform(0, 2*pi)
    b = cp.random.uniform(0.0, 2.0 * cp.pi, size=(D,), dtype=cp.float64)

    # 2. Project X and Y into the D-dimensional random feature space
    # We use cp.outer(X, omega) to create an (n, D) matrix
    # This is the O(n*D) operation
    Z_X = cp.sqrt(2.0 / D) * cp.cos(cp.outer(X, omega) + b)
    Z_Y = cp.sqrt(2.0 / D) * cp.cos(cp.outer(Y, omega) + b)

    # 3. Compute the MMD as the squared distance between the means
    mu_X = cp.mean(Z_X, axis=0)
    mu_Y = cp.mean(Z_Y, axis=0)
    
    diff = mu_X - mu_Y
    mmd_sq_rff = cp.dot(diff, diff) # ||mu_X - mu_Y||^2
    
    return mmd_sq_rff

# ==============================================================================
# ===== HELPER FUNCTIONS: ANALYSIS (CPU)
# ==============================================================================
#
# NOTE: This function is INTENTIONALLY left on the CPU.
# It is an analysis function, not a simulation bottleneck.
# It runs on CPU data (pandas/numpy arrays) after the simulation is finished.
# Moving this to the GPU would make the program SLOWER due to data transfer.
#
# ==============================================================================

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
# ===== THREE INDEPENDENT DECISION FUNCTIONS
# ==============================================================================

# --- REMOVED @cp.fuse() ---
def decision_max_avg_kl_gpu(seqs_gpu, num_outcomes):
    """
    Decision Rule: Max Average KL Divergence. (GPU-Accelerated)
    - Assumes 'seqs_gpu' is a 2D CuPy array (e.g., shape 3 x n)
    """
    n_seqs = seqs_gpu.shape[0] # <-- This line caused the error inside fuse
    n_len = seqs_gpu.shape[1] 

    emp_dists_gpu = cp.zeros((n_seqs, num_outcomes), dtype=cp.float64)
    
    if n_len > 0: 
        for i in range(n_seqs):
            row_counts = cp.bincount(seqs_gpu[i].astype(cp.int32), minlength=num_outcomes)
            emp_dists_gpu[i] = row_counts / n_len
    
    avg_distances_gpu = cp.zeros(n_seqs, dtype=cp.float64)
    for i in range(n_seqs):
        total_dist = 0.0 
        for j in range(n_seqs):
            if i != j:
                total_dist += kl_divergence_gpu(emp_dists_gpu[i], emp_dists_gpu[j])
                
        if n_seqs > 1:
            avg_distances_gpu[i] = total_dist / (n_seqs - 1)

    return cp.argmax(avg_distances_gpu)

# --- REMOVED @cp.fuse() ---
def decision_max_avg_mmd_gpu(seqs_gpu, gamma):
    """
    Decision Rule: Max Average MMD. (GPU-Accelerated)
    - Assumes 'seqs_gpu' is a LIST of 1D CuPy arrays.
    """
    n_seqs = len(seqs_gpu)

    avg_distances_gpu = cp.zeros(n_seqs, dtype=cp.float64)

    for i in range(n_seqs):
        total_dist = 0.0
        for j in range(n_seqs):
            if i != j:
                total_dist += mmd_u_statistic_1d_gpu(seqs_gpu[i], seqs_gpu[j], gamma)

        if n_seqs > 1:
            avg_distances_gpu[i] = total_dist / (n_seqs - 1)

    return cp.argmax(avg_distances_gpu)

# --- REMOVED @cp.fuse() ---
def decision_universal_glt_gpu(seqs_gpu, num_outcomes):
    """
    Decision Rule: Universal GLT. (GPU-Accelerated)
    - Assumes 'seqs_gpu' is a 2D CuPy array (e.g., shape 3 x n)
    """
    n_seqs = seqs_gpu.shape[0] # <-- This line would cause an error inside fuse
    n_len = seqs_gpu.shape[1]

    emp_dists_gpu = cp.zeros((n_seqs, num_outcomes), dtype=cp.float64)
    if n_len > 0:
        for i in range(n_seqs):
            row_counts = cp.bincount(seqs_gpu[i].astype(cp.int32), minlength=num_outcomes)
            emp_dists_gpu[i] = row_counts / n_len
            
    dists_sum_gpu = cp.sum(emp_dists_gpu, axis=0)
    dispersions_gpu = cp.zeros(n_seqs, dtype=cp.float64)
    
    loo_count = n_seqs - 1
    if loo_count == 0:
        return 0 

    for i in range(n_seqs):
        pi_hat_i = (dists_sum_gpu - emp_dists_gpu[i]) / loo_count

        dispersion = 0.0
        for j in range(n_seqs):
            if i == j: 
                continue
            dispersion += kl_divergence_gpu(emp_dists_gpu[j], pi_hat_i)
        
        dispersions_gpu[i] = dispersion

    return cp.argmin(dispersions_gpu)

# mmd rff decision
def decision_max_avg_mmd_rff_gpu(seqs_gpu, gamma, D):
    """
    Decision Rule: Max Average MMD (RFF-Approximated). (GPU-Accelerated)
    - Assumes 'seqs_gpu' is a LIST of 1D CuPy arrays.
    - D: Number of random features
    """
    n_seqs = len(seqs_gpu)
    
    avg_distances_gpu = cp.zeros(n_seqs, dtype=cp.float64)

    for i in range(n_seqs):
        total_dist = 0.0 
        for j in range(n_seqs):
            if i != j:
                # Call the new RFF MMD function
                total_dist += mmd_rff_1d_gpu(seqs_gpu[i], seqs_gpu[j], gamma, D)
        
        if n_seqs > 1:
            avg_distances_gpu[i] = total_dist / (n_seqs - 1)

    return cp.argmax(avg_distances_gpu)

# ==============================================================================
# ===== CORE SIMULATION FUNCTION (GPU-Accelerated)
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, settings, tests_to_run):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario.
    This version is *NOT* parallelized on the CPU.
    It runs in a *single process* and accelerates all math on the GPU.
    """
    lengths, trials, n_seqs = settings['lengths'], settings['trials'], settings['num_sequences']
    
    # --- REMOVED multiprocessing.cpu_count() ---
    print(f"    Using 1 CPU Process --> Accelerating with GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

    results_list_for_df = []
    total_lengths_to_run = len(lengths)

    # Get fixed settings once
    gamma = settings.get('gamma', 1.0)
    D = settings.get('D_features', 200) # <-- ADD THIS
    num_outcomes = 2 # Hard-coding from your previous settings

    for i, n in enumerate(lengths):
        start_time_n = time.time() 
        
        hypothesis_error_probs = {key: [] for key in tests_to_run.keys()}
        hypothesis_avg_times = {key: [] for key in tests_to_run.keys()} 

        print(f"    Running n={n:<4} ({i + 1}/{total_lengths_to_run}) (Testing H1, H2, H3)...", end="", flush=True)

        for k_true in range(n_seqs):
            
            # --- THIS BLOCK REPLACES THE ENTIRE MULTIPROCESSING POOL ---
            
            total_errors_for_k = {key: 0 for key in tests_to_run.keys()}
            total_times_for_k = {key: 0.0 for key in tests_to_run.keys()}

            # --- NEW: Loop over trials in the main process ---
            for _ in range(trials):
                
                # 1. Generate data on CPU (fast, no need for GPU)
                seqs_cpu = [np.random.binomial(1, p_mu if j==k_true else p_pi, n) for j in range(n_seqs)]
                
                # 2. Create GPU-native data structures
                #    - A list of 1D arrays (for MMD)
                #    - A single 2D array (for all other tests)
                seqs_gpu_list = [cp.asarray(s) for s in seqs_cpu]
                seqs_gpu_2d = cp.array(seqs_cpu, dtype=cp.int32) # Use 2D array for vectorized ops

                # 3. Run each test on the GPU
                for test_key, test_info in tests_to_run.items():
                    start_time_test = time.time()
                    
                    # Call the correct _gpu function with the correct data type
                    if test_key == 'kl':
                        decision_gpu = decision_max_avg_kl_gpu(seqs_gpu_2d, num_outcomes)
                    elif test_key == 'mmd':
                        decision_gpu = decision_max_avg_mmd_gpu(seqs_gpu_list, gamma)
                    elif test_key == 'mmd_rff':
                        decision_gpu = decision_max_avg_mmd_rff_gpu(seqs_gpu_list, gamma, D)
                    elif test_key == 'glt':
                        decision_gpu = decision_universal_glt_gpu(seqs_gpu_2d, num_outcomes)
                    else:
                        decision_gpu = 0 # Default for unknown test
                    
                    end_time_test = time.time()
                    
                    # 4. Get result back from GPU to CPU
                    # .item() or int() is the fastest way to move a single number
                    # (decision_closest_pair_gpu already returns a CPU int)
                    if test_key != 'pair':
                         decision_cpu = int(decision_gpu.item())
                    else:
                         decision_cpu = decision_gpu

                    # 5. Record errors and times (all on CPU)
                    if decision_cpu != k_true:
                        total_errors_for_k[test_key] += 1
                    
                    total_times_for_k[test_key] += (end_time_test - start_time_test)
            
            # --- END OF REPLACEMENT BLOCK ---

            # (This logic below remains the same, it just gets data from
            #  the new loop above instead of the multiprocessing pool)
            for key in tests_to_run.keys():
                pe_k = total_errors_for_k[key] / trials
                hypothesis_error_probs[key].append(pe_k)
                
                avg_time_k = total_times_for_k[key] / trials 
                hypothesis_avg_times[key].append(avg_time_k)

        # Build a dictionary for the current 'n' row
        row_data = {'n': n}
        for key in tests_to_run.keys():
            pe_k_list = hypothesis_error_probs[key] 
            pe_max = max(pe_k_list)
            
            avg_time_n = np.mean(hypothesis_avg_times[key]) 
            row_data[f'Avg_Time_s_{key}'] = avg_time_n
            
            for k_idx in range(n_seqs):
                row_data[f'Pe_k{k_idx}_{key}'] = pe_k_list[k_idx]
            
            row_data[f'Pe_max_{key}'] = pe_max
            row_data[f'Log_Pe_max_{key}'] = np.log(pe_max + 1e-9)
            
        results_list_for_df.append(row_data)

        elapsed_n = time.time() - start_time_n
        el_minutes = int(elapsed_n // 60)
        el_seconds = int(elapsed_n % 60)
        print(f" Done. (Elapsed: {el_minutes}m {el_seconds}s)")

    # The dataframe creation is the same, but now it uses CPU-based numpy
    df = pd.DataFrame(results_list_for_df)
    
    # --- ADDED: Clear GPU memory pool ---
    # After a long scenario, free all unused memory on the GPU
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    # --- END ADDED ---

    return df

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    # --- 1. SETTINGS ---
    SETTINGS = {
        'gamma': 1.0,
        'D_features': 200,
        'trials': 10000,
        'num_sequences': 3,
        'tests': {
            'Max Avg KL': {'key': 'kl', 'color': 'blue', 'marker': 'o'},
            'Max Avg MMD': {'key': 'mmd', 'color': 'green', 'marker': 's'},
            #'Max Avg MMD (RFF)': {'key': 'mmd_rff', 'color': 'red', 'marker': 'x'},
            'Universal GLT': {'key': 'glt', 'color': 'cyan', 'marker': 'v'}
        }

    }

    # ==============================================================================
    # --- NEW EXPANDED SCENARIOS (31 CASES FOR HIGH-RESOLUTION PLOT)
    # ==============================================================================
    #
    # This list is expanded to ~30 cases to provide a much smoother final
    # plot and to investigate the "dip" in the curve between TVD 0.20 and 0.25.
    #
    # Logic: Harder cases (smaller TVD) require much longer sequence lengths (n)
    # and larger steps to capture the slow exponential decay.
    #
    # ==============================================================================

    SCENARIOS = {
        # --- "Easy" Region (TVD 0.30 - 0.50) ---
        "Easy_TVD_0.50":    {'params': (0.25, 0.75), 'lengths': np.arange(20, 201, 20)},
        "Easy_TVD_0.45":    {'params': (0.275, 0.725), 'lengths': np.arange(20, 201, 20)},
        "Easy_TVD_0.40":    {'params': (0.3, 0.7), 'lengths': np.arange(20, 201, 20)},
        "Easy_TVD_0.35":    {'params': (0.325, 0.675), 'lengths': np.arange(30, 301, 30)},
        "Easy_TVD_0.32":    {'params': (0.34, 0.66), 'lengths': np.arange(35, 351, 35)},
        "Medium_TVD_0.30":  {'params': (0.35, 0.65), 'lengths': np.arange(40, 401, 40)},
        
        # --- "Dip" Region (TVD 0.20 - 0.29) ---
        "Medium_TVD_0.29":  {'params': (0.355, 0.645), 'lengths': np.arange(41, 411, 41)},
        "Medium_TVD_0.28":  {'params': (0.36, 0.64), 'lengths': np.arange(42, 421, 42)},
        "Medium_TVD_0.27":  {'params': (0.365, 0.635), 'lengths': np.arange(43, 431, 43)},
        "Medium_TVD_0.26":  {'params': (0.37, 0.63), 'lengths': np.arange(44, 441, 44)},
        "Medium_TVD_0.25":  {'params': (0.375, 0.625), 'lengths': np.arange(45, 451, 45)},
        "Medium_TVD_0.24":  {'params': (0.38, 0.62), 'lengths': np.arange(46, 461, 46)},
        "Medium_TVD_0.23":  {'params': (0.385, 0.615), 'lengths': np.arange(47, 471, 47)},
        "Medium_TVD_0.22":  {'params': (0.39, 0.61), 'lengths': np.arange(48, 481, 48)},
        "Medium_TVD_0.21":  {'params': (0.395, 0.605), 'lengths': np.arange(49, 491, 49)},
        "Medium_TVD_0.20":  {'params': (0.4, 0.6), 'lengths': np.arange(50, 501, 50)},

        # --- "Hard" Region (TVD 0.10 - 0.19) ---
        "Medium_TVD_0.19":  {'params': (0.405, 0.595), 'lengths': np.arange(60, 601, 60)},
        "Medium_TVD_0.18":  {'params': (0.41, 0.59), 'lengths': np.arange(70, 701, 70)},
        "Medium_TVD_0.17":  {'params': (0.415, 0.585), 'lengths': np.arange(80, 801, 80)},
        "Hard_TVD_0.16":    {'params': (0.42, 0.58), 'lengths': np.arange(90, 901, 90)},
        "Hard_TVD_0.15":    {'params': (0.425, 0.575), 'lengths': np.arange(100, 1001, 100)},
        "Hard_TVD_0.14":    {'params': (0.43, 0.57), 'lengths': np.arange(120, 1201, 120)},
        "Hard_TVD_0.13":    {'params': (0.435, 0.565), 'lengths': np.arange(140, 1401, 140)},
        "Hard_TVD_0.12":    {'params': (0.44, 0.56), 'lengths': np.arange(160, 1601, 160)},
        "Hard_TVD_0.11":    {'params': (0.445, 0.555), 'lengths': np.arange(180, 1801, 180)},
        "Hard_TVD_0.10":    {'params': (0.45, 0.55), 'lengths': np.arange(200, 2001, 200)},

        # --- "Very Hard" Region (TVD < 0.10) ---
        "VHard_TVD_0.08":   {'params': (0.46, 0.54), 'lengths': np.arange(300, 3001, 300)},
        "VHard_TVD_0.06":   {'params': (0.47, 0.53), 'lengths': np.arange(400, 4001, 400)},
        "XHard_TVD_0.04":   {'params': (0.48, 0.52), 'lengths': np.arange(500, 5001, 500)},
        "XHard_TVD_0.03":   {'params': (0.485, 0.515), 'lengths': np.arange(750, 7501, 750)},
        "XHard_TVD_0.02":   {'params': (0.49, 0.51), 'lengths': np.arange(1000, 10001, 1000)},
    }

    # Define the test functions to be run
    # NOTE: The 'func' and 'kwargs' are no longer strictly needed by the GPU
    # simulation loop, but the names must exist when Python loads the script.
    TEST_FUNCTIONS = {
        'kl': {'func': decision_max_avg_kl_gpu, 'kwargs': {'num_outcomes': 2}},
        'mmd': {'func': decision_max_avg_mmd_gpu, 'kwargs': {'gamma': SETTINGS['gamma']}},
        #'mmd_rff': {'func': decision_max_avg_mmd_rff_gpu, 'kwargs': {'gamma': SETTINGS['gamma'], 'D': SETTINGS['D_features']}},
        'glt': {'func': decision_universal_glt_gpu, 'kwargs': {'num_outcomes': 2}}
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

    # --- MODIFIED BLOCK: Use correct keys and handle ties ---
    min_efficiencies = {}
    for test_name, test_info in SETTINGS['tests'].items():
        key = test_info['key']
        min_efficiencies[test_name] = summary_df[f'Ratio_{key.upper()}'].min(skipna=True)

    # --- NEW LOGIC TO HANDLE TIES ---
    # 1. Find the single best efficiency score
    best_score = max(min_efficiencies.values())
    
    # 2. Find all tests that achieved this best score
    best_tests = [test_name for test_name, score in min_efficiencies.items() if score == best_score]
    
    # 3. Format the conclusion based on how many winners there are
    if len(best_tests) == 1:
        conclusion_str = f"The most robust test is '{best_tests[0]}'"
    else:
        # Join all winners with " and "
        conclusion_str = f"The most robust tests (tied) are: {', '.join(best_tests)}"
    # --- END NEW LOGIC ---

    print("\n--- Maximin Results ---"); print("Worst-Case Efficiency Ratio (min c / 2*DB):")
    for test, value in min_efficiencies.items(): print(f"- {test:<19}: {value:.4f}")
    print("-" * 60); print(f"CONCLUSION (Maximin): {conclusion_str}"); print("="*60)

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