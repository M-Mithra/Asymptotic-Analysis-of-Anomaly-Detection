import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math
import os

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
    # Ensure p and q are the same length for comparison
    len_p, len_q = len(p), len(q)
    if len_p > len_q:
        q = np.pad(q, (0, len_p - len_q), 'constant')
    elif len_q > len_p:
        p = np.pad(p, (0, len_q - len_p), 'constant')
    
    bc_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bc_coefficient + 1e-10)

def estimate_alpha(lengths, error_probs):
    """Robustly estimates the error exponent alpha from simulation results."""
    valid_indices = [i for i, p in enumerate(error_probs) if p > 0]
    if len(valid_indices) < 4: return np.nan
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

def run_simulation(experiment_name, data_generator, test_logic, settings):
    """
    Runs a full Monte Carlo simulation for a single experiment.
    """
    lengths = settings['lengths']
    trials = settings['trials']
    batch_size = settings['batch_size']
    
    detailed_errors = {'kl': [], 'mmd': [], 'tvd': [], 'lrt': []}

    for L in lengths:
        err_counts = {'kl': 0, 'mmd': 0, 'tvd': 0, 'lrt': 0}
        print(f"    Running L={L:<3}...", end="")
        for trial in range(trials):
            if (trial + 1) % batch_size == 0:
                print(f" {int(100*(trial+1)/trials)}%", end="", flush=True)

            # Generate data for this specific experiment
            seq1, seq2, seq3 = data_generator(L)
            
            # Apply all 4 test logics
            if test_logic['kl'](seq1, seq2, seq3) != 0: err_counts['kl'] += 1
            if test_logic['mmd'](seq1, seq2, seq3, settings['gamma']) != 0: err_counts['mmd'] += 1
            if test_logic['tvd'](seq1, seq2, seq3) != 0: err_counts['tvd'] += 1
            if test_logic['lrt'](seq1, seq2, seq3) != 0: err_counts['lrt'] += 1
        
        print(" Done.")
        for key in detailed_errors: detailed_errors[key].append(err_counts[key] / trials)
        
    # Create a detailed DataFrame for this experiment
    results_df = pd.DataFrame({'Sequence_Length': lengths})
    for name, err_counts in [('LRT', detailed_errors['lrt']), ('KL', detailed_errors['kl']), ('MMD', detailed_errors['mmd']), ('TVD', detailed_errors['tvd'])]:
        error_probs = np.array(err_counts)
        results_df[f'Prob_Error_{name}'] = error_probs
        results_df[f'Log_Prob_Error_{name}'] = np.log(error_probs)
        
    return results_df

# ==============================================================================
# ===== DEFINITION OF EXPERIMENTS
# ==============================================================================

# --- Experiment 1: Bernoulli (Biased Coin) ---
pi_bern = 0.5; mu_bern = 0.7
pi_bern_dist = np.array([1-pi_bern, pi_bern]); mu_bern_dist = np.array([1-mu_bern, mu_bern])
def bernoulli_data_generator(L):
    return (np.random.binomial(1, mu_bern, L), 
            np.random.binomial(1, pi_bern, L), 
            np.random.binomial(1, pi_bern, L))

# --- Experiment 2: Binomial (Sum of 10 Coins) ---
n_binomial = 10; pi_bino = 0.5; mu_bino = 0.7
pi_bino_dist = [math.comb(n_binomial, k) * (pi_bino**k) * ((1-pi_bino)**(n_binomial-k)) for k in range(n_binomial + 1)]
mu_bino_dist = [math.comb(n_binomial, k) * (mu_bino**k) * ((1-mu_bino)**(n_binomial-k)) for k in range(n_binomial + 1)]
def binomial_data_generator(L):
    return (np.random.binomial(n_binomial, mu_bino, L), 
            np.random.binomial(n_binomial, pi_bino, L), 
            np.random.binomial(n_binomial, pi_bino, L))

# --- Experiment 3: Categorical (Loaded 4-Sided Die) ---
pi_cat = np.array([0.25, 0.25, 0.25, 0.25])
mu_cat = np.array([0.7, 0.1, 0.1, 0.1])
def categorical_data_generator(L):
    return (np.random.choice(4, L, p=mu_cat), 
            np.random.choice(4, L, p=pi_cat), 
            np.random.choice(4, L, p=pi_cat))
    
# ===== DEFINITION OF TEST LOGICS =====

# Create empirical probability distributions (histograms) from sequences
def get_dist(seq, num_outcomes):
    counts = np.bincount(seq, minlength=num_outcomes)
    return counts / len(seq) if len(seq) > 0 else np.zeros(num_outcomes)

def kl_test(s1, s2, s3, num_outcomes):
    P1, P2, P3 = get_dist(s1, num_outcomes), get_dist(s2, num_outcomes), get_dist(s3, num_outcomes)
    Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
    return np.argmax([kl_divergence(P1,Q1), kl_divergence(P2,Q2), kl_divergence(P3,Q3)])

def tvd_test(s1, s2, s3, num_outcomes):
    P1, P2, P3 = get_dist(s1, num_outcomes), get_dist(s2, num_outcomes), get_dist(s3, num_outcomes)
    Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
    return np.argmax([total_variation_distance(P1,Q1), total_variation_distance(P2,Q2), total_variation_distance(P3,Q3)])

# MMD works on sample means, a simpler statistic
def mmd_test(s1, s2, s3, gamma):
    t1, t2, t3 = np.mean(s1), np.mean(s2), np.mean(s3)
    return np.argmax([rbf_kernel(t2,t3,gamma), rbf_kernel(t1,t3,gamma), rbf_kernel(t1,t2,gamma)])
    
# LRT uses the true distribution parameters
def lrt_bern_test(s1, s2, s3):
    llr1 = np.sum(s1*np.log(mu_bern/pi_bern) + (1-s1)*np.log((1-mu_bern)/(1-pi_bern)))
    llr2 = np.sum(s2*np.log(mu_bern/pi_bern) + (1-s2)*np.log((1-mu_bern)/(1-pi_bern)))
    llr3 = np.sum(s3*np.log(mu_bern/pi_bern) + (1-s3)*np.log((1-mu_bern)/(1-pi_bern)))
    return np.argmax([llr1, llr2, llr3])
    
# (LRT for Binomial and Categorical are more complex, so we'll compare the other 3 for those)

# ==============================================================================
# ===== MAIN SCRIPT
# ==============================================================================

# --- General Settings ---
SETTINGS = {
    'gamma': 1.0,
    'trials': 2000,
    'batch_size': 500,
    'lengths': np.arange(50, 601, 50)
}

save_folder = input("Enter folder to save experiment results: ").strip()
os.makedirs(save_folder, exist_ok=True)
excel_writer = pd.ExcelWriter(os.path.join(save_folder, 'comparative_experiments.xlsx'), engine='openpyxl')

# Define all experiments to run
experiments = {
    "Bernoulli": {
        "data_generator": bernoulli_data_generator,
        "test_logic": {
            'kl': lambda s1,s2,s3: kl_test(s1,s2,s3,2),
            'mmd': lambda s1,s2,s3,gamma: mmd_test(s1,s2,s3,gamma),
            'tvd': lambda s1,s2,s3: tvd_test(s1,s2,s3,2),
            'lrt': lrt_bern_test
        },
        "true_dists": (pi_bern_dist, mu_bern_dist)
    },
    "Binomial": {
        "data_generator": binomial_data_generator,
        "test_logic": {
            'kl': lambda s1,s2,s3: kl_test(s1,s2,s3, n_binomial+1),
            'mmd': lambda s1,s2,s3,gamma: mmd_test(s1,s2,s3,gamma),
            'tvd': lambda s1,s2,s3: tvd_test(s1,s2,s3, n_binomial+1),
            'lrt': lambda s1,s2,s3: -1 # Placeholder for LRT
        },
        "true_dists": (np.array(pi_bino_dist), np.array(mu_bino_dist))
    },
    "Categorical": {
        "data_generator": categorical_data_generator,
         "test_logic": {
            'kl': lambda s1,s2,s3: kl_test(s1,s2,s3, 4),
            'mmd': lambda s1,s2,s3,gamma: mmd_test(s1,s2,s3,gamma),
            'tvd': lambda s1,s2,s3: tvd_test(s1,s2,s3, 4),
            'lrt': lambda s1,s2,s3: -1 # Placeholder for LRT
        },
        "true_dists": (pi_cat, mu_cat)
    }
}

# --- EXPERIMENT LOOP ---
for exp_name, exp_details in experiments.items():
    print(f"\n{'='*60}\nRunning Experiment: {exp_name}\n{'='*60}")
    
    # Run the simulation for this experiment
    scenario_df = run_simulation(exp_name, exp_details['data_generator'], exp_details['test_logic'], SETTINGS)
    scenario_df.to_excel(excel_writer, sheet_name=f"{exp_name}_Details", index=False)
    
    # --- Analyze results for this experiment ---
    summary = {"Test": [], "Estimated Alpha": [], "2*DB_Bound": [], "Efficiency Ratio": []}
    pi_dist, mu_dist = exp_details['true_dists']
    b_dist = bhattacharyya_distance(pi_dist, mu_dist)
    
    for test_name_short, test_name_long in [('LRT','LRT (Optimal)'), ('KL','KL Divergence'), ('MMD','MMD'), ('TVD','Total Variation')]:
        if test_name_short == 'lrt' and exp_name != 'Bernoulli': continue # Skip LRT for non-Bernoulli
        
        alpha = estimate_alpha(SETTINGS['lengths'], scenario_df[f'Prob_Error_{test_name_short.upper()}'].values)
        efficiency = np.minimum(1.0, alpha / (2*b_dist)) if not np.isnan(alpha) else np.nan
        
        summary['Test'].append(test_name_long)
        summary['Estimated Alpha'].append(alpha)
        summary['2*DB_Bound'].append(2*b_dist)
        summary['Efficiency Ratio'].append(efficiency)

    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(excel_writer, sheet_name=f"{exp_name}_Summary", index=False)
    
    # --- Generate Summary Plot for this experiment ---
    fig, ax = plt.subplots(figsize=(10, 7))
    summary_df.plot(x='Test', y='Efficiency Ratio', kind='bar', ax=ax, legend=False, color=['red','blue','green','purple'])
    ax.set_title(f'Test Efficiency for {exp_name} Experiment', fontsize=16)
    ax.set_ylabel('Efficiency Ratio (alpha / 2*DB)', fontsize=12)
    ax.set_xlabel('Test Method', fontsize=12)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Theoretical Maximum')
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plot_path = os.path.join(save_folder, f'summary_plot_{exp_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()

excel_writer.close()
print(f"\nProcess complete! All experiments saved to 'comparative_experiments.xlsx' and plots saved in '{save_folder}'")