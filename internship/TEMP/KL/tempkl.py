import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import entropy

# SETTINGS
S = 5
num_trials = 5000
batch_size = 2500
max_steps = 10000
C_values_alpha_1 = [20,40,60,80]
C_values_alpha_05 = [10,20,25,30]
C_values_alpha_2 = [5,15,25,35]
alphas = [1.0, 0.5]
means_to_test = [0.8, 1.2, 2]
anomaly_type = 'normal'  # 'normal' or 'laplace'

def compute_kl_divergence(x, y, bins=20):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    min_edge = min(np.min(x), np.min(y))
    max_edge = max(np.max(x), np.max(y))
    bin_edges = np.linspace(min_edge, max_edge, bins+1)
    px, _ = np.histogram(x, bins=bin_edges, density=True)
    py, _ = np.histogram(y, bins=bin_edges, density=True)
    px = px + 1e-10
    py = py + 1e-10
    return entropy(px, py)  # KL(px || py)

def sample_p(n): 
    return np.random.normal(0, 1, n)

def sample_q_param(n, mean):
    if anomaly_type == 'normal':
        return np.random.normal(mean, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(mean, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type.")

def np_seq_test(alpha, C, num_trials, max_steps, batch_size, anomaly_mean):
    batch_results = []
    stop_times, errors, thresholds, klvals = [], [], [], []
    for trial in trange(num_trials, desc=f"\u03B1={alpha}, C={C}, mean={anomaly_mean}"):
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
        trial_thresholds = []
        stopped = False
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_q_param(1, anomaly_mean) if i == anomaly_idx else sample_p(1)
                streams[i].append(x)
            kl_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = np.concatenate(others)
                kl_vals[i] = compute_kl_divergence(streams[i], flat_others)
            candidate = np.argmax(kl_vals)
            gamma_n = kl_vals[candidate]
            threshold = C / (n ** alpha)
            trial_thresholds.append(round(float(threshold), 4))
            if gamma_n > threshold:
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                thresholds.append(round(float(threshold), 4))
                klvals.append(round(float(gamma_n), 4))
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
            thresholds.append(round(float(trial_thresholds[-1]), 4) if trial_thresholds else None)
            klvals.append(round(float(gamma_n), 4) if klvals else None)
        # Store batch results every 'batch_size' trials or at the end
        if ((trial + 1) % batch_size == 0) or (trial + 1 == num_trials):
            batch_start = trial + 2 - batch_size if (trial + 1) % batch_size == 0 else (trial + 1) - ((trial + 1) % batch_size) + 1
            batch_end = trial + 1
            batch_stop_times = stop_times[-batch_size:]
            batch_errors = errors[-batch_size:]
            batch_thresholds = thresholds[-batch_size:]
            batch_klvals = klvals[-batch_size:]
            pmax = max(np.mean(batch_errors), 1e-8)
            mean_time = np.mean(batch_stop_times)
            log_pmax = np.log10(1 / pmax)
            avg_kl_batch = np.mean([x for x in batch_klvals if x is not None])
            batch_results.append({
                'alpha': alpha,
                'C': C,
                'mean': anomaly_mean,
                'trial_range': f"{batch_start}-{batch_end}",
                '-log10_pmax': round(log_pmax, 4),
                'E[N]': round(mean_time, 4),
                'kl_list': batch_klvals,
                'avg_kl_batch': round(avg_kl_batch, 4),
                'avg_kl_all': '',  # to be filled in average row
                'Tn_list': batch_thresholds
            })
    # Compute averages for all trials for this (alpha, C, mean)
    avg_logpmax = np.log10(1 / max(np.mean(errors), 1e-8))
    avg_en = np.mean(stop_times)
    avg_kl_all = np.mean([x for x in klvals if x is not None])
    return batch_results, avg_logpmax, avg_en, avg_kl_all

# Ask user for filename
filename_base = input("Enter the base folder/filename to save KL results (no extension): ").strip()
if not os.path.exists(filename_base):
    os.makedirs(filename_base)

# RUN SIMULATION & STORE RESULTS
all_plot_points = dict()  # {mean: [(alpha, C, avg_logpmax, avg_en)]}
for mean_val in means_to_test:
    print(f"Running KL-divergence simulations for anomaly_mean = {mean_val} ...")
    all_results = []
    plot_points = []
    for alpha in alphas:
        if alpha == 1.0:
            C_vals = C_values_alpha_1
        elif alpha == 0.5:
            C_vals = C_values_alpha_05
        elif alpha == 2.0:
            C_vals = C_values_alpha_2
        else:
            C_vals = []
        for C in C_vals:
            batch_results, avg_logpmax, avg_en, avg_kl_all = np_seq_test(
                alpha, C, num_trials, max_steps, batch_size, mean_val)
            all_results.extend(batch_results)
            # Add average row for this (alpha, C, mean)
            all_results.append({
                'alpha': alpha,
                'C': C,
                'mean': mean_val,
                'trial_range': 'AVERAGE',
                '-log10_pmax': round(avg_logpmax, 4),
                'E[N]': round(avg_en, 4),
                'kl_list': '',
                'avg_kl_batch': '',
                'avg_kl_all': round(avg_kl_all, 4),
                'Tn_list': ''
            })
            # Add blank row
            all_results.append({
                'alpha': '', 'C': '', 'mean': '', 'trial_range': '', '-log10_pmax': '', 'E[N]': '',
                'kl_list': '', 'avg_kl_batch': '', 'avg_kl_all': '', 'Tn_list': ''
            })
            plot_points.append((alpha, C, avg_logpmax, avg_en))
    all_plot_points[mean_val] = plot_points
    # Save results to Excel per mean
    results_df = pd.DataFrame(all_results)
    excel_filename = f"{filename_base}/simulation_results_kl_mean_{mean_val}.xlsx"
    results_df.to_excel(excel_filename, index=False)
    # Plot per-mean: for all alphas
    plt.figure(figsize=(7, 5))
    markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
    for alpha in alphas:
        xs = [logp for a, c, logp, en in plot_points if a == alpha]
        ys = [en   for a, c, logp, en in plot_points if a == alpha]
        plt.plot(xs, ys, marker=markers.get(alpha,'o'), label=f'NP-SEQ($\\alpha={alpha}$)')
    plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
    plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
    plt.title(r"KL-NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$"
              + f"\n(anomaly mean={mean_val}, {'Normal' if anomaly_type=='normal' else 'Laplace'})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"{filename_base}/plot_kl_mean_{mean_val}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot to {plot_filename}")

# Overlay comparison, one plot per alpha: all means
for alpha in alphas:
    plt.figure(figsize=(8, 6))
    for mean_val in means_to_test:
        plot_points = all_plot_points[mean_val]
        xs = [logp for a, c, logp, en in plot_points if a == alpha]
        ys = [en for a, c, logp, en in plot_points if a == alpha]
        plt.plot(xs, ys, marker='o', label=f'Mean = {mean_val}')
    plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
    plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
    plt.title(rf"KL-NP-SEQ $-\log_{{10}} P_{{max}}$ vs $\mathbb{{E}}[N]$"
              + f"\n($\\alpha={alpha}$, {'Normal' if anomaly_type=='normal' else 'Laplace'})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"{filename_base}/plot_kl_alpha_{alpha}_all_means.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved overlay comparison plot to {plot_filename}")
