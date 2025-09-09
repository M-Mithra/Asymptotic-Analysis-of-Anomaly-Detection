import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 5000
batch_size = 100
max_steps = 10000
sigma = np.sqrt(0.5)
C_values_alpha_1 = [0.5,0.6,0.7,0.8,0.9,1,1.5]
C_values_alpha_05 = [0.5,0.6,0.7,0.8,0.9,1,1.5]
C_values_alpha_2 = [0.5,0.6,0.7,0.8,0.9,1,1.5]
alphas = [1.0,0.5,2]
anomaly_type = 'normal'  # replace with whatever the distribution we want to test and also make sure to add that distribution definition in the sample funciton

def compute_mmd2_fast(x, y, sigma):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    n = len(x)
    m = len(y)
    if n < 2 or m < 2:
        return 0.0
    Kxx = np.exp(-((x - x.T) ** 2) / (2 * sigma ** 2))
    Kyy = np.exp(-((y - y.T) ** 2) / (2 * sigma ** 2))
    Kxy = np.exp(-((x - y.T) ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (np.sum(Kxx) / (n * (n - 1)) +
            np.sum(Kyy) / (m * (m - 1)) -
            2 * np.sum(Kxy) / (n * m))

#functions for generating samples in sequences
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n):
    if anomaly_type == 'normal':
        return np.random.normal(1.2, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(0, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type.")

#test function
def np_seq_test(alpha, C, num_trials, max_steps, batch_size):
    batch_results = []
    stop_times, errors, thresholds, mmds = [], [], [], []
    for trial in trange(num_trials, desc=f"\u03B1={alpha}, C={C}"):
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
        trial_thresholds = []
        stopped = False
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_q(1) if i == anomaly_idx else sample_p(1)
                streams[i].append(x)
            mmd_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = np.concatenate(others)
                mmd_vals[i] = compute_mmd2_fast(streams[i], flat_others, sigma)
            candidate = np.argmax(mmd_vals)
            gamma_n = mmd_vals[candidate]
            threshold = C / (n ** alpha)
            trial_thresholds.append(round(float(threshold), 4))
            if gamma_n > threshold:
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                thresholds.append(round(float(threshold), 4))
                mmds.append(round(float(gamma_n), 4))
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
            thresholds.append(round(float(trial_thresholds[-1]), 4) if trial_thresholds else None)
            mmds.append(round(float(gamma_n), 4) if mmds else None)

        # Store batch results every 'batch_size' trials or at the end
        if ((trial + 1) % batch_size == 0) or (trial + 1 == num_trials):
            batch_start = trial + 2 - batch_size if (trial + 1) % batch_size == 0 else (trial + 1) - ((trial + 1) % batch_size) + 1
            batch_end = trial + 1
            batch_stop_times = stop_times[-batch_size:]
            batch_errors = errors[-batch_size:]
            batch_thresholds = thresholds[-batch_size:]
            batch_mmds = mmds[-batch_size:]
            pmax = max(np.mean(batch_errors), 1e-8) #here i took 1e-8, can we go still lesser value?.. probably if we want to enter into greater regime.. can go till 1e-12
            mean_time = np.mean(batch_stop_times)
            log_pmax = np.log10(1 / pmax)
            avg_mmd_batch = np.mean([x for x in batch_mmds if x is not None])
            batch_results.append({
                'alpha': alpha,
                'C': C,
                'trial_range': f"{batch_start}-{batch_end}",
                '-log10_pmax': round(log_pmax, 4),
                'E[N]': round(mean_time, 4),
                'mmd_list': batch_mmds,
                'avg_mmd_batch': round(avg_mmd_batch, 4),
                'avg_mmd_all': '',  # to be filled in average row
                'Tn_list': batch_thresholds
            })
    # Compute averages for all trials for this (alpha, C)
    avg_logpmax = np.log10(1 / max(np.mean(errors), 1e-8))
    avg_en = np.mean(stop_times)
    avg_mmd_all = np.mean([x for x in mmds if x is not None])
    return batch_results, avg_logpmax, avg_en, avg_mmd_all

# ask user for filename
filename_base = input("Enter the base folder/filename to save results (no extension): ").strip()
if not os.path.exists(filename_base):
    os.makedirs(filename_base)

# simulate and save data
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
        batch_results, avg_logpmax, avg_en, avg_mmd_all = np_seq_test(alpha, C, num_trials, max_steps, batch_size)
        all_results.extend(batch_results)
        # add average row for this (alpha, C)
        all_results.append({
            'alpha': alpha,
            'C': C,
            'trial_range': 'AVERAGE',
            '-log10_pmax': round(avg_logpmax, 4),
            'E[N]': round(avg_en, 4),
            'mmd_list': '',
            'avg_mmd_batch': '',
            'avg_mmd_all': round(avg_mmd_all, 4),
            'Tn_list': ''
        })
        # add blank row
        all_results.append({
            'alpha': '',
            'C': '',
            'trial_range': '',
            '-log10_pmax': '',
            'E[N]': '',
            'mmd_list': '',
            'avg_mmd_batch': '',
            'avg_mmd_all': '',
            'Tn_list': ''
        })
        # for plotting: use the average over all trials
        plot_points.append((alpha, C, avg_logpmax, avg_en))

# save results to Excel
results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}/{filename_base}.xlsx"
results_df.to_excel(excel_filename, index=False)
print(f"\nSaved results to {excel_filename}")
print(results_df.head())

# plot results: one point per (alpha, C), using the average over all trials
plt.figure(figsize=(7, 5))
markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
for alpha in alphas:
    xs = [logp for a, c, logp, en in plot_points if a == alpha]
    ys = [en for a, c, logp, en in plot_points if a == alpha]
    plt.plot(xs, ys, marker=markers[alpha], label=f'NP-SEQ($\\alpha={alpha}$)')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$" +
          (r" ($\mathcal{N}(1.2,1)$ anomaly)" if anomaly_type == 'normal' else r" (Laplace anomaly)"),
          fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"{filename_base}/{filename_base}.png"
plt.savefig(plot_filename)
plt.show()
print(f"Saved plot to {plot_filename}")
 