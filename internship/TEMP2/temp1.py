import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 3000
batch_size = 200
max_steps = 10000

alphas = [1.0, 0.5, 2]

alpha_to_C = {
    1.0: [1, 2, 3, 4, 5, 6, 7, 8, 8.5],
    0.5: [0.5, 0.7, 1, 2, 2.5, 3],
    2.0: [10, 50, 100, 120, 150, 200, 210, 220]
}

# Define discrete alphabet
alphabet = ['a', 'b', 'c']
k = len(alphabet)
symbol_to_int = {s: i for i, s in enumerate(alphabet)}

# Define PMFs
p_typical = np.array([0.7, 0.2, 0.1])
p_anomalous = np.array([0.2, 0.5, 0.3])

def sample_p(n):
    return np.random.choice(alphabet, size=n, p=p_typical)

def sample_q(n):
    return np.random.choice(alphabet, size=n, p=p_anomalous)

def empirical_pmf(samples):
    counts = np.zeros(k)
    for s in samples:
        counts[symbol_to_int[s]] += 1
    return counts / max(1, len(samples))

def compute_mmd2_discrete(x, y):
    p1 = empirical_pmf(x)
    p2 = empirical_pmf(y)
    return np.sum((p1 - p2) ** 2)

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
                streams[i].append(x[0])
            mmd_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = sum(others, [])
                mmd_vals[i] = compute_mmd2_discrete(streams[i], flat_others)
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

        if ((trial + 1) % batch_size == 0) or (trial + 1 == num_trials):
            batch_start = trial + 2 - batch_size if (trial + 1) % batch_size == 0 else (trial + 1) - ((trial + 1) % batch_size) + 1
            batch_end = trial + 1
            batch_stop_times = stop_times[-batch_size:]
            batch_errors = errors[-batch_size:]
            batch_thresholds = thresholds[-batch_size:]
            batch_mmds = mmds[-batch_size:]
            pmax = max(np.mean(batch_errors), 1e-25)
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
                'avg_mmd_all': '',
                'Tn_list': batch_thresholds
            })
    avg_logpmax = np.log10(1 / max(np.mean(errors), 1e-25))
    avg_en = np.mean(stop_times)
    avg_mmd_all = np.mean([x for x in mmds if x is not None])
    return batch_results, avg_logpmax, avg_en, avg_mmd_all

# Ask user for filename
filename_base = input("Enter the base folder/filename to save results (no extension): ").strip()
if not os.path.exists(filename_base):
    os.makedirs(filename_base)

# RUN SIMULATION AND SAVE DATA
all_results = []
plot_points = []
for alpha in alphas:
    C_vals = alpha_to_C.get(alpha, [])
    print(f"Running alpha={alpha}, C values={C_vals}")  # Debug print
    for C in C_vals:
        batch_results, avg_logpmax, avg_en, avg_mmd_all = np_seq_test(alpha, C, num_trials, max_steps, batch_size)
        all_results.extend(batch_results)
        # Add average row for this (alpha, C)
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
        # Add blank row
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
        plot_points.append((alpha, C, avg_logpmax, avg_en))

results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}/{filename_base}.xlsx"
results_df.to_excel(excel_filename, index=False)
print(f"\nSaved results to {excel_filename}")
print(results_df.head())

# Plot distributions
bar_width = 0.35
x = np.arange(len(alphabet))

plt.figure(figsize=(6, 4))
plt.bar(x - bar_width/2, p_typical, width=bar_width, label='Typical', color='royalblue')
plt.bar(x + bar_width/2, p_anomalous, width=bar_width, label='Anomalous', color='orange')
plt.xticks(x, alphabet)
plt.ylabel('Probability')
plt.title('Typical vs Anomalous Discrete Distributions')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Plot NP-SEQ results
plt.figure(figsize=(7, 5))
markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
for alpha in alphas:
    xs = [logp for a, c, logp, en in plot_points if a == alpha]
    ys = [en for a, c, logp, en in plot_points if a == alpha]
    plt.plot(xs, ys, marker=markers[alpha], label=f'NP-SEQ($\\alpha={alpha}$)')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$ (Discrete)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"{filename_base}/{filename_base}.png"
plt.savefig(plot_filename)
plt.show()
print(f"Saved plot to {plot_filename}")
