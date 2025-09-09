import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 500
batch_size = 100
max_steps = 10000
sigma = np.sqrt(0.5)
C_values_alpha_1 = [1,1.5, 2,2.5, 3,3.5, 4,4.5,4.6]
C_values_alpha_05 = [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2]
C_values_alpha_2 = [1,5,10,15,20,25,30,35,40,45,50]
alphas = [1.0, 0.5, 2.0]
anomaly_type = 'normal'  # 'normal' or 'laplace'

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

def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n):
    if anomaly_type == 'normal':
        return np.random.normal(1.2, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(0, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type.")

def np_seq_test(alpha, C, num_trials, max_steps, batch_size):
    batch_results = []
    stop_times, errors, gammas, thresholds = [], [], [], []
    for trial in trange(num_trials, desc=f"α={alpha}, C={C}"):
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
        trial_gammas = []
        trial_thresholds = []
        stopped = False
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_q(1)[0] if i == anomaly_idx else sample_p(1)[0]
                streams[i].append(x)
            mmd_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = np.concatenate(others)
                mmd_vals[i] = compute_mmd2_fast(streams[i], flat_others, sigma)
            candidate = np.argmax(mmd_vals)
            gamma_n = mmd_vals[candidate]
            threshold = C / (n ** alpha)
            trial_gammas.append(gamma_n)
            trial_thresholds.append(threshold)
            if gamma_n > threshold:
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                gammas.append(gamma_n)
                thresholds.append(threshold)
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
            gammas.append(trial_gammas[-1] if trial_gammas else None)
            thresholds.append(trial_thresholds[-1] if trial_thresholds else None)

        # Store batch results every batch_size trials or at the end
        if ((trial + 1) % batch_size == 0) or (trial + 1 == num_trials):
            batch_start = trial + 2 - batch_size if (trial + 1) % batch_size == 0 else (trial + 1) - ((trial + 1) % batch_size) + 1
            batch_end = trial + 1
            batch_stop_times = stop_times[-batch_size:]
            batch_errors = errors[-batch_size:]
            batch_gammas = gammas[-batch_size:]
            batch_thresholds = thresholds[-batch_size:]
            pmax = max(np.mean(batch_errors), 1e-8)
            mean_time = np.mean(batch_stop_times)
            log_pmax = np.log10(1 / pmax)
            batch_results.append({
                'trial_range': f"{batch_start}-{batch_end}",
                'alpha': alpha,
                'C': C,
                '-log10_pmax': log_pmax,
                'E[N]': mean_time,
                'gamma_n_list': batch_gammas,
                'Tn_list': batch_thresholds
            })
    return batch_results

# Ask user for filename
filename_base = input("Enter the base filename to save results (no extension): ").strip()

# RUN SIMULATION AND SAVE DATA
all_results = []
plot_points = {alpha: [] for alpha in alphas}
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
        batch_results = np_seq_test(alpha, C, num_trials, max_steps, batch_size)
        all_results.extend(batch_results)
        # For plotting, use the last batch for each (alpha, C)
        plot_points[alpha].append((batch_results[-1]['-log10_pmax'], batch_results[-1]['E[N]']))

# Save results to Excel
results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}.xlsx"
results_df.to_excel(excel_filename, index=False)
print(f"\nSaved results to {excel_filename}")
print(results_df.head())

# Plot results
plt.figure(figsize=(7, 5))
markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
for alpha in alphas:
    x, y = zip(*plot_points[alpha])
    plt.plot(x, y, marker=markers[alpha], label=f'NP-SEQ($\\alpha={alpha}$)')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$" +
          (r" ($\mathcal{N}(1.2,1)$ anomaly)" if anomaly_type == 'normal' else r" (Laplace anomaly)"),
          fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"{filename_base}.png"
plt.savefig(plot_filename)
plt.show()
print(f"Saved plot to {plot_filename}")
