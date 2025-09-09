import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 500
max_steps = 10000
sigma = np.sqrt(0.5)
C_values_alpha_1 = [1,1.5, 2,2.5, 3,3.5, 4,4.5,5,5.5]
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

def np_seq_test(alpha, C, num_trials, max_steps):
    stop_times = []
    errors = []
    for trial in trange(num_trials, desc=f"α={alpha}, C={C}"):
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
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
            if gamma_n > threshold:
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
    return np.array(stop_times), np.array(errors)

# RUN SIMULATION AND SAVE DATA
results = {alpha: [] for alpha in alphas}
all_results = []
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
        times, errs = np_seq_test(alpha, C, num_trials, max_steps)
        pmax = max(np.mean(errs), 1e-8)
        mean_time = np.mean(times)
        log_pmax = np.log10(1 / pmax)
        results[alpha].append((log_pmax, mean_time))
        all_results.append({
            'alpha': alpha,
            'C': C,
            '-log10_pmax': log_pmax,
            'E[N]': mean_time,
            'threshold_schedule': f'C / n^{alpha}'
        })

# Save results to Excel
results_df = pd.DataFrame(all_results)
results_df.to_excel('np_seq_simulation_results.xlsx', index=False)
print("Saved results to np_seq_simulation_results.xlsx")
print(results_df.head())

# Plot results
plt.figure(figsize=(7, 5))
markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
for alpha in alphas:
    x, y = zip(*results[alpha])
    plt.plot(x, y, marker=markers[alpha], label=f'NP-SEQ($\\alpha={alpha}$)')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$" +
          (r" ($\mathcal{N}(1.2,1)$ anomaly)" if anomaly_type == 'normal' else r" (Laplace anomaly)"),
          fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
