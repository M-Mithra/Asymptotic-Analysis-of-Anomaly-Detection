import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 1000
max_steps = 10000
batch_size = 200
alpha = 0.5
C_values = [0.01,0.02,0.05,0.1,0.3,0.4, 0.5,0.9, 1.0, 1.2,1.3]
sigma = np.sqrt(0.5)
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

def np_seq_multiA(alpha, C, A):
    stop_times, mis_idents = [], []
    for trial in trange(num_trials, desc=f"A={A}, C={C}"):
        anomaly_indices = np.random.choice(S, A, replace=False)
        streams = [[] for _ in range(S)]
        stopped = False
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_q(1) if i in anomaly_indices else sample_p(1)
                streams[i].append(x)
            mmd_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = np.concatenate(others)
                mmd_vals[i] = compute_mmd2_fast(streams[i], flat_others, sigma)
            top_A_indices = np.argsort(mmd_vals)[-A:]
            gamma_n = mmd_vals[top_A_indices[-1]]
            threshold = C / (n ** alpha)
            if gamma_n > threshold:
                stop_times.append(n)
                correct = set(anomaly_indices) == set(top_A_indices)
                mis_idents.append(0 if correct else 1)
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            mis_idents.append(1)
    mean_stop = np.mean(stop_times)
    mis_prob = np.mean(mis_idents)
    return mean_stop, -np.log10(max(mis_prob, 1e-10))  # avoid log(0)

# ==== Output folder setup ====
folder_name = input("Enter folder name to save results: ").strip()
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# ==== Run for A = 1 and A = 2 ====
results = {1: [], 2: []}
plot_data = []
for A in [1, 2]:
    for C in C_values:
        mean_stop, log_mis = np_seq_multiA(alpha, C, A)
        results[A].append({
            'alpha': alpha,
            'A': A,
            'C': C,
            'E[N]': round(mean_stop, 4),
            '-log10 P_I[δ ≠ I]': round(log_mis, 4)
        })
        plot_data.append((A, C, mean_stop, log_mis))

# ==== Save to Excel ====
df_all = pd.DataFrame([row for A_rows in results.values() for row in A_rows])
excel_file = os.path.join(folder_name, f"{folder_name}.xlsx")
df_all.to_excel(excel_file, index=False)
print(f"\nSaved Excel file to: {excel_file}")

# ==== Plot ====
plt.figure(figsize=(7, 5))
for A in [1, 2]:
    xs = [log_mis for a, c, en, log_mis in plot_data if a == A]
    ys = [en for a, c, en, log_mis in plot_data if a == A]
    marker = 'o' if A == 1 else 's'
    plt.plot(xs, ys, marker=marker, label=f"A = {A}")
plt.xlabel(r"$-\log_{10} \mathbb{P}_I[\delta \neq I]$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: Mis-identification vs Expected Stopping Time", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_file = os.path.join(folder_name, f"{folder_name}.png")
plt.savefig(plot_file)
plt.show()
print(f"Saved plot to: {plot_file}")
