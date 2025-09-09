import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# SETTINGS
S = 5
num_trials = 400
max_steps = 10000
alpha = 0.5
C_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
sigma = np.sqrt(0.5)
anomaly_type = 'normal'  # distribution doesn't matter here since no anomaly

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

def sample_p(n): return np.random.normal(0, 1, n)  # only P used (no anomalies)

def simulate_false_alarm(alpha, C):
    false_alarms = 0
    for trial in trange(num_trials, desc=f"False Alarm C={C}"):
        streams = [[] for _ in range(S)]
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_p(1)
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
                false_alarms += 1
                break  # false alarm occurred
    prob_false_alarm = false_alarms / num_trials
    return -np.log10(max(prob_false_alarm, 1e-10)), prob_false_alarm  # avoid log(0)

# === Folder to save ===
folder_name = input("Enter folder name to save false alarm results: ").strip()
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# === Run for all C values ===
results = []
for C in C_values:
    log_false_alarm, raw_prob = simulate_false_alarm(alpha, C)
    results.append({'C': C, '-log10 ℙ₀[δ ≠ ∅]': round(log_false_alarm, 4), 'ℙ₀[δ ≠ ∅]': round(raw_prob, 6)})

# === Save Excel ===
df = pd.DataFrame(results)
excel_file = os.path.join(folder_name, f"{folder_name}_false_alarm.xlsx")
df.to_excel(excel_file, index=False)
print(f"\nSaved false alarm results to: {excel_file}")

# === Plot ===
plt.figure(figsize=(7, 5))
xs = df['-log10 ℙ₀[δ ≠ ∅]']
ys = df['C']
plt.plot(xs, ys, marker='o', color='red')
plt.xlabel(r"$-\log_{10} \mathbb{P}_0[\delta \neq \emptyset]$", fontsize=13)
plt.ylabel(r"C", fontsize=13)
plt.title(r"False Alarm Rate vs C (NP-SEQ, $\alpha=0.5$)", fontsize=14)
plt.grid(True)
plt.tight_layout()

plot_file = os.path.join(folder_name, f"{folder_name}_false_alarm.png")
plt.savefig(plot_file)
plt.show()
print(f"Saved plot to: {plot_file}")
