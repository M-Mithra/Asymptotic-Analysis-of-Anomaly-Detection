import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# ===== SETTINGS =====
S = 3  # number of sequences
num_trials = 3000
batch_size = 200
max_steps = 10000

alphas = [1.0, 2.0]

alpha_to_C = {
    0.5: [0.5, 0.7, 1, 2, 2.5],
    1.0: [1, 2, 3, 4, 5, 6, 7, 8],
    2.0: [10, 50, 100, 120, 150, 200]
}

# ===== ALPHABET & PMFs =====
alphabet = ['a', 'b', 'c']
k = len(alphabet)
symbol_to_int = {s: i for i, s in enumerate(alphabet)}

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

def squared_distance(p1, p2):
    return np.sum((p1 - p2)**2)

# ===== SAVE FOLDER =====
filename_base = input("Enter base folder/filename to save results (no extension): ").strip()
if not os.path.exists(filename_base):
    os.makedirs(filename_base)

# ===== SIMULATION: leave-one-out sum-of-distances method =====
all_results = []

for alpha in alphas:
    C_vals = alpha_to_C.get(alpha, [])
    for C in C_vals:
        stop_times_LOO, errors_LOO = [], []

        for trial in trange(num_trials, desc=f"α={alpha}, C={C}"):
            anomaly_idx = np.random.randint(S)
            streams = [[] for _ in range(S)]
            stopped = False

            for n in range(2, max_steps):
                # Sample one symbol per stream
                for i in range(S):
                    x = sample_q(1) if i == anomaly_idx else sample_p(1)
                    streams[i].append(x[0])

                # Compute types
                types = [empirical_pmf(streams[i]) for i in range(S)]

                # ===== Leave-one-out sum-of-distances =====
                D = np.zeros(S)
                for i in range(S):
                    # centroid of others
                    centroid = np.mean([types[j] for j in range(S) if j != i], axis=0)
                    # sum distances of remaining sequences to centroid
                    others = [types[j] for j in range(S) if j != i]
                    D[i] = sum(squared_distance(t, centroid) for t in others)

                # candidate = sequence whose removal minimizes total distance
                candidate = np.argmin(D)
                threshold = C / (n**alpha)
                if D[candidate] > threshold:
                    stop_times_LOO.append(n)
                    errors_LOO.append(0 if candidate == anomaly_idx else 1)
                    stopped = True
                    break

            # if never stopped
            if not stopped:
                stop_times_LOO.append(max_steps)
                errors_LOO.append(1)

        # record results
        all_results.append({
            'alpha': alpha,
            'C': C,
            'method': 'LOO_sum_distance',
            'E[N]': np.mean(stop_times_LOO),
            'error': np.mean(errors_LOO)
        })

results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}/{filename_base}_LOO.xlsx"
results_df.to_excel(excel_filename, index=False)
print(f"Saved results to {excel_filename}")

# ===== DISTRIBUTION HISTOGRAM =====
bar_width = 0.35
x = np.arange(len(alphabet))
plt.figure(figsize=(6,4))
plt.bar(x - bar_width/2, p_typical, width=bar_width, label='Typical', color='royalblue')
plt.bar(x + bar_width/2, p_anomalous, width=bar_width, label='Anomalous', color='orange')
plt.xticks(x, alphabet)
plt.ylabel('Probability')
plt.title('Typical vs Anomalous Discrete Distributions')
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig(f"{filename_base}/{filename_base}_distributions.png")
plt.show()

# ===== PLOT: E[N] vs error for different alpha =====
plt.figure(figsize=(7,5))
for alpha in alphas:
    subset = results_df[results_df['alpha'] == alpha]
    plt.plot(-np.log10(subset['error']), subset['E[N]'], marker='o', linestyle='-', label=f'α={alpha}')
plt.xlabel(r"$-\log_{10} \mathrm{error}$")
plt.ylabel(r"$\mathbb{E}[N]$")
plt.title("Leave-one-out sum-of-distances method: E[N] vs -log10(Error)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{filename_base}/{filename_base}_LOO_plot.png")
plt.show()
