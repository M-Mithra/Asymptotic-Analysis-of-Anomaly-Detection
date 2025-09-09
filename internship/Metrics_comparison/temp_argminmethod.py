import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# ===== SETTINGS =====
S = 3
num_trials = 3000
max_steps = 10000
alphas = [0.5, 1.0, 2.0]

alpha_to_C = {
    0.5: [0.5, 0.7, 1, 2, 2.5, 3],
    1.0: [1, 2, 3, 4, 5, 6, 7, 8, 8.5],
    2.0: [10, 50, 100, 120, 150, 200, 210, 220]
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

def empirical_pmf_vectorized(streams):
    """streams: list of lists, one per sequence"""
    counts = np.zeros((S, k))
    for i, seq in enumerate(streams):
        for s in seq:
            counts[i, symbol_to_int[s]] += 1
    return counts / np.maximum(1, counts.sum(axis=1, keepdims=True))

def squared_distance_matrix(types, centroids):
    """Compute squared distances between each row in types and its centroid"""
    return np.sum((types - centroids)**2, axis=1)

# ===== SAVE FOLDER =====
filename_base = input("Enter base folder/filename to save results (no extension): ").strip()
if not os.path.exists(filename_base):
    os.makedirs(filename_base)

# ===== SIMULATION: vectorized LOO sum-of-distances =====
all_results = []

for alpha in alphas:
    C_vals = alpha_to_C.get(alpha, [])
    for C in C_vals:
        stop_times_LOO = np.zeros(num_trials)
        errors_LOO = np.zeros(num_trials)

        for trial in trange(num_trials, desc=f"α={alpha}, C={C}"):
            anomaly_idx = np.random.randint(S)
            streams = [[] for _ in range(S)]
            stopped = False

            # Initialize types array
            types = np.zeros((S, k))

            for n in range(2, max_steps):
                # Sample one symbol per sequence
                for i in range(S):
                    x = sample_q(1) if i == anomaly_idx else sample_p(1)
                    streams[i].append(x[0])

                # Vectorized empirical PMFs
                types = empirical_pmf_vectorized(streams)

                # ===== Vectorized LOO sum-of-distances =====
                D = np.zeros(S)
                for i in range(S):
                    # centroid of all sequences except i
                    mask = np.arange(S) != i
                    centroid = np.mean(types[mask, :], axis=0)
                    # sum distances of remaining sequences to centroid
                    D[i] = np.sum(squared_distance_matrix(types[mask, :], centroid))

                # candidate is sequence whose removal minimizes sum
                candidate = np.argmin(D)
                threshold = C / (n ** alpha)
                if D[candidate] > threshold:
                    stop_times_LOO[trial] = n
                    errors_LOO[trial] = 0 if candidate == anomaly_idx else 1
                    stopped = True
                    break

            if not stopped:
                stop_times_LOO[trial] = max_steps
                errors_LOO[trial] = 1

        # Record results
        all_results.append({
            'alpha': alpha,
            'C': C,
            'method': 'LOO_sum_distance',
            'E[N]': np.mean(stop_times_LOO),
            'error': np.mean(errors_LOO)
        })

# Save results
results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}/{filename_base}_LOO_vectorized.xlsx"
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
plt.title("LOO sum-of-distances method: E[N] vs -log10(Error)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{filename_base}/{filename_base}_LOO_vectorized_plot.png")
plt.show()
