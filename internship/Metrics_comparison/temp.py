import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# ===== SETTINGS =====
S = 5
num_trials = 3000
batch_size = 500
max_steps = 10000

alphas = [0.5, 1.0, 2.0]

alpha_to_C = {
    0.5: [0.5, 0.7, 1, 2, 2.5, 3],
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

# ===== SIMULATION =====
all_results = []

for alpha in alphas:
    C_vals = alpha_to_C.get(alpha, [])
    for C in C_vals:
        stop_times_Cd, errors_Cd = [], []
        stop_times_Nd, errors_Nd = [], []

        for trial in trange(num_trials, desc=f"α={alpha}, C={C}"):
            anomaly_idx = np.random.randint(S)
            streams = [[] for _ in range(S)]
            stopped_Cd = False
            stopped_Nd = False

            for n in range(2, max_steps):
                # Sample one symbol per stream
                for i in range(S):
                    x = sample_q(1) if i == anomaly_idx else sample_p(1)
                    streams[i].append(x[0])

                # Compute types
                types = [empirical_pmf(streams[i]) for i in range(S)]

                # Centroid distances
                Cd = np.zeros(S)
                for i in range(S):
                    centroid = np.mean([types[j] for j in range(S) if j != i], axis=0)
                    Cd[i] = squared_distance(types[i], centroid)
                if not stopped_Cd:
                    candidate_Cd = np.argmax(Cd)
                    if Cd[candidate_Cd] > C / (n**alpha):
                        stop_times_Cd.append(n)
                        errors_Cd.append(0 if candidate_Cd == anomaly_idx else 1)
                        stopped_Cd = True

                # NP-SEQ distances
                Nd = np.zeros(S)
                for i in range(S):
                    pooled = np.concatenate([streams[j] for j in range(S) if j != i])
                    pooled_type = empirical_pmf(pooled)
                    Nd[i] = squared_distance(types[i], pooled_type)
                if not stopped_Nd:
                    candidate_Nd = np.argmax(Nd)
                    if Nd[candidate_Nd] > C / (n**alpha):
                        stop_times_Nd.append(n)
                        errors_Nd.append(0 if candidate_Nd == anomaly_idx else 1)
                        stopped_Nd = True

                if stopped_Cd and stopped_Nd:
                    break

            # If never stopped
            if not stopped_Cd:
                stop_times_Cd.append(max_steps)
                errors_Cd.append(1)
            if not stopped_Nd:
                stop_times_Nd.append(max_steps)
                errors_Nd.append(1)

        # Record results
        all_results.append({
            'alpha': alpha, 'C': C, 'method': 'Centroid',
            'E[N]': np.mean(stop_times_Cd), 'error': np.mean(errors_Cd)
        })
        all_results.append({
            'alpha': alpha, 'C': C, 'method': 'NP-SEQ',
            'E[N]': np.mean(stop_times_Nd), 'error': np.mean(errors_Nd)
        })

results_df = pd.DataFrame(all_results)
excel_filename = f"{filename_base}/{filename_base}.xlsx"
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

# ===== PLOT Centroid vs α =====
plt.figure(figsize=(7,5))
for alpha in alphas:
    subset = results_df[(results_df['method']=='Centroid') & (results_df['alpha']==alpha)]
    plt.plot(-np.log10(subset['error']), subset['E[N]'], marker='o', linestyle='-', label=f'α={alpha}')
plt.xlabel(r"$-\log_{10} \mathrm{error}$")
plt.ylabel(r"$\mathbb{E}[N]$")
plt.title("Centroid Method: E[N] vs -log10(Error) for different α")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{filename_base}/{filename_base}_centroid_alpha.png")
plt.show()

# ===== PLOT NP-SEQ vs α =====
plt.figure(figsize=(7,5))
for alpha in alphas:
    subset = results_df[(results_df['method']=='NP-SEQ') & (results_df['alpha']==alpha)]
    plt.plot(-np.log10(subset['error']), subset['E[N]'], marker='s', linestyle='-', label=f'α={alpha}')
plt.xlabel(r"$-\log_{10} \mathrm{error}$")
plt.ylabel(r"$\mathbb{E}[N]$")
plt.title("NP-SEQ Method: E[N] vs -log10(Error) for different α")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{filename_base}/{filename_base}_npseq_alpha.png")
plt.show()

# ===== PLOT Centroid vs NP-SEQ comparison for each α =====
for alpha in alphas:
    plt.figure(figsize=(7,5))
    subset_Cd = results_df[(results_df['method']=='Centroid') & (results_df['alpha']==alpha)]
    subset_Nd = results_df[(results_df['method']=='NP-SEQ') & (results_df['alpha']==alpha)]
    plt.plot(-np.log10(subset_Cd['error']), subset_Cd['E[N]'], marker='o', linestyle='-', label='Centroid')
    plt.plot(-np.log10(subset_Nd['error']), subset_Nd['E[N]'], marker='s', linestyle='--', label='NP-SEQ')
    plt.xlabel(r"$-\log_{10} \mathrm{error}$")
    plt.ylabel(r"$\mathbb{E}[N]$")
    plt.title(f"Centroid vs NP-SEQ Comparison (α={alpha})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}/{filename_base}_comparison_alpha_{alpha}.png")
    plt.show()
