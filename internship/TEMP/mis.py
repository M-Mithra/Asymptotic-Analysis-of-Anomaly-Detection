import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random

# --- Sampling distributions ---
def sample_p(n):  # Nominal
    return np.random.normal(loc=0.0, scale=1.0, size=(n, 1))

def sample_q(n):  # Anomalous
    return np.random.normal(loc=1.2, scale=1.0, size=(n, 1))

# --- Gaussian Kernel ---
def gaussian_kernel(x, y, sigma2=1.0):
    return np.exp(-np.sum((x - y)**2) / (2 * sigma2))

def compute_mmd2(X, Y, sigma2=1.0):
    n, m = len(X), len(Y)
    XX = np.mean([gaussian_kernel(X[i], X[j], sigma2) for i in range(n) for j in range(n) if i != j])
    YY = np.mean([gaussian_kernel(Y[i], Y[j], sigma2) for i in range(m) for j in range(m) if i != j])
    XY = np.mean([gaussian_kernel(X[i], Y[j], sigma2) for i in range(n) for j in range(m)])
    return XX + YY - 2 * XY

# --- Main NP-SEQ-UA Simulation ---
def np_seq_ua(C, A, S, max_steps, trials, anomalous_dist, false_alarm=False):
    mis_id, false_alarm_count = 0, 0

    for _ in range(trials):
        if false_alarm:
            anomaly_indices = []
        else:
            anomaly_indices = sorted(random.sample(range(S), A))

        sequences = []
        for i in range(S):
            dist = anomalous_dist if i in anomaly_indices else sample_p
            sequences.append(dist(10))  # init with 10 samples

        declared, t = False, 1
        while not declared and t <= max_steps:
            mmd_scores = {}
            for A_hat in combinations(range(S), A):
                indices = list(A_hat)
                min_mmd = float('inf')
                for i in range(A):
                    for j in range(i + 1, A):
                        mmd = compute_mmd2(sequences[indices[i]], sequences[indices[j]])
                        min_mmd = min(min_mmd, mmd)
                mmd_scores[A_hat] = min_mmd

            best_subset = max(mmd_scores, key=mmd_scores.get)
            stat = mmd_scores[best_subset]
            threshold = (C * np.log(t)) / t

            if stat > threshold:
                declared = True
                if false_alarm:
                    if len(best_subset) > 0:
                        false_alarm_count += 1
                else:
                    if set(best_subset) != set(anomaly_indices):
                        mis_id += 1
                break
            else:
                for i in range(S):
                    dist = anomalous_dist if i in anomaly_indices else sample_p
                    sequences[i] = np.vstack([sequences[i], dist(1)])
                t += 1

    if false_alarm:
        return false_alarm_count / trials
    else:
        return mis_id / trials

# --- Run simulation for multiple C values ---
C_values = [0.5, 1.0, 1.5, 2.0]
S = 5
A = 2
max_steps = 200
trials = 300

false_alarm_rates = []
mis_id_rates = []

for C in C_values:
    fa = np_seq_ua(C, A, S, max_steps, trials, sample_q, false_alarm=True)
    mis = np_seq_ua(C, A, S, max_steps, trials, sample_q, false_alarm=False)
    false_alarm_rates.append(fa)
    mis_id_rates.append(mis)

# --- Convert to -log10 scale for x-axis ---
neg_log10_FA = [-np.log10(x) if x > 0 else 0 for x in false_alarm_rates]
neg_log10_MIS = [-np.log10(x) if x > 0 else 0 for x in mis_id_rates]
neg_log10_Pmax = [-np.log10((C * np.log(max_steps)) / max_steps) for C in C_values]

# --- Plot False Alarm ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(neg_log10_Pmax, neg_log10_FA, marker='o', color='red')
plt.xlabel(r'$-\log_{10} \left( \frac{C \log t}{t} \right)$')
plt.ylabel(r'$-\log_{10} \, \mathbb{P}_0[\delta \neq \emptyset]$')
plt.title('False Alarm Rate vs $-\log_{10} P_{\max}$')
plt.grid(True)

# --- Plot Misidentification ---
plt.subplot(1, 2, 2)
plt.plot(neg_log10_Pmax, neg_log10_MIS, marker='o', color='blue')
plt.xlabel(r'$-\log_{10} \left( \frac{C \log t}{t} \right)$')
plt.ylabel(r'$-\log_{10} \, \mathbb{P}_{\mathcal{I}}[\delta \ne \mathcal{I}]$')
plt.title('Misidentification Rate vs $-\log_{10} P_{\max}$')
plt.grid(True)

plt.tight_layout()
plt.show()
