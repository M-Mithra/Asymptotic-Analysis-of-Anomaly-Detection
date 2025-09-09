import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# SETTINGS FROM THE PAPER
# ----------------------------
S = 5
num_trials = 100000  # Increased to match paper more closely
max_steps = 100
sigma = np.sqrt(0.5)
C_values = [10,15,20,25,30,35,40]

# ----------------------------
# Gaussian kernel used:
#   k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
# ----------------------------

def compute_mmd2_fast(x, y, sigma):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    n = len(x)
    if n < 2:
        return 0.0
    Kxx = np.exp(-((x - x.T)**2) / (2 * sigma**2))
    Kyy = np.exp(-((y - y.T)**2) / (2 * sigma**2))
    Kxy = np.exp(-((x - y.T)**2) / (2 * sigma**2))
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (np.sum(Kxx) + np.sum(Kyy)) / (n*(n-1)) - 2 * np.sum(Kxy) / (n*n)

# Distribution samplers
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n): return np.random.normal(1.2, 1, n)

# NP-SEQ(alpha, ∞) Test with extended print statements
def np_seq_test(alpha, C):
    stop_times, errors = [], []
    print(f"\nRunning NP-SEQ(α={alpha}, ∞), C = {C}")
    for trial in range(num_trials):
        if trial % 200 == 0:
            print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]

        for n in range(2, max_steps):
            if n % 50 == 0 and trial % 200 == 0:
                print(f"    Trial {trial + 1}, Step {n}/{max_steps}")
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
                if trial % 200 == 0:
                    print(f"    ✓ STOP at step {n}, γₙ = {gamma_n:.4f}, τₙ = {threshold:.4f}, candidate = {candidate}, true = {anomaly_idx}")
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                break
        else:
            if trial % 200 == 0:
                print(f"    ✗ MAX steps reached. No decision made.")
            stop_times.append(max_steps)
            errors.append(1)
    return np.array(stop_times), np.array(errors)

# Run NP-SEQ tests for alpha = 1.0, 0.5, 2.0
results_seq_1, results_seq_2 = [], []
for C in C_values:
    for alpha, result_list in zip([1.0, 2.0], [results_seq_1, results_seq_2]):
        times, errs = np_seq_test(alpha, C)
        pmax = max(np.mean(errs), 1e-5)  # Prevent log10(0)
        result_list.append((np.log10(1 / pmax), np.mean(times)))

# Plotting the results
plt.figure(figsize=(6, 5))
x1, y1 = zip(*results_seq_1)
x2, y2 = zip(*results_seq_2)
plt.plot(x2, y2, 'mo-', label='NP-SEQ(2, ∞)')
plt.plot(x1, y1, 'r*-', label='NP-SEQ-1')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=12)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=12)
plt.title(r"(a) $\mathcal{N}(1.2,1)$", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()