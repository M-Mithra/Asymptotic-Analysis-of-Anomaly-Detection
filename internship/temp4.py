import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# SETTINGS
# ----------------------------
S = 5
num_trials = 2000    # Adjust for your needs
max_steps = 10000   # Large value to approximate infinity
sigma = np.sqrt(0.5)
C_values = [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2]
alphas = [1.0, 0.5]
anomaly_type = 'normal'  # 'normal' or 'laplace'
verbose = True           # Set to False to suppress detailed outputs

# ----------------------------
# KERNEL & MMD
# ----------------------------
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

# ----------------------------
# DISTRIBUTION SAMPLERS
# ----------------------------
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n):
    if anomaly_type == 'normal':
        return np.random.normal(1.2, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(0, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type.")

# ----------------------------
# NP-SEQ(α, ∞) Test
# ----------------------------
def np_seq_test(alpha, C, num_trials):
    stop_times = []
    errors = []
    for trial in range(num_trials):
        if verbose and trial % (num_trials // 5) == 0:
            print(f"  Trial {trial+1}/{num_trials}")
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
                if verbose and trial % 100 == 0:
                    print(f"    ✓ Trial {trial+1}: Stopped at n = {n}, γₙ = {gamma_n:.4f}, "
                          f"Tₙ = {threshold:.4f}, Candidate = {candidate}, True = {anomaly_idx}")
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
            if verbose and trial % 100 == 0:
                print(f"    ✗ Trial {trial+1}: Reached max steps ({max_steps}), anomaly was at index {anomaly_idx}")
    return np.array(stop_times), np.array(errors)

# ----------------------------
# RUN SIMULATION
# ----------------------------
results = {alpha: [] for alpha in alphas}
for alpha in alphas:
    print(f"\n▶ Starting simulations for α = {alpha}")
    for C in C_values:
        print(f"   → Running with threshold constant C = {C}")
        times, errs = np_seq_test(alpha, C, num_trials)
        pmax = max(np.mean(errs), 1e-8)
        mean_time = np.mean(times)
        results[alpha].append((np.log10(1 / pmax), mean_time))
        print(f"      ↪ Done: log10(1/Pmax) = {np.log10(1 / pmax):.4f},  E[N] = {mean_time:.2f}")

# ----------------------------
# PLOT RESULTS
# ----------------------------
plt.figure(figsize=(7, 5))
for alpha in alphas:
    x, y = zip(*results[alpha])
    plt.plot(x, y, marker='o', label=f'NP-SEQ($\\alpha={alpha}$)')
plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
plt.title(r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$" +
          (r" ($\mathcal{N}(1.2,1)$ anomaly)" if anomaly_type == 'normal' else r" (Laplace anomaly)"),
          fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#C_values = [0.5,1,1.1,1.2,1.3,1.4,1.5,2] temp4 fig2, trails 1000