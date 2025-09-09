import numpy as np
import matplotlib.pyplot as plt
import time

# SETTINGS
S = 5
num_trials = 1000
max_steps = 100
sigma = np.sqrt(0.5)
C_values = [0.5,0.6,0.7,0.8,0.9,1.0,1.5,2,2.5,2.6,2.7,2.8,2.9,3,3.5,4]
alphas = [1.0, 0.5]
anomaly_type = 'normal'
verbose = True  # Set to False to suppress detailed outputs

# KERNEL & MMD
def compute_mmd2_fast(x, y, sigma):
    # x, y: 1D numpy arrays
    n = len(x)
    m = len(y)
    if n < 2 or m < 2:
        return 0.0
    Kxx = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sigma ** 2))
    Kyy = np.exp(-((y[:, None] - y[None, :]) ** 2) / (2 * sigma ** 2))
    Kxy = np.exp(-((x[:, None] - y[None, :]) ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (np.sum(Kxx) / (n * (n - 1)) +
            np.sum(Kyy) / (m * (m - 1)) -
            2 * np.sum(Kxy) / (n * m))

# DISTRIBUTION SAMPLERS
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n):
    if anomaly_type == 'normal':
        return np.random.normal(1.2, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(0, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type.")

# NP-SEQ(α, ∞) Test (optimized)
def np_seq_test(alpha, C, num_trials, max_steps):
    stop_times = np.zeros(num_trials, dtype=int)
    errors = np.zeros(num_trials, dtype=int)
    for trial in range(num_trials):
        if verbose and trial % (num_trials // 5) == 0:
            print(f"  Trial {trial+1}/{num_trials}")
        anomaly_idx = np.random.randint(S)
        # Pre-allocate for speed
        streams = np.zeros((S, max_steps))
        # Fill first sample for all streams
        streams[:, 0] = sample_p(S)
        streams[anomaly_idx, 0] = sample_q(1)
        stopped = False
        for n in range(2, max_steps):
            # Add next sample for all streams
            streams[:, n] = sample_p(S)
            streams[anomaly_idx, n] = sample_q(1)
            # Compute MMD for each stream vs rest
            mmd_vals = np.zeros(S)
            for i in range(S):
                x = streams[i, :n+1]
                others = np.delete(streams[:, :n+1], i, axis=0).flatten()
                mmd_vals[i] = compute_mmd2_fast(x, others, sigma)
            candidate = np.argmax(mmd_vals)
            gamma_n = mmd_vals[candidate]
            threshold = C / (n ** alpha)
            if gamma_n > threshold:
                stop_times[trial] = n+1
                errors[trial] = 0 if candidate == anomaly_idx else 1
                if verbose and trial % 100 == 0:
                    print(f"    ✓ Trial {trial+1}: Stopped at n = {n+1}, γₙ = {gamma_n:.4f}, "
                          f"Tₙ = {threshold:.4f}, Candidate = {candidate}, True = {anomaly_idx}")
                stopped = True
                break
        if not stopped:
            stop_times[trial] = max_steps
            errors[trial] = 1
            if verbose and trial % 100 == 0:
                print(f"    ✗ Trial {trial+1}: Reached max steps ({max_steps}), anomaly was at index {anomaly_idx}")
    return stop_times, errors

# RUN SIMULATION
results = {alpha: [] for alpha in alphas}
start_time = time.time()
for alpha in alphas:
    print(f"\n▶ Starting simulations for α = {alpha}")
    for C in C_values:
        print(f"   → Running with threshold constant C = {C}")
        times, errs = np_seq_test(alpha, C, num_trials, max_steps)
        pmax = max(np.mean(errs), 1e-8)
        mean_time = np.mean(times)
        results[alpha].append((np.log10(1 / pmax), mean_time))
        print(f"      ↪ Done: log10(1/Pmax) = {np.log10(1 / pmax):.4f},  E[N] = {mean_time:.2f}")
end_time = time.time()
print(f"\nTotal simulation time: {end_time - start_time:.1f} seconds")

# PLOT RESULTS
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