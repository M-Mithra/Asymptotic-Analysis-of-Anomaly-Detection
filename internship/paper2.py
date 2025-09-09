import numpy as np
import matplotlib.pyplot as plt

# Parameters
S = 5
num_trials = 500
max_steps = 200
C_values = [5, 10, 20, 40, 80]
sigma = 0.5

# Distributions
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q_normal(n): return np.random.normal(1.2, 1, n)
def sample_q_laplace(n): return np.random.laplace(0, 1 / np.sqrt(2), n)

# Fast unbiased MMD^2 estimator
def compute_mmd2_fast(x, y, sigma=0.5):
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    n = len(x)
    if n < 2: return 0.0
    Kxx = np.exp(-((x - x.T)**2) / (2 * sigma**2))
    Kyy = np.exp(-((y - y.T)**2) / (2 * sigma**2))
    Kxy = np.exp(-((x - y.T)**2) / (2 * sigma**2))
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (np.sum(Kxx) + np.sum(Kyy)) / (n*(n-1)) - 2 * np.sum(Kxy) / (n*n)

# NP-SEQ(α, ∞)
def np_seq_test(p_sampler, q_sampler, C, alpha=1.0):
    stop_times, errors = [], []
    print(f"\nNP-SEQ test running with alpha = {alpha}, C = {C}")

    for trial in range(1, num_trials + 1):
        if trial % 100 == 0: print(f"  Trial {trial}/{num_trials}")
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]

        for n in range(2, max_steps):
            if n % 100 == 0: print(f"    Step {n}/{max_steps}")
            for i in range(S):
                sample = q_sampler(1) if i == anomaly_idx else p_sampler(1)
                streams[i].append(sample[0])

            mmd_matrix = np.zeros((S, S))
            for i in range(S):
                for j in range(S):
                    if i != j:
                        mmd_matrix[i][j] = compute_mmd2_fast(streams[i], streams[j])

            min_vals = [min([mmd_matrix[i][j] for j in range(S) if j != i]) for i in range(S)]
            candidate = np.argmax(min_vals)
            gamma_n = min_vals[candidate]

            Tn = C / (n**alpha)
            if gamma_n > Tn:
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                break
        else:
            stop_times.append(max_steps)
            errors.append(1)

    return np.array(stop_times), np.array(errors)

# NP-FSS Test (Fixed sample size)
def np_fss_test(p_sampler, q_sampler, n):
    errors = []
    print(f"\nNP-FSS test running with sample size n = {n}")

    for trial in range(1, num_trials + 1):
        if trial % 100 == 0: print(f"  Trial {trial}/{num_trials}")
        anomaly_idx = np.random.randint(S)
        streams = [q_sampler(n) if i == anomaly_idx else p_sampler(n) for i in range(S)]

        # Estimate MMD²(i, others)
        mmd_vals = []
        for i in range(S):
            others = np.concatenate([streams[j] for j in range(S) if j != i])
            mmd_val = compute_mmd2_fast(streams[i], others)
            mmd_vals.append(mmd_val)

        candidate = np.argmax(mmd_vals)
        errors.append(0 if candidate == anomaly_idx else 1)

    return np.mean(errors)

# Main experiment
def run_all_tests(p_sampler, q_sampler):
    results_seq1, results_seq05, results_seq2, results_fss = [], [], [], []
    for C in C_values:
        # NP-SEQ-1
        times, errs = np_seq_test(p_sampler, q_sampler, C, alpha=1.0)
        safe_p = max(np.mean(errs), 1e-5)
        results_seq1.append((np.log10(1 / safe_p), np.mean(times)))

        # NP-SEQ(0.5)
        times, errs = np_seq_test(p_sampler, q_sampler, C, alpha=0.5)
        safe_p = max(np.mean(errs), 1e-5)
        results_seq05.append((np.log10(1 / safe_p), np.mean(times)))

        # NP-SEQ(2.0)
        times, errs = np_seq_test(p_sampler, q_sampler, C, alpha=2.0)
        safe_p = max(np.mean(errs), 1e-5)
        results_seq2.append((np.log10(1 / safe_p), np.mean(times)))

        # NP-FSS using n = mean stopping time from NP-SEQ-1
        n_fss = int(np.mean(times))
        fss_error = np_fss_test(p_sampler, q_sampler, n=n_fss)
        safe_fss = max(fss_error, 1e-5)
        results_fss.append((np.log10(1 / safe_fss), n_fss))

    return results_seq1, results_seq05, results_seq2, results_fss

# Plotting function
def plot_all(results_list, title):
    labels = ["NP-SEQ-1", "NP-SEQ(0.5)", "NP-SEQ(2.0)", "NP-FSS"]
    styles = ["o-", "s--", "d-.", "x:"]

    plt.figure(figsize=(8, 6))
    for results, label, style in zip(results_list, labels, styles):
        x, y = zip(*results)
        plt.plot(x, y, style, label=label)
    plt.xlabel(r"$-\log_{10} P_{\max}$")
    plt.ylabel(r"$\mathbb{E}[N]$")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run everything for Gaussian and Laplace q
print("=== Running with q = N(1.2,1) ===")
res1 = run_all_tests(sample_p, sample_q_normal)
plot_all(res1, r"Fig 1a: q = $\mathcal{N}(1.2,1)$")

print("=== Running with q = Laplace(0, 1/√2) ===")
res2 = run_all_tests(sample_p, sample_q_laplace)
plot_all(res2, r"Fig 1b: q = Laplace(0, $1/\sqrt{2}$)")
