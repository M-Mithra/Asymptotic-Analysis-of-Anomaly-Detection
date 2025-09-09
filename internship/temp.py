import numpy as np
import matplotlib.pyplot as plt

# Parameters
S = 5
num_trials = 10000
max_steps = 100
sigma = 0.5
C_values = [5, 10, 20, 40]

# Distribution samplers
def sample_p(n): return np.random.normal(0, 1, n)
def sample_q(n): return np.random.normal(1.2, 1, n)

# Fast unbiased MMD² estimator
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
def np_seq_test(alpha, C):
    print(f"\n>>> Running NP-SEQ(alpha={alpha}, C={C})")
    stop_times, errors = [], []
    for trial in range(1, num_trials + 1):
        if trial % 100 == 0:
            print(f"  Trial {trial}/{num_trials}")
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
        for n in range(2, max_steps):
            if trial % 100 == 0 and n % 20 == 0:
                print(f"    Step {n}/{max_steps}")
            for i in range(S):
                x = sample_q(1)[0] if i == anomaly_idx else sample_p(1)[0]
                streams[i].append(x)
            mmd_vals = np.zeros(S)
            for i in range(S):
                others = [streams[j] for j in range(S) if j != i]
                flat_others = np.concatenate(others)
                mmd_vals[i] = compute_mmd2_fast(streams[i], flat_others)
            candidate = np.argmax(mmd_vals)
            gamma_n = mmd_vals[candidate]
            if gamma_n > C / (n ** alpha):
                stop_times.append(n)
                errors.append(0 if candidate == anomaly_idx else 1)
                break
        else:
            stop_times.append(max_steps)
            errors.append(1)
    return np.array(stop_times), np.array(errors)

# NP-FSS test
def np_fss_test(n):
    print(f"\n>>> Running NP-FSS test with fixed sample size n = {n}")
    errors = []
    for trial in range(1, num_trials + 1):
        if trial % 100 == 0:
            print(f"  FSS Trial {trial}/{num_trials}")
        anomaly_idx = np.random.randint(S)
        streams = [sample_q(n) if i == anomaly_idx else sample_p(n) for i in range(S)]
        mmd_vals = []
        for i in range(S):
            others = np.concatenate([streams[j] for j in range(S) if j != i])
            mmd_vals.append(compute_mmd2_fast(streams[i], others))
        candidate = np.argmax(mmd_vals)
        errors.append(0 if candidate == anomaly_idx else 1)
    return np.mean(errors)

# Run all tests and collect results
def run_all_tests():
    res_1, res_05, res_2, res_fss = [], [], [], []
    for C in C_values:
        print(f"\n===== Starting tests for C = {C} =====")
        for alpha, res_list in zip([1.0, 0.5, 2.0], [res_1, res_05, res_2]):
            times, errs = np_seq_test(alpha, C)
            pmax = max(np.mean(errs), 1e-5)
            res_list.append((np.log10(1/pmax), np.mean(times)))

        fss_n = int(np.mean(times))
        fss_err = np_fss_test(fss_n)
        res_fss.append((np.log10(1/max(fss_err, 1e-5)), fss_n))
    return res_1, res_05, res_2, res_fss

# Plot results
def plot_results(res_1, res_05, res_2, res_fss):
    x1, y1 = zip(*res_1)
    x05, y05 = zip(*res_05)
    x2, y2 = zip(*res_2)
    xfss, yfss = zip(*res_fss)
    plt.figure(figsize=(6, 5))
    plt.plot(x05, y05, 'go-', label='NP-SEQ(0.5, ∞)')
    plt.plot(xfss, yfss, 'bs-', label='NP-FSS')
    plt.plot(x2, y2, 'mo-', label='NP-SEQ(2, ∞)')
    plt.plot(x1, y1, 'r*-', label='NP-SEQ-1')
    plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=12)
    plt.ylabel(r"$\mathbb{E}[N]$", fontsize=12)
    plt.title(r"(a) $\mathcal{N}(1.2,1)$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run everything
res_1, res_05, res_2, res_fss = run_all_tests()
plot_results(res_1, res_05, res_2, res_fss)
