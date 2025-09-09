import numpy as np

# Settings
S = 5                 # Number of streams
num_trials = 200      # Number of trials (increase for more reliable stats)
max_steps = 10000     # Max time steps per trial
sigma = np.sqrt(0.5)  # Gaussian kernel width for MMD
alpha = 1.0           # Exponent for threshold decay (tune as needed)
C = 3                 # Threshold scaling (tune as needed)
anomaly_type = 'laplace'  # 'normal' or 'laplace'

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

def sample_p(n):
    return np.random.normal(0, 1, n)

def sample_q(n):
    if anomaly_type == 'normal':
        return np.random.normal(1.2, 1, n)
    elif anomaly_type == 'laplace':
        return np.random.laplace(0, 1/np.sqrt(2), n)
    else:
        raise ValueError("Unknown anomaly_type")

def np_seq_test_simple(alpha, C, num_trials, max_steps):
    stop_times = []
    errors = []
    for trial in range(num_trials):
        anomaly_idx = np.random.randint(S)
        streams = [[] for _ in range(S)]
        stopped = False
        for n in range(2, max_steps):
            for i in range(S):
                x = sample_q(1) if i == anomaly_idx else sample_p(1)
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
                stopped = True
                break
        if not stopped:
            stop_times.append(max_steps)
            errors.append(1)
    return stop_times, errors

# Example usage
stop_times, errors = np_seq_test_simple(alpha, C, num_trials, max_steps)
mean_stop_time = np.mean(stop_times)
error_rate = np.mean(errors)
print("Average stop time:", mean_stop_time)
print("Empirical error rate:", error_rate)
