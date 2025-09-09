import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

# SETTINGS
S = 5  # number of streams
num_trials = 500
batch_size = 250
m = 100  # fixed sample size per stream for FSS test
sigma = np.sqrt(0.5)  # bandwidth for Gaussian kernel in MMD

C = 2.0  # threshold multiplier (can be tuned)

typical_mu, typical_sigma = 0.0, 1.0
anomaly_mu, anomaly_sigma = 1.2, 1.0

def compute_mmd2_fast(x, y, sigma):
    """Compute unbiased estimate of squared MMD between samples x and y."""
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
    mmd2 = (np.sum(Kxx) / (n * (n - 1)) +
            np.sum(Kyy) / (m * (m - 1)) -
            2 * np.sum(Kxy) / (n * m))
    return mmd2

def sample_typical(n):
    return np.random.normal(typical_mu, typical_sigma, n)

def sample_anomalous(n):
    return np.random.normal(anomaly_mu, anomaly_sigma, n)

def fss_test(num_trials, batch_size, m, S, C, sigma):
    batch_results = []
    errors = []
    stop_trials = []  # For FSS, stop time = m fixed
    mmd_values_all = []
    thresholds_all = []
    
    for trial in trange(num_trials, desc=f"FSS test: Trials"):
        anomaly_idx = np.random.randint(S)

        # Sample fixed m points per stream
        streams = []
        for i in range(S):
            if i == anomaly_idx:
                streams.append(sample_anomalous(m))
            else:
                streams.append(sample_typical(m))
        
        # Compute MMD for each stream against rest combined
        mmd_vals = np.zeros(S)
        for i in range(S):
            others = [streams[j] for j in range(S) if j != i]
            flat_others = np.concatenate(others)
            mmd_vals[i] = compute_mmd2_fast(streams[i], flat_others, sigma)
        
        # Determine candidate anomaly: stream with max MMD
        candidate = np.argmax(mmd_vals)
        gamma_m = mmd_vals[candidate]
        
        # Threshold for fixed sample size test can be tuned; here simple
        threshold = C / (m ** 0.5)  # Example scaling, can adjust
        exceeded = gamma_m > threshold
        
        stop_trials.append(m)  # Fixed sample size test: always m
        
        # Error: 0 if correct anomaly detected, else 1
        error = 0 if candidate == anomaly_idx and exceeded else 1
        errors.append(error)
        
        mmd_values_all.append(round(float(gamma_m), 5))
        thresholds_all.append(round(float(threshold), 5))
        
        # Store batch results periodically
        if ((trial + 1) % batch_size == 0) or (trial + 1 == num_trials):
            batch_start = trial + 2 - batch_size if (trial + 1) % batch_size == 0 else (trial + 1) - ((trial + 1) % batch_size) + 1
            batch_end = trial + 1
            
            batch_errors = errors[-batch_size:]
            batch_mmds = mmd_values_all[-batch_size:]
            batch_thresholds = thresholds_all[-batch_size:]
            
            pmax = max(np.mean(batch_errors), 1e-8)  # Avoid log(0)
            mean_time = np.mean(stop_trials[-batch_size:])
            log_pmax = np.log10(1 / pmax)
            avg_mmd_batch = np.mean(batch_mmds)
            
            batch_results.append({
                'trial_range': f"{batch_start}-{batch_end}",
                '-log10_pmax': round(log_pmax, 5),
                'E[N]': round(mean_time, 5),
                'avg_mmd': round(avg_mmd_batch, 5),
                'pmax': round(np.mean(batch_errors), 5),
                'mmd_list': batch_mmds,
                'threshold_list': batch_thresholds
            })

    # Overall stats
    avg_logpmax = np.log10(1 / max(np.mean(errors), 1e-8))
    avg_en = np.mean(stop_trials)
    avg_mmd_all = np.mean(mmd_values_all)
    
    return batch_results, avg_logpmax, avg_en, avg_mmd_all

if __name__ == "__main__":
    # Ask user folder for saving results
    filename_base = input("Enter folder name to save results (will be created if not exist): ").strip()
    if not os.path.exists(filename_base):
        os.makedirs(filename_base)
    
    print("Starting Fixed Sample Size (FSS) Detection Simulation ...")
    batch_results, avg_logpmax, avg_en, avg_mmd_all = fss_test(num_trials, batch_size, m, S, C, sigma)
    
    # Save results to Excel
    df = pd.DataFrame(batch_results)
    
    avg_row = {
        'trial_range': 'AVERAGE',
        '-log10_pmax': round(avg_logpmax, 5),
        'E[N]': round(avg_en, 5),
        'avg_mmd': round(avg_mmd_all, 5),
        'pmax': '',
        'mmd_list': '',
        'threshold_list': ''
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    excel_path = os.path.join(filename_base, "fss_test_results.xlsx")
    df.to_excel(excel_path, index=False)
    
    print(f"\nSimulation complete. Results saved to {excel_path}")
    print(df.head())
    
    # Optional: Plot -log10(pmax) vs E[N] as just one point here because FSS has fixed N=m
    plt.figure(figsize=(6,4))
    plt.plot(avg_logpmax, avg_en, 'o', label=f'FSS test (m={m})')
    plt.xlabel(r'$-\log_{10} P_{max}$')
    plt.ylabel(r'$\mathbb{E}[N]$')
    plt.title('FSS Test: Detection Performance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(filename_base, "fss_performance.png"))
    plt.show()
