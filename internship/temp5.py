import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import entropy

# === Global SETTINGS ===
S = 5
num_trials = 5000
batch_size = 500
max_steps = 10000
sigma = np.sqrt(0.5)
# Note: Specific C_values and alphas will be set within each function
# to avoid conflicts.

# === Core Helper Functions ===

def compute_mmd2_fast(x, y, sigma):
    """Computes the unbiased empirical MMD^2 with an RBF kernel."""
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

def compute_kl_divergence(x, y, bins=20):
    """
    Computes KL divergence between two data streams using histograms.
    Adds a small value to bins to prevent log(0).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    min_edge = min(np.min(x), np.min(y))
    max_edge = max(np.max(x), np.max(y))
    bin_edges = np.linspace(min_edge, max_edge, bins+1)
    px, _ = np.histogram(x, bins=bin_edges, density=True)
    py, _ = np.histogram(y, bins=bin_edges, density=True)
    px = px + 1e-10
    py = py + 1e-10
    return entropy(px, py)  # KL(px || py)

# Global variable to manage anomaly type across functions
_anomaly_type = 'normal'

def sample_p(n):
    """Generates typical samples based on a normal distribution."""
    return np.random.normal(0, 1, n)

def sample_q_param(n, mean, dist_type=_anomaly_type):
    """Generates anomalous samples based on a specified distribution and mean."""
    if dist_type == 'normal':
        return np.random.normal(mean, 1, n)
    elif dist_type == 'laplace':
        return np.random.laplace(mean, 1/np.sqrt(2), n)
    elif dist_type == 'uniform':
        return np.random.uniform(mean, mean + 1, n)
    else:
        raise ValueError(f"Unknown anomaly distribution type: {dist_type}.")

def plot_results(plot_points, plot_title, filename, alphas, y_label=r"$\mathbb{E}[N]$"):
    """Generates and saves a standard performance plot."""
    plt.figure(figsize=(7, 5))
    markers = {1.0: 'o', 0.5: 's', 2.0: '^'}
    for alpha in alphas:
        xs = [logp for a, c, logp, en in plot_points if a == alpha]
        ys = [en for a, c, logp, en in plot_points if a == alpha]
        plt.plot(xs, ys, marker=markers.get(alpha, 'o'), label=f'$\\alpha={alpha}$')
    plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.title(plot_title, fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

# === Generic Simulation Helper Function ===

def _run_test_case(alphas, C_values_map, num_trials, max_steps, batch_size,
                   anomaly_mean, dist_type, metric_type, multi_anomaly=False, A=1):
    """
    A single, generic function to handle all simulation cases.
    """
    global _anomaly_type
    original_anomaly_type = _anomaly_type
    _anomaly_type = dist_type

    all_results = []
    plot_points = []
    
    for alpha in alphas:
        C_vals = C_values_map.get(alpha, [])
        for C in C_vals:
            stop_times, errors = [], []
            for trial in trange(num_trials, desc=f"Running Test (alpha={alpha}, C={C}, mean={anomaly_mean}, metric={metric_type})"):
                if multi_anomaly:
                    anomaly_indices = np.random.choice(S, A, replace=False)
                else:
                    anomaly_indices = [np.random.randint(S)]

                streams = [[] for _ in range(S)]
                stopped = False

                for n in range(2, max_steps):
                    for i in range(S):
                        is_anomaly = i in anomaly_indices
                        x = sample_q_param(1, anomaly_mean, dist_type) if is_anomaly else sample_p(1)
                        streams[i].append(x)
                    
                    metric_vals_n = np.zeros(S)
                    for i in range(S):
                        others = [streams[j] for j in range(S) if j != i]
                        flat_others = np.concatenate(others)
                        if metric_type == 'mmd':
                            metric_vals_n[i] = compute_mmd2_fast(streams[i], flat_others, sigma)
                        elif metric_type == 'kl':
                            metric_vals_n[i] = compute_kl_divergence(streams[i], flat_others)

                    if multi_anomaly:
                        candidate_indices = np.argsort(metric_vals_n)[-A:]
                        is_correct = set(candidate_indices) == set(anomaly_indices)
                    else:
                        candidate_index = np.argmax(metric_vals_n)
                        is_correct = candidate_index == anomaly_indices[0]

                    gamma_n = np.max(metric_vals_n)
                    threshold = C / (n ** alpha)
                    
                    if gamma_n > threshold:
                        stop_times.append(n)
                        errors.append(0 if is_correct else 1)
                        stopped = True
                        break
                
                if not stopped:
                    stop_times.append(max_steps)
                    errors.append(1)

            avg_logpmax = np.log10(1 / max(np.mean(errors), 1e-8))
            avg_en = np.mean(stop_times)
            plot_points.append((alpha, C, avg_logpmax, avg_en))
            all_results.append({'alpha': alpha, 'C': C, 'mean': anomaly_mean, 'log10_pmax': avg_logpmax, 'E[N]': avg_en})
    
    _anomaly_type = original_anomaly_type # Revert global setting
    return all_results, plot_points


# === Simulation Functions (NP-SEQ Variants) ===

def run_np_seq_gaussian_normal():
    """Simulates NP-SEQ for single anomaly with normal or laplace distributions."""
    C_values_alpha_1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]
    C_values_alpha_05 = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]
    C_values_alpha_2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]
    alphas = [1.0, 0.5, 2]
    anomaly_mean = 1.2
    
    filename_base = "np_seq_gaussian_normal"
    if not os.path.exists(filename_base):
        os.makedirs(filename_base)

    C_values_map = {1.0: C_values_alpha_1, 0.5: C_values_alpha_05, 2.0: C_values_alpha_2}
    all_results, plot_points = _run_test_case(
        alphas=alphas, C_values_map=C_values_map, num_trials=num_trials,
        max_steps=max_steps, batch_size=batch_size, anomaly_mean=anomaly_mean,
        dist_type='normal', metric_type='mmd', multi_anomaly=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(f"{filename_base}/{filename_base}.xlsx", index=False)
    plot_title = r"NP-SEQ: $-\log_{10} P_{\max}$ vs $\mathbb{E}[N]$" + r" ($\mathcal{N}(1.2,1)$ anomaly)"
    plot_filename = f"{filename_base}/{filename_base}.png"
    plot_results(plot_points, plot_title, plot_filename, alphas)

def run_np_seq_multi_anomaly():
    """Simulates NP-SEQ for multiple anomalies."""
    alpha = 0.5
    C_values = [0.01, 0.02, 0.05, 0.1, 0.3, 0.4, 0.5, 0.9, 1.0, 1.2, 1.3]
    As = [1, 2]
    anomaly_mean = 1.2
    
    filename_base = "np_seq_multi_anomaly"
    if not os.path.exists(filename_base):
        os.makedirs(filename_base)

    all_results = []
    plot_data = []

    for A in As:
        C_values_map = {alpha: C_values}
        results, points = _run_test_case(
            alphas=[alpha], C_values_map=C_values_map, num_trials=num_trials,
            max_steps=max_steps, batch_size=batch_size, anomaly_mean=anomaly_mean,
            dist_type='normal', metric_type='mmd', multi_anomaly=True, A=A)
        
        for res in results:
            res['A'] = A
        for p in points:
            plot_data.append((A, p[1], p[3], p[2])) # A, C, E[N], log_pmax
        all_results.extend(results)
    
    df_all = pd.DataFrame(all_results)
    excel_file = os.path.join(filename_base, f"{filename_base}.xlsx")
    df_all.to_excel(excel_file, index=False)
    print(f"\nSaved Excel file to: {excel_file}")

    plt.figure(figsize=(7, 5))
    for A in As:
        xs = [log_mis for a, c, en, log_mis in plot_data if a == A]
        ys = [en for a, c, en, log_mis in plot_data if a == A]
        marker = 'o' if A == 1 else 's'
        plt.plot(xs, ys, marker=marker, label=f"A = {A}")
    plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
    plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
    plt.title(r"NP-SEQ: Mis-identification vs Expected Stopping Time", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(filename_base, f"{filename_base}.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Saved plot to: {plot_file}")

def run_np_seq_uniform_dist():
    """Simulates NP-SEQ with uniform distributions and varying means."""
    C_values_alpha_1 = [1, 1.5, 2, 2.5, 3]
    C_values_alpha_05 = [0.5, 0.6, 1, 1.2, 1.5]
    C_values_alpha_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = [1.0, 0.5]
    means_to_test = [0.5, 1]

    filename_base = "np_seq_uniform_dist"
    if not os.path.exists(filename_base):
        os.makedirs(filename_base)

    all_plot_points = dict()
    C_values_map = {1.0: C_values_alpha_1, 0.5: C_values_alpha_05, 2.0: C_values_alpha_2}
    
    for mean_val in means_to_test:
        all_results, plot_points = _run_test_case(
            alphas=alphas, C_values_map=C_values_map, num_trials=num_trials,
            max_steps=max_steps, batch_size=batch_size, anomaly_mean=mean_val,
            dist_type='uniform', metric_type='mmd', multi_anomaly=False)

        all_plot_points[mean_val] = plot_points
        
        results_df = pd.DataFrame(all_results)
        excel_filename = f"{filename_base}/simulation_results_uniform_mean_{mean_val}.xlsx"
        results_df.to_excel(excel_filename, index=False)
        plot_title = rf"MMD-NP-SEQ: $-\log_{{10}} P_{{max}}$ vs $\mathbb{{E}}[N]$" + f"\n(anomaly mean={mean_val}, Uniform)"
        plot_filename = f"{filename_base}/plot_mmd_uniform_mean_{mean_val}.png"
        plot_results(plot_points, plot_title, plot_filename, alphas)

    # Overlay plot
    for alpha in alphas:
        plt.figure(figsize=(8, 6))
        for mean_val in means_to_test:
            plot_points = all_plot_points[mean_val]
            xs = [logp for a, c, logp, en in plot_points if a == alpha]
            ys = [en for a, c, logp, en in plot_points if a == alpha]
            plt.plot(xs, ys, marker='o', label=f'Mean = {mean_val}')
        plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
        plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
        plt.title(rf"MMD-NP-SEQ $-\log_{{10}} P_{{max}}$ vs $\mathbb{{E}}[N]$" + f"\n($\\alpha={alpha}$, Uniform)", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_filename = f"{filename_base}/plot_mmd_uniform_alpha_{alpha}_all_means.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved overlay comparison plot to {plot_filename}")


def run_np_seq_kl_divergence():
    """Simulates NP-SEQ using KL-Divergence as the metric."""
    C_values_alpha_1 = [20, 40, 60, 80]
    C_values_alpha_05 = [10, 20, 25, 30]
    C_values_alpha_2 = [5, 15, 25, 35]
    alphas = [1.0, 0.5]
    means_to_test = [0.8, 1.2, 2]

    filename_base = "np_seq_kl_divergence"
    if not os.path.exists(filename_base):
        os.makedirs(filename_base)

    all_plot_points = dict()
    C_values_map = {1.0: C_values_alpha_1, 0.5: C_values_alpha_05, 2.0: C_values_alpha_2}
    
    for mean_val in means_to_test:
        all_results, plot_points = _run_test_case(
            alphas=alphas, C_values_map=C_values_map, num_trials=num_trials,
            max_steps=max_steps, batch_size=batch_size, anomaly_mean=mean_val,
            dist_type='normal', metric_type='kl', multi_anomaly=False)

        all_plot_points[mean_val] = plot_points
        
        results_df = pd.DataFrame(all_results)
        excel_filename = f"{filename_base}/simulation_results_kl_mean_{mean_val}.xlsx"
        results_df.to_excel(excel_filename, index=False)
        plot_title = rf"KL-NP-SEQ: $-\log_{{10}} P_{{max}}$ vs $\mathbb{{E}}[N]$" + f"\n(anomaly mean={mean_val}, Normal)"
        plot_filename = f"{filename_base}/plot_kl_mean_{mean_val}.png"
        plot_results(plot_points, plot_title, plot_filename, alphas)

    # Overlay plot
    for alpha in alphas:
        plt.figure(figsize=(8, 6))
        for mean_val in means_to_test:
            plot_points = all_plot_points[mean_val]
            xs = [logp for a, c, logp, en in plot_points if a == alpha]
            ys = [en for a, c, logp, en in plot_points if a == alpha]
            plt.plot(xs, ys, marker='o', label=f'Mean = {mean_val}')
        plt.xlabel(r"$-\log_{10} P_{\max}$", fontsize=13)
        plt.ylabel(r"$\mathbb{E}[N]$", fontsize=13)
        plt.title(rf"KL-NP-SEQ $-\log_{{10}} P_{{max}}$ vs $\mathbb{{E}}[N]$" + f"\n($\\alpha={alpha}$, Normal)", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_filename = f"{filename_base}/plot_kl_alpha_{alpha}_all_means.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved overlay comparison plot to {plot_filename}")


# === Main Execution Block ===
if __name__ == "__main__":
    print("Select a simulation to run:")
    print("1. Single Anomaly (Normal/Laplace)")
    print("2. Multiple Anomalies")
    print("3. Uniform Distributions")
    print("4. KL Divergence Metric")
    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        run_np_seq_gaussian_normal()
    elif choice == '2':
        run_np_seq_multi_anomaly()
    elif choice == '3':
        run_np_seq_uniform_dist()
    elif choice == '4':
        run_np_seq_kl_divergence()
    else:
        print("Invalid choice. Please run the script again and select a number from 1 to 4.")
