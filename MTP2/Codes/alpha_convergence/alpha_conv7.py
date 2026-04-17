import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import defaultdict
from tqdm import trange
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE STATISTICAL FUNCTIONS
# ==========================================
def compute_mmd2_fast(x, y, sigma):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    n, m = len(x), len(y)
    if n < 2 or m < 2: return 0.0
    
    gamma = 1.0 / (2 * sigma ** 2)
    Kxx = np.exp(-gamma * ((x - x.T) ** 2))
    Kyy = np.exp(-gamma * ((y - y.T) ** 2))
    Kxy = np.exp(-gamma * ((x - y.T) ** 2))
    
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    
    return (np.sum(Kxx)/(n*(n-1)) + np.sum(Kyy)/(m*(m-1)) - 2*np.sum(Kxy)/(n*m))

sigma = np.sqrt(0.5)
np.random.seed(42)

# ==========================================
# PART 1: DERIVING ALPHA FROM THE NOISE FLOOR
# ==========================================
print("--- PART 1: Empirically Deriving Alpha ---")
steps_alpha = np.arange(100, 1000, 50)
num_trials_alpha = 50
avg_noise_floor = np.zeros(len(steps_alpha))

for t in trange(num_trials_alpha, desc="Calculating Asymptote"):
    stream_typ1 = np.random.normal(0, 1, max(steps_alpha))
    stream_typ2 = np.random.normal(0, 1, max(steps_alpha))
    noise = [abs(compute_mmd2_fast(stream_typ1[:n], stream_typ2[:n], sigma)) for n in steps_alpha]
    avg_noise_floor += np.array(noise)

avg_noise_floor /= num_trials_alpha

log_n = np.log10(steps_alpha)
log_noise = np.log10(avg_noise_floor)
slope, intercept = np.polyfit(log_n, log_noise, 1)
empirical_alpha = -slope

print(f"\n>>> Empirically Derived Alpha (\u03B1): {empirical_alpha:.3f}")
print("------------------------------------------\n")

# ==========================================
# PART 2: THE SEQUENTIAL TEST TRADE-OFF SWEEP
# ==========================================
print("--- PART 2: Generating Trade-off Curves ---")
S = 5                 # Matching the paper's probable setup
anomaly_idx = 0
max_steps = 300       # Plenty of runway for E[N] up to 100
num_trials = 5000     # Required to measure -logPmax up to 3.5

alphas_to_test = [0.5, 1.0, 2.0, empirical_alpha]

# MATHEMATICALLY CALIBRATED RANGES 
# Tuned specifically for a true MMD distance of ~0.44 to stretch the curves
C_ranges = {
    0.5: [0.5, 1.0, 1.5, 2.0, 2.5, 3.2, 4.0],
    1.0: [2.0, 5.0, 10.0, 15.0, 22.0, 30.0, 42.0],
    2.0: [20.0, 100.0, 300.0, 800.0, 1500.0, 3000.0, 5000.0],    
    empirical_alpha: [2.0, 5.0, 10.0, 15.0, 22.0, 30.0, 42.0]
}

results = {alpha: [] for alpha in alphas_to_test}

for alpha in alphas_to_test:
    print(f"Simulating Sequential Test for \u03B1 = {alpha:.3f}")
    
    for C in C_ranges[alpha]:
        stop_times = []
        errors = 0
        
        for _ in trange(num_trials, desc=f"  C={C:>6.1f}", leave=False):
            streams = [np.random.normal(1.2, 1, max_steps) if i == anomaly_idx 
                       else np.random.normal(0, 1, max_steps) for i in range(S)]
            stopped = False
            
            # Step by 2 to speed up the massive 5000 trial loop
            for n in range(5, max_steps, 2): 
                mmd_vals = np.zeros(S)
                for i in range(S):
                    # Compare stream i against the union of all others
                    others = np.concatenate([streams[j][:n] for j in range(S) if j != i])
                    mmd_vals[i] = compute_mmd2_fast(streams[i][:n], others, sigma)
                
                candidate = np.argmax(mmd_vals)
                gamma_n = mmd_vals[candidate]
                
                if gamma_n > (C / (n ** alpha)):
                    stop_times.append(n)
                    if candidate != anomaly_idx: 
                        errors += 1 
                    stopped = True
                    break
                    
            if not stopped:
                stop_times.append(max_steps)
                errors += 1 
                
        p_max = max(errors / num_trials, 1 / num_trials) 
        mean_n = np.mean(stop_times)
        log_pmax = -np.log10(p_max)
        
        results[alpha].append((log_pmax, mean_n))

# ==========================================
# 3. VISUALIZATION (Matching the Paper)
# ==========================================
plt.figure(figsize=(9, 7))

markers = {0.5: 's', 1.0: 'o', 2.0: '^', empirical_alpha: '*'}
colors = {0.5: '#4c72b0', 1.0: '#c44e52', 2.0: '#8172b3', empirical_alpha: '#55a868'}
labels = {0.5: r'NP-SEQ(0.5, $\infty$)', 
          1.0: r'NP-SEQ-1 (Optimum)', 
          2.0: r'NP-SEQ(2, $\infty$)', 
          empirical_alpha: f'Empirical $\\alpha={empirical_alpha:.2f}$'}

for alpha in alphas_to_test:
    pts = sorted(results[alpha])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, marker=markers[alpha], color=colors[alpha], 
             label=labels[alpha], linewidth=2, markersize=8, alpha=0.9)

plt.title(r'Trade-off: $\mathbb{E}[N]$ vs $-\log_{10} P_{\max}$', fontsize=16)
plt.xlabel(r'$-\log_{10} P_{\max}$', fontsize=14)
plt.ylabel(r'$\mathbb{E}[N]$', fontsize=14)

plt.xlim(0, 4.0)
plt.ylim(0, 100)

# Styling to match the paper's clean LaTeX look
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=12, loc='upper left', framealpha=1.0, edgecolor='black')
plt.tight_layout()

plt.savefig('paper_replica_tradeoff.png', dpi=300)
print("\nSaved replica plot to 'paper_replica_tradeoff.png'")