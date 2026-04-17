import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import trange # Added for a nice progress bar

# ==========================================
# 1. CORE FUNCTIONS
# ==========================================
def compute_mmd2_fast(x, y, sigma):
    """Highly vectorized calculation of the unbiased MMD^2 estimator."""
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

# ==========================================
# 2. SIMULATION SETTINGS
# ==========================================
sigma = np.sqrt(0.5)
np.random.seed(42)

# FIX 1: The "Burn-In" Period
# We start measuring at n=200. MMD only decays at exactly O(1/n) asymptotically.
# Measuring below 200 introduces small-sample noise that skews the slope.
steps = np.arange(200, 2000, 50) 

# FIX 2: The Law of Large Numbers
# We run 100 independent trials and average them to perfectly smooth out 
# the random stochastic variance of the Gaussian generation.
num_trials = 100 

avg_noise_floor = np.zeros(len(steps))

# ==========================================
# 3. RUNNING THE MONTE CARLO SIMULATION
# ==========================================
print(f"Running {num_trials} Monte Carlo trials to isolate the asymptotic noise floor...")

for t in trange(num_trials, desc="Simulating"):
    # Generate two mathematically identical typical streams (p=q)
    stream_1 = np.random.normal(0, 1, max(steps))
    stream_2 = np.random.normal(0, 1, max(steps))
    
    noise_this_trial = []
    for n in steps:
        dist = abs(compute_mmd2_fast(stream_1[:n], stream_2[:n], sigma))
        noise_this_trial.append(dist)
        
    avg_noise_floor += np.array(noise_this_trial)

# Get the true Expected Value (E[M_u^2])
avg_noise_floor /= num_trials

# ==========================================
# 4. MATHEMATICAL PROOF (LOG-LOG REGRESSION)
# ==========================================
log_n = np.log10(steps)
log_noise = np.log10(avg_noise_floor)

# We use scipy's linregress to get the exact slope and R-squared value
regression = linregress(log_n, log_noise)
empirical_alpha = -regression.slope
r_squared = regression.rvalue ** 2

print("\n" + "="*50)
print(f" MATHEMATICAL RESULTS FOR PROFESSOR")
print("="*50)
print(f" Empirical Alpha (\u03B1)  : {empirical_alpha:.4f}  <-- (Target is 1.000)")
print(f" R-Squared Value (R\u00B2) : {r_squared:.4f}  <-- (1.0 means perfect power-law fit)")
print("="*50 + "\n")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(14, 6))

# Plot 1: Linear Scale (The Asymptotic Curve)
plt.subplot(1, 2, 1)
plt.plot(steps, avg_noise_floor, color='black', linewidth=3, label='Empirical Noise Floor $\mathbb{E}[|M_u^2|]$')

# We calculate the Y-intercept to perfectly overlay the mathematical C/n curve
C_constant = 10 ** regression.intercept
plt.plot(steps, C_constant / (steps**1.0), 'g--', linewidth=2.5, label=r'Theoretical Asymptote ($C/n^1$)')

plt.title(f'1. The True Noise Floor\n(Averaged over {num_trials} Monte Carlo runs)', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length ($n$)', fontsize=12)
plt.ylabel('MMD Distance', fontsize=12)
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=11)

# Plot 2: Log-Log Scale (The Power Law Proof)
plt.subplot(1, 2, 2)
plt.plot(log_n, log_noise, 'o', color='gray', alpha=0.6, markersize=8, label='Log-Transformed Data')
plt.plot(log_n, regression.intercept + regression.slope * log_n, 'b-', linewidth=3, 
         label=f'Linear Regression\nSlope ($-\\alpha$) = {-empirical_alpha:.3f}')

plt.title(f'2. Power Law Derivation (Log-Log Scale)\nEmpirical $\\alpha \\approx {empirical_alpha:.3f}$ ($R^2={r_squared:.3f}$)', 
          fontsize=14, fontweight='bold')
plt.xlabel('$\log_{10}(n)$', fontsize=12)
plt.ylabel('$\log_{10}(|Noise|)$', fontsize=12)
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('final_alpha_proof.png', dpi=300, bbox_inches='tight')
plt.show()