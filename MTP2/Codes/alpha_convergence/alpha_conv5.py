import numpy as np
import matplotlib.pyplot as plt

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

# --- SETTINGS ---
sigma = np.sqrt(0.5)
np.random.seed(42)
steps = np.arange(50, 2000, 25) # Start at 50 to avoid early extreme variance
num_trials = 15                 # Average over 15 runs to smooth the noise

avg_noise_floor = np.zeros(len(steps))

print(f"Running {num_trials} trials to smooth the stochastic noise...")

for t in range(num_trials):
    # Both typical streams MUST have the exact same mean (0) 
    # for the noise floor to decay to 0.
    stream_1 = np.random.normal(0, 1, max(steps))
    stream_2 = np.random.normal(0, 1, max(steps))
    
    noise_this_trial = []
    for n in steps:
        # Calculate MMD distance (which should theoretically be 0)
        dist = abs(compute_mmd2_fast(stream_1[:n], stream_2[:n], sigma))
        noise_this_trial.append(dist)
        
    avg_noise_floor += np.array(noise_this_trial)

# Get the average across all trials
avg_noise_floor /= num_trials

# --- CALCULATE ALPHA USING LOG-LOG SLOPE ---
log_n = np.log10(steps)
log_noise = np.log10(avg_noise_floor)

# Fit a straight line to the log-log data
slope, intercept = np.polyfit(log_n, log_noise, 1)
derived_alpha = -slope

print(f"------------------------------------------------")
print(f"Empirically derived Alpha: {derived_alpha:.3f}")
print(f"------------------------------------------------")

# --- PLOTTING ---
plt.figure(figsize=(12, 5))

# Plot 1: Linear Scale (The Averaged Noise Floor)
plt.subplot(1, 2, 1)
plt.plot(steps, avg_noise_floor, color='gray', linewidth=2, label='Averaged Noise Floor')
plt.plot(steps, (10**intercept) / (steps**1.0), 'g--', linewidth=2, label=r'Perfect $1/n$ Curve')
plt.title(f'The Noise Floor\n(Averaged over {num_trials} runs)', fontsize=12, fontweight='bold')
plt.xlabel('Sequence Length (n)')
plt.ylabel('MMD Distance')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Log-Log Scale
plt.subplot(1, 2, 2)
plt.plot(log_n, log_noise, 'o', color='gray', alpha=0.5, label='Log Data Points')
plt.plot(log_n, slope * log_n + intercept, 'b-', linewidth=2, label=f'Fitted Line (Slope = {slope:.3f})')
plt.title(f'Deriving Alpha (Log-Log Scale)\nEmpirical $\\alpha$ = {derived_alpha:.3f}', fontsize=12, fontweight='bold')
plt.xlabel('Log10 of Sequence Length: $\log_{10}(n)$')
plt.ylabel('Log10 of Noise: $\log_{10}(|\Gamma_n|)$')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()