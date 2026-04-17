import numpy as np
import matplotlib.pyplot as plt

# --- SETTINGS ---
S = 3  # Total streams
max_steps = 1500
sigma = np.sqrt(0.5)
np.random.seed(42)

def compute_mmd2_fast(x, y, sigma):
    """Computes the unbiased MMD^2 estimator."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    n, m = len(x), len(y)
    if n < 2 or m < 2: return 0.0
    
    Kxx = np.exp(-((x - x.T) ** 2) / (2 * sigma ** 2))
    Kyy = np.exp(-((y - y.T) ** 2) / (2 * sigma ** 2))
    Kxy = np.exp(-((x - y.T) ** 2) / (2 * sigma ** 2))
    
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    
    return (np.sum(Kxx)/(n*(n-1)) + np.sum(Kyy)/(m*(m-1)) - 2*np.sum(Kxy)/(n*m))

# --- GENERATE DATA (3 Streams) ---
# Stream 0: Anomaly (Mean=1.2)
# Stream 1 & 2: Typical (Mean=0)
streams = [
    np.random.normal(1.2, 1, max_steps),  # Anomaly
    np.random.normal(0, 1, max_steps),    # Typical
    np.random.normal(0, 1, max_steps)     # Typical
]

# --- CALCULATE STATISTIC OVER TIME ---
steps = np.arange(10, max_steps, 5) 
gamma_history = {0: [], 1: [], 2: []}

for n in steps:
    for i in range(S):
        # Calculate MMD against all other individual streams, then take the minimum
        mmd_against_others = []
        for j in range(S):
            if i != j:
                mmd_val = compute_mmd2_fast(streams[i][:n], streams[j][:n], sigma)
                mmd_against_others.append(mmd_val)
        gamma_history[i].append(min(mmd_against_others))

# --- EMPIRICALLY DERIVE ALPHA (LOG-LOG FIT) ---
# We isolate the "Noise Floor" (the magnitude of the typical streams)
noise_floor = np.maximum(np.abs(gamma_history[1]), np.abs(gamma_history[2]))

log_n = np.log10(steps)
log_noise = np.log10(noise_floor)

# Fit a line to find the slope (-alpha)
slope, intercept = np.polyfit(log_n, log_noise, 1)
print(f"Empirically derived Alpha: {-slope:.3f}")

# --- PLOTTING ---
plt.figure(figsize=(14, 6))

# Plot 1: Linear Scale
plt.subplot(1, 2, 1)
plt.plot(steps, gamma_history[0], color='purple', label='Stream 0 (Anomaly)', linewidth=2)
plt.plot(steps, gamma_history[1], color='gray', label='Stream 1 (Typical)', alpha=0.7)
plt.plot(steps, gamma_history[2], color='darkgray', label='Stream 2 (Typical)', alpha=0.7)

# Overlay theoretical threshold
plt.plot(steps, 1.0 / (steps**1.0), 'g--', label=r'Threshold ($1/n^1$)', linewidth=2)
plt.title('Plot 1: The Test Statistic over Time\n(Linear Scale)', fontsize=12, fontweight='bold')
plt.xlabel('Sequence Length (n)')
plt.ylabel(r'Statistic $\Gamma_n$')
plt.ylim(-0.05, 0.4)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Log-Log Scale
plt.subplot(1, 2, 2)
plt.plot(log_n, log_noise, 'o', color='gray', alpha=0.4, label='Noise Floor (Raw Data)')
plt.plot(log_n, slope * log_n + intercept, 'b-', linewidth=2, label=f'Line of Best Fit\nSlope = {slope:.3f}')
plt.title('Plot 2: "Discovering" Alpha\n(Log-Log Scale)', fontsize=12, fontweight='bold')
plt.xlabel('Log10 of Sequence Length: $\log_{10}(n)$')
plt.ylabel('Log10 of Noise Magnitude: $\log_{10}(|\Gamma_n|)$')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()