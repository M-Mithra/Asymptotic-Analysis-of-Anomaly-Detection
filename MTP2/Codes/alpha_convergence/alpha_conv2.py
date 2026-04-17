import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X, Y, sigma2=0.5):
    """Computes the RBF/Gaussian kernel matrix."""
    gamma = 1.0 / (2 * sigma2)
    X2 = np.sum(X**2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
    dist = X2 + Y2 - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * dist)

def compute_unbiased_mmd2(X, Y, sigma2=0.5):
    """Computes the unbiased MMD^2 estimator."""
    n = X.shape[0]
    if n < 2: return 0.0
    K_XX = rbf_kernel(X, X, sigma2)
    K_YY = rbf_kernel(Y, Y, sigma2)
    K_XY = rbf_kernel(X, Y, sigma2)
    
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term_XX = np.sum(K_XX) / (n * (n - 1))
    term_YY = np.sum(K_YY) / (n * (n - 1))
    term_XY = np.sum(K_XY) / (n**2) 
    
    return term_XX + term_YY - 2 * term_XY

# --- Simulation Parameters ---
np.random.seed(10)
max_n = 400
step = 5
n_values = np.arange(10, max_n + 1, step)

# Scenario 1: Null Hypothesis (No Anomaly, p = q)
X_null = np.random.randn(max_n, 1)
Y_null = np.random.randn(max_n, 1)

# Scenario 2: Alternative Hypothesis (Anomaly exists, p != q)
X_alt = np.random.randn(max_n, 1)
Y_alt = np.random.randn(max_n, 1) + 0.8  # Mean shift of 0.8

mmd2_null = []
mmd2_alt = []

# Compute MMD^2 sequentially
for n in n_values:
    mmd2_null.append(compute_unbiased_mmd2(X_null[:n], Y_null[:n]))
    mmd2_alt.append(compute_unbiased_mmd2(X_alt[:n], Y_alt[:n]))

mmd2_null = np.array(mmd2_null)
mmd2_alt = np.array(mmd2_alt)

# --- Define Thresholds ---
# We pick a constant C so that the alpha=1 threshold stays just above the noise
C_val = 1.0 
T_05 = C_val / (n_values ** 0.5)
T_10 = C_val / (n_values ** 1.0)
T_15 = C_val / (n_values ** 1.5)

# --- Plotting ---
plt.figure(figsize=(14, 6))

# Subplot 1: Null Hypothesis (Preventing False Alarms)
plt.subplot(1, 2, 1)
plt.plot(n_values, np.abs(mmd2_null), label='MMD Statistic (Noise)', color='gray', alpha=0.7, linewidth=2)
plt.plot(n_values, T_05, label='Threshold $\\alpha=0.5$', color='blue', linestyle='--')
plt.plot(n_values, T_10, label='Threshold $\\alpha=1.0$', color='green', linewidth=2)
plt.plot(n_values, T_15, label='Threshold $\\alpha=1.5$', color='red', linestyle='-.')

plt.title('Scenario 1: Identical Streams (Null Hypothesis)\nGoal: Threshold must stay ABOVE the noise', fontsize=12, fontweight='bold')
plt.xlabel('Sequence Length (n)')
plt.ylabel('$M_u^2$ Distance')
plt.ylim(0, 0.2)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Alternative Hypothesis (Detecting the Anomaly)
plt.subplot(1, 2, 2)
plt.plot(n_values, mmd2_alt, label='MMD Statistic (True Anomaly)', color='purple', linewidth=2)
plt.plot(n_values, T_05, label='Threshold $\\alpha=0.5$', color='blue', linestyle='--')
plt.plot(n_values, T_10, label='Threshold $\\alpha=1.0$', color='green', linewidth=2)
plt.plot(n_values, T_15, label='Threshold $\\alpha=1.5$', color='red', linestyle='-.')

plt.title('Scenario 2: Anomalous Stream (Alternative Hypothesis)\nGoal: Cross the threshold as FAST as possible', fontsize=12, fontweight='bold')
plt.xlabel('Sequence Length (n)')
plt.ylabel('$M_u^2$ Distance')
plt.ylim(0, 0.4)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_race_plot.png', dpi=300)
plt.close()