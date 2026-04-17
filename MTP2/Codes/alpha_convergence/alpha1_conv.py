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
    K_XX = rbf_kernel(X, X, sigma2)
    K_YY = rbf_kernel(Y, Y, sigma2)
    K_XY = rbf_kernel(X, Y, sigma2)
    
    # Remove diagonal elements for unbiased estimation
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term_XX = np.sum(K_XX) / (n * (n - 1))
    term_YY = np.sum(K_YY) / (n * (n - 1))
    
    # For independent samples, the cross term is often computed over all n^2 pairs
    term_XY = np.sum(K_XY) / (n**2) 
    
    return term_XX + term_YY - 2 * term_XY

# --- Simulation Parameters ---
np.random.seed(42)
max_n = 1000
step = 10
n_values = np.arange(10, max_n + 1, step)

# Generate typical streams (Null Hypothesis p=q)
X_stream = np.random.randn(max_n, 1)
Y_stream = np.random.randn(max_n, 1)

mmd2_values = []

# Compute MMD^2 sequentially
print("Running simulation...")
for n in n_values:
    X_n = X_stream[:n]
    Y_n = Y_stream[:n]
    mmd2 = compute_unbiased_mmd2(X_n, Y_n)
    mmd2_values.append(mmd2)

mmd2_values = np.array(mmd2_values)

# --- Compute the Magnifications ---
# Because MMD can fluctuate below 0 slightly due to being unbiased, 
# we take the absolute value to clearly see the magnitude of the noise envelope.
mag_05 = (n_values ** 0.5) * np.abs(mmd2_values)
mag_10 = (n_values ** 1.0) * np.abs(mmd2_values)
mag_15 = (n_values ** 1.5) * np.abs(mmd2_values)

# --- Plotting ---
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n_values, mag_05, color='blue')
plt.title('Alpha = 0.5 (Decays to 0)')
plt.ylabel('$n^{0.5} |M_u^2|$')

plt.subplot(3, 1, 2)
plt.plot(n_values, mag_10, color='green')
plt.title('Alpha = 1.0 (Stabilizes / Converges to Constant Distribution)')
plt.ylabel('$n^{1.0} |M_u^2|$')

plt.subplot(3, 1, 3)
plt.plot(n_values, mag_15, color='red')
plt.title('Alpha = 1.5 (Explodes to Infinity)')
plt.ylabel('$n^{1.5} |M_u^2|$')
plt.xlabel('Sequence Length (n)')

plt.tight_layout()
plt.show()