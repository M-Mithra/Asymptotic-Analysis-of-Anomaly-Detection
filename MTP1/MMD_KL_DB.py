import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress log(0) warnings that can occur if p_error becomes 0
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===== HELPER FUNCTIONS =====

def kl_divergence(p, q):
    """ Calculates KL Divergence D_KL(p || q) for discrete distributions. """
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def rbf_kernel(x, y, gamma):
    """ Computes the RBF kernel between two points. """
    return np.exp(-gamma * (x - y)**2)

def bhattacharyya_distance(p, q):
    """ Calculates Bhattacharyya distance for discrete distributions. """
    bc_coefficient = np.sum(np.sqrt(p * q))
    # Add a small epsilon to prevent log(0) if distributions are perfectly disjoint
    return -np.log(bc_coefficient + 1e-10)

def chernoff_information(p, q):
    """ Calculates Chernoff Information C(P,Q), the tightest error exponent bound. """
    # We find the minimum of the Chernoff divergence by searching over lambda from 0 to 1
    lambdas = np.linspace(0, 1, 100)
    divergences = [-np.log(np.sum(p**l * q**(1-l))) for l in lambdas]
    return np.min(divergences)

# ===== 1. SIMULATION SETTINGS =====
# Define the "normal" and "outlier" distributions
P_NORMAL = 0.4  # Probability of '1' for normal sequences
P_OUTLIER = 0.7 # Probability of '1' for the outlier sequence

# MMD decision rule parameter
GAMMA = 1.0

# Simulation parameters
NUM_TRIALS = 10000
SEQUENCE_LENGTHS = np.arange(10, 301, 10)

# ===== 2. THE SIMULATION =====
error_prob_kl = []
error_prob_mmd = []
print(f"Running {NUM_TRIALS} trials for each sequence length...")

for L in SEQUENCE_LENGTHS:
    error_count_kl = 0
    error_count_mmd = 0
    for _ in range(NUM_TRIALS):
        # Assume H1 is true: Sequence 1 is the outlier
        seq1 = np.random.binomial(1, P_OUTLIER, L)
        seq2 = np.random.binomial(1, P_NORMAL, L)
        seq3 = np.random.binomial(1, P_NORMAL, L)
        
        # Represent each sequence by its sample mean (proportion of 1s)
        t1, t2, t3 = np.mean(seq1), np.mean(seq2), np.mean(seq3)
        
        # --- KL Divergence Test ---
        P1, P2, P3 = np.array([t1, 1-t1]), np.array([t2, 1-t2]), np.array([t3, 1-t3])
        Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
        D_kl = np.array([kl_divergence(P1, Q1), kl_divergence(P2, Q2), kl_divergence(P3, Q3)])
        if np.argmax(D_kl) != 0:
            error_count_kl += 1
            
        # --- MMD Test ---
        k23 = rbf_kernel(t2, t3, gamma=GAMMA)
        k13 = rbf_kernel(t1, t3, gamma=GAMMA)
        k12 = rbf_kernel(t1, t2, gamma=GAMMA)
        D_mmd = np.array([k23, k13, k12])
        if np.argmax(D_mmd) != 0:
            error_count_mmd += 1
            
    error_prob_kl.append(error_count_kl / NUM_TRIALS)
    error_prob_mmd.append(error_count_mmd / NUM_TRIALS)

print("Simulation complete.")

# ===== 3. CALCULATE THEORETICAL BOUNDS AND ESTIMATE EXPONENTS =====
# Define the true distributions for theoretical calculations
p_norm_dist = np.array([P_NORMAL, 1 - P_NORMAL])
p_outlier_dist = np.array([P_OUTLIER, 1 - P_OUTLIER])

# Calculate theoretical bounds
b_dist = bhattacharyya_distance(p_norm_dist, p_outlier_dist)
chernoff_bound = chernoff_information(p_norm_dist, p_outlier_dist)

# --- Estimate exponent for KL ---
valid_indices_kl = [i for i, p in enumerate(error_prob_kl) if p > 0]
if valid_indices_kl:
    lengths_kl = SEQUENCE_LENGTHS[valid_indices_kl]
    log_p_kl = np.log(np.array(error_prob_kl)[valid_indices_kl])
    slope_kl, _ = np.polyfit(lengths_kl, log_p_kl, 1)
    alpha_kl = -slope_kl
else:
    alpha_kl = float('inf') # No errors detected

# --- Estimate exponent for MMD ---
valid_indices_mmd = [i for i, p in enumerate(error_prob_mmd) if p > 0]
if valid_indices_mmd:
    lengths_mmd = SEQUENCE_LENGTHS[valid_indices_mmd]
    log_p_mmd = np.log(np.array(error_prob_mmd)[valid_indices_mmd])
    slope_mmd, _ = np.polyfit(lengths_mmd, log_p_mmd, 1)
    alpha_mmd = -slope_mmd
else:
    alpha_mmd = float('inf') # No errors detected

# ===== 4. PRINT AND VISUALIZE THE FINAL COMPARISON =====

print("\n" + "="*50)
print("       RESULTS COMPARISON")
print("="*50)
print(f"Estimated Error Exponent (alpha) for KL:   {alpha_kl:.4f}")
print(f"Estimated Error Exponent (alpha) for MMD:  {alpha_mmd:.4f}")
print("-" * 50)
print("Theoretical Bounds:")
print(f"Chernoff Information (Tightest Bound):     {chernoff_bound:.4f}")
print(f"2 * Bhattacharyya Distance (Upper Bound):  {2 * b_dist:.4f}")
print("="*50)

# --- Create the Comparison Plot ---
plt.figure(figsize=(12, 8))
# Plot simulation results
plt.plot(SEQUENCE_LENGTHS, np.log(error_prob_kl), 'o', color='blue', label='KL Divergence (Simulation Results)')
plt.plot(SEQUENCE_LENGTHS, np.log(error_prob_mmd), 'x', color='green', label='MMD (Simulation Results)')

# Plot the theoretical bound
plt.plot(SEQUENCE_LENGTHS, -chernoff_bound * SEQUENCE_LENGTHS + log_p_kl[0], 'k--', 
         label=f'Theoretical Bound (Slope = {-chernoff_bound:.4f})')

plt.title('Error Exponent Comparison: KL vs. MMD vs. Theoretical Bound', fontsize=16)
plt.xlabel('Sequence Length (L)', fontsize=12)
plt.ylabel('Log of Error Probability (ln Pe)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(bottom=np.log(1/NUM_TRIALS)-1) # Set y-limit for better view
plt.show()