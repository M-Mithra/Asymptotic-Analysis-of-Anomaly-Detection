import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

# ------------------------------
# Helper Function: Generate samples from a discrete PMF
# ------------------------------
def sample_from_pmf(pmf, alphabet, n_samples):
    return np.random.choice(alphabet, size=n_samples, p=pmf)

# ------------------------------
# Helper Function: Compute empirical PMF (histogram normalized to 1)
# ------------------------------
def empirical_pmf(samples, alphabet):
    count = Counter(samples)
    return np.array([count[sym] / len(samples) for sym in alphabet])

# ------------------------------
# Helper Function: Compute Kullback-Leibler (KL) divergence
# D(p || q) = sum p(x) log(p(x)/q(x))
# ------------------------------
def kl_divergence(p, q):
    eps = 1e-12  # to avoid log(0)
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

# ------------------------------
# Main Function: GL Test
# Given three sequences, find the outlier using the GL test
# ------------------------------
def gl_test(sequences, alphabet):
    n = len(sequences[0])
    empirical_pmfs = [empirical_pmf(seq, alphabet) for seq in sequences]

    scores = []
    for i in range(3):
        # Candidate outlier: i
        # Others: j, k
        j, k = [x for x in range(3) if x != i]
        # GL score: sum of KL divergences of j and k to i
        score = kl_divergence(empirical_pmfs[j], empirical_pmfs[i]) + \
                kl_divergence(empirical_pmfs[k], empirical_pmfs[i])
        scores.append(score)

    # Lower score means more likely to be the common distribution
    # So the highest score corresponds to the outlier
    return np.argmax(scores)

# ------------------------------
# Example: Synthetic Experiment
# ------------------------------

# Define alphabet and PMFs
alphabet = [0, 1, 2, 3]
pi = [0.1, 0.2, 0.3, 0.4]  # Common distribution
mu = [0.3, 0.3, 0.2, 0.2]  # Outlier distribution

n_samples = 100  # Samples per sequence

# Generate sequences (sequence 0 is the outlier)
x1 = sample_from_pmf(mu, alphabet, n_samples)
x2 = sample_from_pmf(pi, alphabet, n_samples)
x3 = sample_from_pmf(pi, alphabet, n_samples)

# Run GL Test
outlier_index = gl_test([x1, x2, x3], alphabet)

# Print result
print("Detected outlier sequence index:", outlier_index)

# Optionally visualize
plt.figure(figsize=(8, 4))
for i, seq in enumerate([x1, x2, x3]):
    pmf = empirical_pmf(seq, alphabet)
    plt.bar([x + 0.25 * i for x in alphabet], pmf, width=0.25, label=f"Seq {i}")
plt.xticks(alphabet)
plt.xlabel("Symbol")
plt.ylabel("Probability")
plt.title("Empirical PMFs of Sequences")
plt.legend()
plt.grid(True)
plt.show()
