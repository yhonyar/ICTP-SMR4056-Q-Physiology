import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress indicator
from matplotlib import cm
plt.rcParams.update({'font.size': 18})

# Simulation parameters
sigma = 0.2                        # Standard deviation of interaction strengths
n_values = np.arange(4, 120, 4)      # System sizes (n) to test for m=1 (single block)
num_trials = 500                   # Number of trials per n value
connectance_values = [1.0, 0.5]      # Different connectance values to try

# Dictionary to store the probability of instability for each connectance value.
# We'll store results as: results[C] = (n_values, unstable_prob_list)
results = {}

# Loop over each connectance value
for C in connectance_values:
    unstable_prob = []  # List to store instability probability for each n
    # Loop over n-values with a progress indicator.
    for n in tqdm(n_values, desc=f"Processing n-values for C = {C}"):
        unstable_count = 0
        for _ in range(num_trials):
            # Build the interaction matrix A and the community matrix M for m=1.
            # Single block: full n x n matrix.
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and np.random.rand() < C:
                        A[i, j] = np.random.randn() * sigma
            M = -np.eye(n) + A

            # Compute eigenvalues and check for instability.
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals.real > 0):
                unstable_count += 1
        unstable_prob.append(unstable_count / num_trials)
    results[C] = (n_values, unstable_prob)

# Set up the plot.
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Define a color map so that each connectance value gets a unique color.
colors = cm.tab10(np.linspace(0, 1, len(connectance_values)))
color_idx = 0

for C, (n_vals, probs) in results.items():
    color = colors[color_idx]
    label = f"C = {C}"
    ax.plot(n_vals, probs, marker='o', linestyle='-', color=color, label=label)
    # Calculate the May stability threshold for m=1:
    # sigma * sqrt(n * C) = 1  =>  n = 1/(sigma^2 * C)
    n_thresh = 1 / (sigma**2 * C)
    # Plot vertical line if the threshold is within the range of n-values for this curve.
    if n_thresh >= n_vals[0] and n_thresh <= n_vals[-1]:
        ax.axvline(x=n_thresh, color=color, linestyle='--', 
                   label=f"May crit.: C = {C} (n={n_thresh:.1f})")
    color_idx += 1

ax.set_xlabel('n (Number of Species)')
ax.set_ylabel('Probability of at least one positive eigenvalue')
ax.set_title("May's Stability Criterion Simulation (m = 1)")
ax.legend()
ax.grid(True)
plt.savefig('may.pdf')
plt.show()

