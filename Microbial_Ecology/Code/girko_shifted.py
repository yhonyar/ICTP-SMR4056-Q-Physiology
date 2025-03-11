import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Set the dimension of the matrix
N = 2000

# Generate a random complex matrix (Ginibre ensemble)
X = np.random.randn(N, N) 

# Normalize the matrix so that the eigenvalue distribution (for a standard Ginibre matrix)
# converges to the uniform distribution on the unit disk.
X = X / np.sqrt(N)

# Replace the diagonal entries with -1, shifting the eigenvalues.
np.fill_diagonal(X, -1)

# Compute the eigenvalues of the modified non-Hermitian matrix.
eigvals = np.linalg.eigvals(X)

# Plot the eigenvalues in the complex plane.
plt.figure(figsize=(8, 8))
plt.scatter(eigvals.real, eigvals.imag, s=5, alpha=0.5, color='blue', label='Eigenvalues')

# Plot the unit circle shifted by -1 (centered at (-1,0)) for reference.
theta = np.linspace(0, 2 * np.pi, 200)
circle_x = -1 + np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, 'r-', lw=2, label='Unit Circle (shifted)')

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title("Girko's Circular Law with Diagonal Shift: Eigenvalue Distribution")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('girko_shifted.pdf')
plt.show()

