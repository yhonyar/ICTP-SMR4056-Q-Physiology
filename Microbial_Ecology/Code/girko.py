import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Set the dimension of the matrix
N = 2000

# Generate a random complex matrix from the Ginibre ensemble.
X = np.random.randn(N, N) 

# Normalize the matrix by 1/sqrt(N) so that the eigenvalue distribution converges to the uniform distribution on the unit disk.
X = X / np.sqrt(N)

# Compute the eigenvalues of the non-Hermitian matrix.
eigvals = np.linalg.eigvals(X)

# Create a scatter plot of the eigenvalues in the complex plane.
plt.figure(figsize=(8, 8))
plt.scatter(eigvals.real, eigvals.imag, s=5, alpha=0.5, color='blue', label='Eigenvalues')

# Plot the unit circle for reference.
theta = np.linspace(0, 2 * np.pi, 200)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, 'r-', lw=2, label='Unit Circle')

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title("Girko's Circular Law: Eigenvalue Distribution")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('girko.pdf')
plt.show()

