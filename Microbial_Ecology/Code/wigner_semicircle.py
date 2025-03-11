import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Set the dimension of the matrix
N = 1000
#stdev=np.sqrt(2)
stdev=1

# Generate a random upper triangular matrix (excluding the diagonal)
upper_tri = np.triu(np.random.normal(loc=0, scale=stdev, size=(N, N)), k=1)

# Create a symmetric matrix: for i < j, set A_ij = upper_tri[i,j] and A_ji = A_ij.
A = upper_tri + upper_tri.T

# Set the diagonal entries independently from N(0,stdev^2)
diag = np.random.normal(loc=0, scale=stdev, size=N)
np.fill_diagonal(A, diag)

# Normalize the matrix so that the eigenvalue density converges to the semicircle law
M = A / np.sqrt(N)

# Compute the eigenvalues using a symmetric eigensolver (more efficient for symmetric matrices)
eigenvalues = np.linalg.eigvalsh(M)

# Plot histogram of eigenvalues
plt.figure(figsize=(8, 6))
counts, bins, _ = plt.hist(eigenvalues, bins=50, density=True, alpha=0.6, color='blue', label='Simulated Density')

# Define the semicircular density function
x = np.linspace(-stdev*2, stdev*2, 400)
semicircle_density = (1 / (2 * np.pi * (stdev**2))) * np.sqrt(np.maximum(0, 4*(stdev**2) - x**2))

# Overlay the theoretical semicircle law
plt.plot(x, semicircle_density, 'r-', lw=2, label="Wigner's Semicircle Law")
plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.title("Wigner's Semicircle Theorem: Eigenvalue Distribution")
plt.legend()
plt.savefig('wigner.pdf')
plt.show()

