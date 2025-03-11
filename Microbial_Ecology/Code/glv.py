import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 18})

# Number of species
n = 10

# Random intrinsic growth rates (ensuring they are positive)
r = np.random.uniform(0.5, 1.5, size=n)

# Interaction matrix: off-diagonals from a normal distribution, self-regulation set to -1
A = np.random.randn(n, n) * 0.1
np.fill_diagonal(A, -1.0)

def glv(t, N):
    # GLV dynamics: dN_i/dt = N_i (r_i + sum_j a_ij * N_j)
    return N * (r + A.dot(N))

# Simulation time span and initial conditions
t_span = (0, 50)
N0 = np.random.uniform(0.1, 2.0, size=n)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the system of ODEs
sol = solve_ivp(glv, t_span, N0, t_eval=t_eval, rtol=1e-8)

# Plot time series of species abundances
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(sol.t, sol.y[i], label=f"Species {i+1}")
plt.xlabel("Time")
plt.ylabel("Population Density")
plt.title("Generalized Lotka–Volterra Dynamics for 10 Species")
plt.legend(loc='upper right')
#plt.grid(True)
plt.tight_layout()
plt.savefig('glv.pdf')
plt.show()

