import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 18})

# Parameters for the LV system with logistic growth
r = 1.0         # intrinsic growth rate of the prey
K = 50          # carrying capacity
beta = 0.1      # predation rate coefficient
gamma = 1.5     # mortality rate of the predator
delta = 0.075   # efficiency of converting consumed prey into predator offspring

def lv_logistic(t, z):
    x, y = z
    dxdt = r * x * (1 - x/K) - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Compute the attractor (nontrivial fixed point)
x_fixed = gamma / delta
y_fixed = (r / beta) * (1 - (gamma / (delta * K)))
print("Attractor (fixed point):", (x_fixed, y_fixed))

# Create a grid for the phase plane: 0 <= x <= 70, 0 <= y <= 50
x_vals = np.linspace(0, 70, 25)
y_vals = np.linspace(0, 50, 25)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the vector field at each grid point
U = r * X * (1 - X/K) - beta * X * Y    # dx/dt
V = delta * X * Y - gamma * Y            # dy/dt
speed = np.sqrt(U**2 + V**2)
# Normalize vectors to plot directions consistently
U_norm = np.divide(U, speed, out=np.zeros_like(U), where=speed!=0)
V_norm = np.divide(V, speed, out=np.zeros_like(V), where=speed!=0)

plt.figure(figsize=(10, 8))
# Plot the vector field with arrows colored by the speed (magnitude)
q = plt.quiver(X, Y, U_norm, V_norm, speed, cmap='viridis', pivot='mid', scale=50)
#plt.colorbar(q, label='Speed (magnitude)')

# Prey nullcline:
# From dx/dt = 0, for x ≠ 0:  r*(1 - x/K) - beta*y = 0  =>  y = (r/beta)*(1 - x/K)
x_nullcline_vals = np.linspace(0, K, 100)
y_nullcline_prey = (r / beta) * (1 - x_nullcline_vals / K)
plt.plot(x_nullcline_vals, y_nullcline_prey, 'r--', label=r'Prey nullcline: $y=\frac{r}{\beta}(1-\frac{x}{K})$')

# Predator nullcline:
# From dy/dt = 0, for y ≠ 0:  delta*x - gamma = 0  =>  x = gamma/delta
plt.axvline(x=x_fixed, color='blue', linestyle='--', label=r'Predator nullcline: $x=\frac{\gamma}{\delta}$')

# Mark the attractor (nontrivial fixed point)
plt.plot(x_fixed, y_fixed, 'ko', markersize=8, label='Attractor (Fixed Point)')

# Simulate trajectories from several initial conditions
initial_conditions = [[10, 5], [30, 40], [40, 10], [20, 20]]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 500)
for ic in initial_conditions:
    sol = solve_ivp(lv_logistic, t_span, ic, t_eval=t_eval, rtol=1e-8)
    plt.plot(sol.y[0], sol.y[1], label=f'IC: {ic}')

plt.xlabel('Prey population, $x$')
plt.ylabel('Predator population, $y$')
plt.title('Phase Plane of the Lotka--Volterra System with Logistic Growth\nNullclines, Vector Field, and Attractor')
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
plt.grid(True)
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.tight_layout()
plt.savefig('lv_logistic.pdf')
plt.show()

