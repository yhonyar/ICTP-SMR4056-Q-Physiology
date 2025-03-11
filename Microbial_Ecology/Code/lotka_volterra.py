import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 18})

# Parameters for the Lotka–Volterra system
alpha = 1.0    # intrinsic growth rate of the prey
beta  = 0.1    # predation rate coefficient
gamma = 1.5    # mortality rate of the predator
delta = 0.075  # efficiency of converting consumed prey into predator offspring

def lv_system(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Create a grid for the phase plane over 0 <= x <= 70 and 0 <= y <= 50
x_vals = np.linspace(0, 70, 25)
y_vals = np.linspace(0, 50, 25)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the vector field at each grid point
U = alpha * X - beta * X * Y  # dx/dt
V = delta * X * Y - gamma * Y  # dy/dt
speed = np.sqrt(U**2 + V**2)
# Normalize vectors for consistent arrow lengths in quiver
U_norm = np.divide(U, speed, out=np.zeros_like(U), where=speed != 0)
V_norm = np.divide(V, speed, out=np.zeros_like(V), where=speed != 0)

plt.figure(figsize=(10, 8))
# Plot the vector field with arrows colored by speed (magnitude)
q = plt.quiver(X, Y, U_norm, V_norm, speed, cmap='viridis', pivot='mid', scale=50)
plt.colorbar(q, label='Speed (magnitude)')

# Compute and plot the nontrivial nullclines:
# For dx/dt = 0: y = alpha / beta (prey nullcline)
y_nullcline = alpha / beta
# For dy/dt = 0: x = gamma / delta (predator nullcline)
x_nullcline = gamma / delta
plt.axhline(y=y_nullcline, color='red', linestyle='--', 
            label=r'$x$-nullcline: $y=\alpha/\beta$')
plt.axvline(x=x_nullcline, color='blue', linestyle='--', 
            label=r'$y$-nullcline: $x=\gamma/\delta$')

# Mark the nontrivial fixed point (intersection of the nullclines)
plt.plot(x_nullcline, y_nullcline, 'ko', markersize=8, 
         label='Nontrivial Fixed Point')

# Simulate trajectories from several initial conditions (IC stands for Initial Condition)
initial_conditions = [[10, 5], [30, 40], [40, 10], [20, 20]]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 500)
for ic in initial_conditions:
    sol = solve_ivp(lv_system, t_span, ic, t_eval=t_eval, rtol=1e-8)
    plt.plot(sol.y[0], sol.y[1], label=f'IC: {ic}')

# Emphasize the dynamics near the axes:
# Along the y=0 axis (predators extinct), dx/dt = alpha*x, so prey grows exponentially.
# Along the x=0 axis (prey extinct), dy/dt = -gamma*y, so predators decay exponentially.
plt.xlabel('Prey population, $x$')
plt.ylabel('Predator population, $y$')
plt.title('Phase Plane of the Lotka–Volterra System\nwith Nullclines, Vector Field, and Trajectories')
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
plt.grid(True)
plt.xlim(0, 70)
plt.ylim(0, 50)  # Set y-axis limit to [0, 50]

plt.tight_layout()
plt.savefig('lotka_volterra.pdf')
plt.show()

