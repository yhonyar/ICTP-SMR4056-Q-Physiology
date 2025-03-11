import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18}) 

# Parameter definitions
gamma = 0.9        # dimensionless growth parameter
c_in = 1.0         # inflowing nutrient concentration

# Fixed points
c_star = 1.0 / (gamma - 1)           # from gamma*c/(1+c)=1, so c = 1/(gamma-1)
rho_star = c_in - c_star             # coexistence fixed point
c_wash = c_in                       # washout state for nutrient
rho_wash = 0                        # washout state for bacterial density

# Define the ODEs for the chemostat:
#   d(c)/dt = c_in - c - gamma*(c/(1+c))*rho
#   d(rho)/dt = rho*(gamma*(c/(1+c)) - 1)
def dcdt(c, rho):
    return c_in - c - gamma * (c/(1+c)) * rho

def drhodt(c, rho):
    return rho * (gamma * (c/(1+c)) - 1)

# Prepare grid for the phase field (avoid c=0 to prevent division by zero)
c_vals = np.linspace(0.01, 1.5, 20)
rho_vals = np.linspace(-0.1, 1.2, 20)
C, Rho = np.meshgrid(c_vals, rho_vals)
dC = dcdt(C, Rho)
dRho = drhodt(C, Rho)
# Normalize the vectors for clear quiver plotting
magnitude = np.sqrt(dC**2 + dRho**2)
dC_norm = dC / magnitude
dRho_norm = dRho / magnitude

# ------------------------------
# Figure 1: Nullclines and Fixed Points
# ------------------------------
fig1, ax1 = plt.subplots(figsize=(8,6))

# Nutrient nullcline:  d(c)/dt = 0  ->  rho = ((c_in - c)*(1+c))/(gamma*c)
c_line = np.linspace(0.001, c_in - 0.001, 200)
rho_nullcline = ((c_in - c_line) * (1 + c_line)) / (gamma * c_line)
ax1.plot(c_line, rho_nullcline, 'b-', label='Nutrient nullcline (d𝑐/dt = 0)')

# Bacterial nullcline:  d(rho)/dt = 0  ->  c = 1/(gamma-1) (for rho ≠ 0)
ax1.axvline(x=c_star, color='r', linestyle='-', label='Bacterial nullcline (dρ/dt = 0)')
# Also plot the horizontal line for the trivial solution rho=0
ax1.axhline(y=0, color='r', linestyle='--', label='Washout (ρ = 0)')

# Plot fixed points
ax1.plot(c_star, rho_star, 'ko', markersize=8, label='Coexistence fixed point')
ax1.plot(c_wash, rho_wash, 'mo', markersize=8, label='Washout fixed point')

ax1.set_xlabel('Nutrient concentration, c')
ax1.set_ylabel('Bacterial density, rho')
ax1.set_title('Phase Diagram with Nullclines and Fixed Points')
ax1.set_xlim(0, 1.5)
ax1.set_ylim(-0.1, 1.2)
ax1.legend()
ax1.grid()

# ------------------------------
# Figure 2: Phase Field with Trajectories
# ------------------------------
fig2, ax2 = plt.subplots(figsize=(8,6))

# Plot the vector field
ax2.quiver(C, Rho, dC_norm, dRho_norm, color='gray')

# Overlay the nullclines and fixed points (same as in Figure 1)
ax2.plot(c_line, rho_nullcline, 'b-', label='Nutrient nullcline')
ax2.axvline(x=c_star, color='r', linestyle='-', label='Bacterial nullcline')
ax2.axhline(y=0, color='r', linestyle='--', label='Washout (ρ = 0)')
ax2.plot(c_star, rho_star, 'ko', markersize=8, label='Coexistence fixed point')
ax2.plot(c_wash, rho_wash, 'mo', markersize=8, label='Washout fixed point')

# Define a simple Euler integrator for trajectories
def euler_trajectory(c0, rho0, dt=0.01, steps=500):
    cs = [c0]
    rhos = [rho0]
    c_current, rho_current = c0, rho0
    for _ in range(steps):
        dc = dcdt(c_current, rho_current)
        drho = drhodt(c_current, rho_current)
        c_current += dt * dc
        rho_current += dt * drho
        cs.append(c_current)
        rhos.append(rho_current)
    return np.array(cs), np.array(rhos)

# Plot trajectories from several initial conditions
initial_conditions = [
    (0.2, 0.2),
    (0.8, 0.8),
    (1.2, 0.2),
    (0.2, 1.0),
    (1.2, 1.0)
]
for c0, rho0 in initial_conditions:
    cs_traj, rhos_traj = euler_trajectory(c0, rho0)
    ax2.plot(cs_traj, rhos_traj, '--', lw=2)

ax2.set_xlabel('Nutrient concentration, c')
ax2.set_ylabel('Bacterial density, rho')
ax2.set_title('Phase Field with Trajectories')
ax2.set_xlim(0, 1.5)
ax2.set_ylim(-0.1, 1.2)
ax2.legend()
ax2.grid()
plt.savefig('chemostat_washout.pdf')
plt.show()

