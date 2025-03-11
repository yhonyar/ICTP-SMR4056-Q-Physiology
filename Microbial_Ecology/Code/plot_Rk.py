import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 18})

# Parameters
D = 0.2            # Dilution rate (1/time)
c_in = 5.0         # Inflow nutrient concentration
Y = 1.0            # Yield coefficient (1 substrate unit produces 1 biomass unit)

# Species A parameters (r-strategist): higher maximum growth rate but higher Monod constant
kmax_A = 1.0       # Maximum growth rate of Species A
K_A = 1.0          # Monod constant for Species A

# Species B parameters (K-strategist): lower maximum growth rate but lower Monod constant
kmax_B = 0.8       # Maximum growth rate of Species B
K_B = 0.5          # Monod constant for Species B

# Define the Monod growth functions
def mu_A(c):
    return kmax_A * c / (K_A + c)

def mu_B(c):
    return kmax_B * c / (K_B + c)

# ODE system: substrate consumption is scaled so that Y substrate units are consumed per biomass produced.
def chemostat_competition(t, y):
    c, rho_A, rho_B = y
    dc_dt = D * (c_in - c) - (mu_A(c) * rho_A + mu_B(c) * rho_B) / Y
    drho_A_dt = rho_A * (mu_A(c) - D)
    drho_B_dt = rho_B * (mu_B(c) - D)
    return [dc_dt, drho_A_dt, drho_B_dt]

# Initial conditions: start with the inflow nutrient concentration and small biomasses for both species
c0 = c_in
rho_A0 = 0.1
rho_B0 = 0.1
y0 = [c0, rho_A0, rho_B0]

# Extend the simulation time to clearly reach steady state
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE system
solution = solve_ivp(chemostat_competition, t_span, y0, t_eval=t_eval, method='RK45')

# Extract the results
t = solution.t
c = solution.y[0]
rho_A = solution.y[1]
rho_B = solution.y[2]
rho_total = rho_A + rho_B

# Create a two-panel figure: nutrient timeseries (top) and biomass timeseries (bottom)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot nutrient timeseries with dashed lines at K_A and K_B
ax1.plot(t, c, color='black', label="Nutrient concentration, $c$", lw=2)
ax1.axhline(K_A, color='blue', linestyle="--", label=r"$K_A$")
ax1.axhline(K_B, color='red', linestyle="--", label=r"$K_B$")
ax1.set_ylabel("Nutrient concentration, $c$", fontsize=14)
ax1.set_title("Nutrient Timeseries", fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plot biomass timeseries for both species and their total
ax2.plot(t, rho_A, label=r"Species A (r-strategist) Biomass ($\rho_A$)", lw=2)
ax2.plot(t, rho_B, label=r"Species B (K-strategist) Biomass ($\rho_B$)", lw=2)
ax2.plot(t, rho_total, label=r"Total Biomass ($\rho_A+\rho_B$)", lw=2, linestyle="--")
ax2.set_xlabel("Time", fontsize=14)
ax2.set_ylabel("Biomass", fontsize=14)
ax2.set_title("Biomass Dynamics: Approach to Steady State", fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True)

# Annotate the steady state values in the biomass plot
steady_time = t[-1]
steady_rho_A = rho_A[-1]
steady_rho_B = rho_B[-1]
ax2.scatter([steady_time], [steady_rho_A], color='C0', s=50)
ax2.scatter([steady_time], [steady_rho_B], color='C1', s=50)
ax2.text(steady_time * 0.95, steady_rho_A * 1.05, r'$\rho_A^*$', color='C0', fontsize=12)
ax2.text(steady_time * 0.95, steady_rho_B * 1.05, r'$\rho_B^*$', color='C1', fontsize=12)

plt.tight_layout()
plt.savefig('competition_Rk.pdf')
plt.show()

