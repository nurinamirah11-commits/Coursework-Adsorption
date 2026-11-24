# Part 2 : Dynamic measurements draft
# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Data and constants
# Column dimensions
L = 0.32        # m (column length)
epsilon = 0.32  # m (column void fraction)
rho_s = 990.0   # kg/m^3 (particle density of adsorbents A and B)
r_p = 0.001     # m (particle radius)

# Properties and constants
tau = 3.0        # - (tortuosity)
D_m = 7.481e-5   # m^2/s (molecular diffusivity)
R = 8.314        # m^3 Pa mol^-1 K^-1 (universal gas constant)

# Operating conditions
T = 303.0       # K (process temperature)
P = 100.0 * 1e3 # Pa (process pressure)
u_in = 4.864e-3 # m/s (inlet velocity, superficial(?))

# Interstitial inlet velocity, v
v_in = u_in / epsilon # m/s

# Part 1, we use pressure in bar
P_bar = P / 1e5

# Adsorption term constant in equation 1, that multiplies dq/dt
ads = (R * T / P) * rho_s * (1.0 - epsilon) / epsilon

# Constant in equation Glueckauf (LDF : Linear Driving Force) = 15De/rp^2
D_e = D_m / tau                 # m^2/s (effective pore diffusivity)
k_i = (15.0 * D_e) / (r_p ** 2) # 1/s

# Axial dispersion coefficient (equation 2)
D_L = 0.7 * D_m + v_in * r_p

# Number of grid cells along the column
N_cells = 50

# Parameters from Part 1 (if combined with part 1 codes, then just use the same name of variables to call later, no need to redefine)
# Langmuir-Freundlich for Romulus, ns in mol/kg, m no unity and K in bar^-1
ns_A_Ro, K_A_Ro, m_A_Ro = 4.7000, 3.4173e-2, 0.9998 # oxygen
ns_B_Ro, K_B_Ro, m_B_Ro = 2.7714, 3.8944e-1, 1.0941 # nitrogen

# Langmuir for Remus, ns in mol/kg and K in bar^-1 (we can put m=1 if we want to use the same equations for multi-component)
ns_A_Re, K_A_Re, m_A_Re = 6.0977, 1.6875e-2, 1.0000 # oxygen
ns_B_Re, K_B_Re, m_B_Re = 2.8344, 9.8015e-2, 1.0000 # nitrogen

# Each adsorbent is stored as a dictionary : easy to read and use
adsorbent_parameters = {
    "Romulus": {"ns_A": ns_A_Ro, "K_A": K_A_Ro, "m_A": m_A_Ro,
                "ns_B": ns_B_Ro, "K_B": K_B_Ro, "m_B": m_B_Ro,},
    "Remus": {"ns_A": ns_A_Re, "K_A": K_A_Re, "m_A": m_A_Re,
              "ns_B": ns_B_Re, "K_B": K_B_Re, "m_B": m_B_Re,},
}

# Equations of multi-component (now, we're considering a realistic mixture loading of O2 and N2)
# Calculate the mixture equilibrium loadings q_O2* and q_N2* for LDF later
# Extended Langmuir-Freundlich isotherm for both adsorbents even though Remus follows Langmuir (with m=1, Langmuir Freundlich reduces to Langmuir)
def multicomponent_Langmuir_Freundlich(P_A, P_B, a):
    # to avoid problems P approaching zero (NaN)
    P_A = np.maximum(P_A, 1e-12)
    P_B = np.maximum(P_B, 1e-12)

    # terms in denominator
    term_A = a["K_A"] * P_A ** (1.0 / a["m_A"])
    term_B = a["K_B"] * P_B ** (1.0 / a["m_B"])
    denominator = 1.0 + term_A + term_B

    q_A_eq = (a["ns_A"] * term_A) / denominator
    q_B_eq = (a["ns_B"] * term_B) / denominator
    return q_A_eq, q_B_eq

# From PDF, list of equations :
# (1) : dy/dt = D_L * d2y/dz2 - d(y*v)/dz - ads * dq/dt // gas balance for N2
# (5) : dq_i/dt = k_i * (q_i_eq - q_i) // LDF, solid balance

# function for breakthrough time (method of lines ; discretization)
def run_breakthrough_sim(adsorbent_name, Nz = N_cells, t_end = 1100.0, n_time = 250):
    a = adsorbent_parameters[adsorbent_name]
    delta_z = L / Nz # small length for discretization

    # Initial condition: column saturated with pure O2 (A) : y_B = 0, y_A = 1
    # Inlet feed: air rich with N2 (B), consider pure N2? : y_B_feed = 1, y_A_feed = 0 (at first we need to desorb the O2 in the column)
    y_B_0 = np.zeros(Nz)
    P_A_0 = np.ones(Nz) * P_bar
    P_B_0 = np.zeros(Nz) * P_bar
    q_A_0, q_B_0 = multicomponent_Langmuir_Freundlich(P_A_0, P_B_0, a)
    state0 = np.concatenate([y_B_0, q_A_0, q_B_0])

    def ode_rhs(t, state):
        y_B = np.clip(state[0:Nz], 0.0, 1.0)
        y_A = 1.0 - y_B
        q_A = state[Nz : 2 * Nz]
        q_B = state[2 * Nz : 3 * Nz]

        # Convert mole fractions to partial pressures in bar for the isotherm
        q_A_eq, q_B_eq = multicomponent_Langmuir_Freundlich(y_A * P_bar, y_B * P_bar, a)

        # LDF mass transfer rates
        dqA_dt = k_i * (q_A_eq - q_A)
        dqB_dt = k_i * (q_B_eq - q_B)

        # Interstitial velocity profile along cell faces
        v_faces = np.zeros(Nz + 1)
        v_faces[0] = v_in
        dv_dz = -ads * (dqA_dt + dqB_dt)
        v_faces[1:] = v_in + np.cumsum(dv_dz) * delta_z

        # Molar flux of N2 (component B) across cell faces
        flux_B = np.zeros(Nz + 1)
        flux_B[0] = v_in * 1.0
        flux_B[1:Nz] = v_faces[1:Nz] * y_B[:-1] - D_L * (y_B[1:] - y_B[:-1]) / delta_z
        flux_B[Nz] = v_faces[Nz] * y_B[-1]

        dyB_dt = -(flux_B[1:] - flux_B[:-1]) / delta_z - ads * dqB_dt
        return np.concatenate([dyB_dt, dqA_dt, dqB_dt])

    t_eval = np.linspace(0.0, t_end, n_time)
    sol = solve_ivp(
        ode_rhs, (0.0, t_end), state0, method="BDF", t_eval=t_eval, rtol=1e-4, atol=1e-6
    )

    y_B_out = np.clip(sol.y[Nz - 1, :], 0.0, 1.0)
    y_A_out = 1.0 - y_B_out

    # Ideal stoichiometric (equilibrium shock) time for pure N2 feed
    _, q_B_feed_eq = multicomponent_Langmuir_Freundlich(0.0, P_bar, a)
    t_stoich_eq = (L / v_in) * (1.0 + ads * q_B_feed_eq)

    # Breakthrough time when outlet N2 reaches 5% (0.05)
    hit_5pct = np.where(y_B_out >= 0.05)[0]
    t_bt_5pct = sol.t[hit_5pct[0]] if len(hit_5pct) > 0 else np.nan

    return sol.t, y_B_out, y_A_out, t_stoich_eq, t_bt_5pct


# Run breakthrough simulations for both adsorbents
t_Ro, yB_Ro, yA_Ro, teq_Ro, t5_Ro = run_breakthrough_sim("Romulus", t_end=1100.0)
print(f"Romulus -> 5% N2 breakthrough: {t5_Ro:.1f} s | Equilibrium shock time: {teq_Ro:.1f} s")

t_Re, yB_Re, yA_Re, teq_Re, t5_Re = run_breakthrough_sim("Remus", t_end=1100.0)
print(f"Remus   -> 5% N2 breakthrough: {t5_Re:.1f} s | Equilibrium shock time: {teq_Re:.1f} s")

# Plot breakthrough curves
plt.figure(figsize=(7, 4.5))
plt.plot(t_Ro, yB_Ro, "b-", lw=2, label=f"Romulus N2 (t5% = {t5_Ro:.1f} s)")
plt.axvline(teq_Ro, color="b", ls="--", lw=1.2, label=f"Romulus Eq ({teq_Ro:.1f} s)")

plt.plot(t_Re, yB_Re, "r-", lw=2, label=f"Remus N2 (t5% = {t5_Re:.1f} s)")
plt.axvline(teq_Re, color="r", ls="--", lw=1.2, label=f"Remus Eq ({teq_Re:.1f} s)")

plt.axhline(0.05, color="k", ls=":", lw=1, label="5% Breakthrough threshold")
plt.title("N2 Breakthrough Curves (Romulus vs. Remus)")
plt.xlabel("Time [s]")
plt.ylabel("Outlet N2 mole fraction [-]")
plt.xlim(0, 1100)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, ls=":", alpha=0.6)

plt.tight_layout()
plt.savefig("breakthrough_results.png", dpi=250, bbox_inches="tight")
plt.show()