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

# Part 1, we use pressure in bar
P_bar = P / 1e5

# Adsorption term constant in equation 1, that multiplies dq/dt
ads = (R * T / P) * rho_s * (1.0 - epsilon) / epsilon

# Constant in equation Glueckauf (LDF : Linear Driving Force) = 15De/rp^2
D_e = D_m / tau                 # m^2/s (effective pore diffusivity)
k_i = (15.0 * D_e) / (r_p ** 2) # 1/s

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
# Reminder : as the values for K are in bar-1, the pressure here is in bar too
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

# function for breakthrough time (method of lines ; discretization)
def run_breakthrough_sim(adsorbent_name, Nz = N_cells, t_end = 1100.0, n_time = 250, L_sim=L,u_sim=u_in):
    a = adsorbent_parameters[adsorbent_name]
    delta_z = L_sim / Nz # small length for discretization

    # Interstitial inlet velocity, v
    v_in = u_sim / epsilon # m/s 

    # Axial dispersion coefficient (equation 2)
    D_L = 0.7 * D_m + v_in * r_p

    # Initial condition: column saturated with pure O2 (A) : y_B = 0, y_A = 1
    y_B_0 = np.zeros(Nz)

    # P_i = y_i * Ptotal
    P_A_0 = np.ones(Nz) * P_bar   
    P_B_0 = np.zeros(Nz) * P_bar

    # Calculate initial values for qA and qB using isotherm
    q_A_0, q_B_0 = multicomponent_Langmuir_Freundlich(P_A_0, P_B_0, a)
    state0 = np.concatenate([y_B_0, q_A_0, q_B_0])

    def ode_rhs(t, state):
        y_B = np.clip(state[0:Nz], 0.0, 1.0)
        y_A = 1.0 - y_B
        q_A = state[Nz : 2 * Nz]
        q_B = state[2 * Nz : 3 * Nz]

        # Calculate adsorption capacity equilibrium using multicomponent isotherm
        # Don't forget to use partial pressure in the bracket as we define in function isotherm
        # P_i = y_i * Ptotal
        q_A_eq, q_B_eq = multicomponent_Langmuir_Freundlich(y_A * P_bar, y_B * P_bar, a)

        # LDF mass transfer rates (equation 5)
        dqA_dt = k_i * (q_A_eq - q_A)
        dqB_dt = k_i * (q_B_eq - q_B)

        # Interstitial velocity profile along cell faces
        # Velocity is a variable that changes as gas moves from left to right because molecules are adsorbed/desorbed in the 'small box/tranche'.
        # meaning that velocity into the small volume is different with velocity going out the small volume
        # so we need Nz+1 values of v (Nz+1 faces) ; more details in Amirah's goodnotes
        v_faces = np.zeros(Nz + 1)                      
        v_faces[0] = v_in                               # the first value of v is inlet velocity (INTERSTITIAL)
        dv_dz = -ads * (dqA_dt + dqB_dt)                # equation of evolution of velocity in space by summing 2 equations (1) for O2 and N2
        v_faces[1:] = v_in + np.cumsum(dv_dz) * delta_z # cumsum : cumulative sum

        # Molar flux, J of N2 (B) across cell faces : dispersion + convection terms
        # same reason as velocity ; why we need Nz+1 and not just Nz
        flux_B = np.zeros(Nz + 1)
        flux_B[0] = v_in * 1.0             # with BC z=0 and we assume y_B_feed = 1 (pure N2) because we want to desorb a column concentrated with O2?
        flux_B[1:Nz] = v_faces[1:Nz] * y_B[:-1] - D_L * (y_B[1:] - y_B[:-1]) / delta_z # equation of convective + dispersion term // v_faces[1:Nz] means point with index 1 and stop before index Nz (meaning second last point)
        flux_B[Nz] = v_faces[Nz] * y_B[-1] # with BC z=L and the last point of yB 

        dyB_dt = -(flux_B[1:] - flux_B[:-1]) / delta_z - ads * dqB_dt
        return np.concatenate([dyB_dt, dqA_dt, dqB_dt])

    t_eval = np.linspace(0.0, t_end, n_time)
    sol = solve_ivp(ode_rhs, (0.0, t_end), state0, method="BDF", t_eval=t_eval)

    # to extract the composition of B and A at outlet/final column box (index 49 : Nz-1) and at all time step (:)
    # y_B is at index 49 because we defined earlier that y_B will be in Nz from 0 to 49
    y_B_out = np.clip(sol.y[Nz - 1, :], 0.0, 1.0) # clip to put range for y between 0 to 1 (avoid problem y exceeds 1)
    y_A_out = 1.0 - y_B_out

    # Equilibrium theory for favorable isotherm like Langmuir (applicable to LF too? because the difference is just with parameter m that's approaching 1?)
    # Ideal stoichiometric (equilibrium shock) time for pure N2 feed
    q_A_feed_eq, q_B_feed_eq = multicomponent_Langmuir_Freundlich(0.0, P_bar, a) # again, assuming in feed stream, we have pure N2 and no O2, so P_A = 0 and P_B = P
    t_b_eq = (L_sim / v_in) * (1.0 + ads * q_B_feed_eq)

    # Breakthrough time when outlet N2 reaches 5% (0.05) because usually in industry, we must stop the adsorption step once purity of O2 drops to 95%
    # so we need to regenrate the bed to again restart the adsorption to get high purity O2
    # in equilibrium theory, axial dispersion and instantaneous mass transfer are negligible so we'll get a vertical shock but in real dynamic model,
    # these 2 effects need to be considered
    hit_5percent = np.where(y_B_out >= 0.05)[0]                                  # find the array when condition yB>=0.05 is true and [0] is to unpack actual list of time-step indices
    t_b_5percent = sol.t[hit_5percent[0]] if len(hit_5percent) > 0 else np.nan   # extract the value of time when N2 first reach 0.05 at outlet but to be safe, we put if else, with 'else np.nan' : meaning
    # that if there's nothing in hit_5percent, N2 havent reached 5% yet throughout our simulation (need longer time), so it will return NaN instead of cause problem later

    return sol.t, y_B_out, y_A_out, t_b_eq, t_b_5percent

# Run breakthrough simulations for both adsorbents
t_Ro, yB_Ro, yA_Ro, teq_Ro, t5_Ro = run_breakthrough_sim("Romulus", t_end=1100.0)
t_Re, yB_Re, yA_Re, teq_Re, t5_Re = run_breakthrough_sim("Remus", t_end=1100.0)

# Print breakthrough time using equilibrium theory and 5% N2 at outlet
print("Variable\tRemus\t\tRomulus")
print("-----------------------------------------------")
print(f"Equilibrium time\t{teq_Re:.1f} s\t\t{teq_Ro:.1f} s")
print(f"5% N2 breakthrough\t{t5_Re:.1f} s\t\t{t5_Ro:.1f} s")

# SENSITIVTY ANALYSIS
# this analysis will be carried out only for the better adsorbent : Romulus
# change values of u_in to test what's the effect to the breakthrough time/curve
# results : the slower (lower u), the longer breakthrough time, meaning lower u more efficient, right?
u_cases = [0.5 * u_in, u_in, 2.0 * u_in] # smaller u, u condition, greater u

# run simulation by using the same ode but to 3 different cases of u
u_results = {u: run_breakthrough_sim("Romulus", u_sim=u, t_end=2000.0) for u in u_cases}

# Print u results (unpacking the 5 returned values: t, yB_out, yA_out, teq, t5)
print("\nSensitivity Analysis Results")
print(f"{'Superficial u (m/s)':<22} | {'Interstitial v (m/s)':<22} | {'teq (s)':<10} | {'t5% (s)':<10}")
print("-" * 74)

for u_sim, (_, _, _, teq_u, t5_u) in u_results.items():
    v_sim = u_sim / epsilon
    print(f"{u_sim:<22.3e} | {v_sim:<22.4f} | {teq_u:<10.1f} | {t5_u:<10.1f}")

# change values of L, length of column to test what's the effect to the breakthrough time/curve
L_cases = [0.5 * L, L, 2.0 * L] # smaller L, L current, greater L

# run simulation by using the same ode but to 3 different cases of L
L_results = {l: run_breakthrough_sim("Romulus", L_sim=l, t_end=2000.0) for l in L_cases}

# Print u results (unpacking the 5 returned values: t, yB_out, yA_out, teq, t5)
print(f"\n{'Length of column, L (m)':<22} | {'teq (s)':<10} | {'t5% (s)':<10}")
print("-" * 54)

for L_sim, (_, _, _, teq_L, t5_L) in L_results.items():
    print(f"{L_sim:<22.4f} | {teq_L:<10.1f} | {t5_L:<10.1f}")

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

# Plot sensitivity analysis (different u and different L)