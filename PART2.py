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
P = 100.0e3     # Pa (process pressure)
u_in = 4.864e-3 # m/s (inlet velocity)

# Constant in equation Glueckauf (LDF : Linear Driving Force) = 15De/rp^2
D_e = D_m / tau # m^2/s (effective pore diffusivity)
glueckauf = (15 * D_e) / (r_p ** 2) # 1/s

# Parameters from Part 1 (if combined with part 1 codes, then just use the same name of variables to call later, no need to redefine)
# Langmuir? ns in mol/kg and K in bar^-1
ns_O2_Ro, K_O2_Ro = 4.704886, 0.034592
ns_N2_Ro, K_N2_Ro = 2.541915, 0.440749

ns_O2_Re, K_O2_Re = 6.0977, 1.6875e-2 
ns_N2_Re, K_N2_Re = 2.8344, 9.8015e-2 

# Use multi-component Langmuir isotherm to predict the adsorption because now, we're considering a realistic mixture equilibrium loadings with O2/N2 mixture
# Calculate the mixture equilibrium loadings q_O2* and q_N2* for LDF later
def multicomponent_Langmuir(y_N2, P_bar, ns_O2, K_O2, ns_N2, K_N2):
    y_O2 = 1.0 - y_N2
    p_O2 = y_O2 * P_bar
    p_N2 = y_N2 * P_bar

    denominator = 1.0 + (K_O2 * p_O2) + (K_N2 * p_N2)
    q_O2_eq = (ns_O2 * K_O2 * p_O2) / denominator
    q_N2_eq = (ns_N2 * K_N2 * p_N2) / denominator
    return q_O2_eq, q_N2_eq

